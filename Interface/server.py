import os
import sys
import json
import csv
import numpy as np
import soundfile as sf
import joblib
from flask import Flask, jsonify, request, send_from_directory, render_template

# Add parent directory to path to import robust_features
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)

from robust_features import extract_features_v2

app = Flask(__name__, static_folder='static', template_folder='templates')

# Paths to models and thresholds
MODEL_COMBINED_PATH = os.path.join(PARENT_DIR, "model.joblib")
MODEL_EN_PATH = os.path.join(PARENT_DIR, "model_en_only.joblib")
THRESH_COMBINED_PATH = os.path.join(PARENT_DIR, "thresholds.json")
THRESH_EN_PATH = os.path.join(PARENT_DIR, "thresh_en.json")

# Data directories
DATA_DIR_EN = os.path.join(PARENT_DIR, "eot_handout", "eot_data", "english")
DATA_DIR_HI = os.path.join(PARENT_DIR, "eot_handout", "eot_data", "hindi")

# Global variables to store loaded models and thresholds
models = {}
thresholds = {}

def load_models_and_thresholds():
    global models, thresholds
    
    # Load Combined (EN+HI) Model
    if os.path.exists(MODEL_COMBINED_PATH):
        try:
            models['combined'] = joblib.load(MODEL_COMBINED_PATH)
            print("Loaded Combined model successfully.")
        except Exception as e:
            print(f"Error loading Combined model: {e}")
    else:
        print(f"Combined model not found at {MODEL_COMBINED_PATH}")
        
    # Load English-Only Model
    if os.path.exists(MODEL_EN_PATH):
        try:
            models['en_only'] = joblib.load(MODEL_EN_PATH)
            print("Loaded English-only model successfully.")
        except Exception as e:
            print(f"Error loading English-only model: {e}")
    else:
        print(f"English-only model not found at {MODEL_EN_PATH}")
        
    # Load Combined Thresholds
    if os.path.exists(THRESH_COMBINED_PATH):
        try:
            with open(THRESH_COMBINED_PATH, 'r') as f:
                thresholds['combined'] = json.load(f)
            print("Loaded Combined thresholds successfully.")
        except Exception as e:
            print(f"Error loading Combined thresholds: {e}")
    else:
        # Default fallback
        thresholds['combined'] = {"threshold": 0.38, "delay": 0.68}
        
    # Load English-Only Thresholds
    if os.path.exists(THRESH_EN_PATH):
        try:
            with open(THRESH_EN_PATH, 'r') as f:
                thresholds['en_only'] = json.load(f)
            print("Loaded English-only thresholds successfully.")
        except Exception as e:
            print(f"Error loading English-only thresholds: {e}")
    else:
        # Default fallback
        thresholds['en_only'] = {"threshold": 0.05, "delay": 0.95}

# Initialize models
load_models_and_thresholds()

# Helper to load dataset samples
def load_dataset_samples(data_dir, lang):
    labels_path = os.path.join(data_dir, "labels.csv")
    if not os.path.exists(labels_path):
        return {}
        
    samples = {}
    try:
        with open(labels_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                turn_id = row['turn_id']
                audio_file = row['audio_file']
                pause_index = int(row['pause_index'])
                pause_start = float(row['pause_start'])
                pause_end = float(row['pause_end'])
                label = row['label']
                
                if turn_id not in samples:
                    samples[turn_id] = {
                        "turn_id": turn_id,
                        "language": lang,
                        "audio_file": audio_file,
                        "pauses": []
                    }
                    
                samples[turn_id]['pauses'].append({
                    "pause_index": pause_index,
                    "pause_start": pause_start,
                    "pause_end": pause_end,
                    "duration": round(pause_end - pause_start, 3),
                    "label": label
                })
        
        # Sort pauses by pause_index for each turn
        for turn_id in samples:
            samples[turn_id]['pauses'].sort(key=lambda x: x['pause_index'])
            
    except Exception as e:
        print(f"Error loading samples from {labels_path}: {e}")
        
    return samples

# API: Get available samples
@app.route('/api/samples', methods=['GET'])
def get_samples():
    en_samples = load_dataset_samples(DATA_DIR_EN, "english")
    hi_samples = load_dataset_samples(DATA_DIR_HI, "hindi")
    
    # Merge samples
    all_samples = {**en_samples, **hi_samples}
    return jsonify({
        "status": "success",
        "count": len(all_samples),
        "samples": all_samples
    })

# API: Serve dataset audio files
@app.route('/api/audio/<lang>/<filename>', methods=['GET'])
def serve_audio(lang, filename):
    if lang == 'english':
        directory = os.path.join(DATA_DIR_EN, "audio")
    elif lang == 'hindi':
        directory = os.path.join(DATA_DIR_HI, "audio")
    else:
        return jsonify({"error": "Invalid language"}), 400
        
    return send_from_directory(directory, filename)

# FEATURE NAMES AND DESCRIPTIONS
FEATURE_METADATA = [
    {"name": "pitch_slope", "label": "Pitch Slope", "desc": "Terminal F0 pitch trajectory (slope). Negative indicates falling pitch (often associated with statement completion/EOT)."},
    {"name": "energy_slope", "label": "Energy Slope", "desc": "Energy decay rate over the last 250ms. Negative indicates fading volume (associated with phrase endings)."},
    {"name": "voicing_density_ratio", "label": "Voicing Density Ratio", "desc": "Ratio of voiced frames in final window compared to prior window. Drop indicates silence or breathiness."},
    {"name": "energy_final", "label": "Energy Final", "desc": "Normalized energy in the last frame. Low values mean the speaker has gone quiet."},
    {"name": "pitch_final", "label": "Pitch Final", "desc": "Normalized F0 pitch in the last voiced frame. Helps differentiate between high pitch (question/hold) and low pitch (drop)."},
    {"name": "spectral_stability", "label": "Spectral Stability", "desc": "Hesitation signal (spectral centroid stability). High values indicate stable vowels (like 'uh', 'um' - holding turn)."},
    {"name": "pause_index", "label": "Pause Index", "desc": "Number of pauses in the current turn. More pauses can mean the speaker is nearing the end of their turn."},
    {"name": "turn_fraction", "label": "Turn Fraction", "desc": "Soft-normalized time position in the turn. The longer the speaker talks, the more likely they are to finish."},
    {"name": "n_voiced_fraction", "label": "Voiced Fraction", "desc": "Fraction of voiced frames in the last 350ms. Low value means the speaker stopped phonating."}
]

def make_prediction(features_array, pause_dur):
    """
    Run predictions for all loaded models on a single feature vector.
    features_array shape: (9,)
    """
    results = {}
    
    # Prepare features matrix: shape (1, 9)
    X = np.array([features_array], dtype=np.float32)
    
    for model_key in ['combined', 'en_only']:
        model = models.get(model_key)
        thresh_info = thresholds.get(model_key, {"threshold": 0.5, "delay": 1.0})
        
        if model is None:
            results[model_key] = {
                "available": False,
                "error": "Model not loaded"
            }
            continue
            
        # Predict probability of EOT
        try:
            p_eot = float(model.predict_proba(X)[0, 1])
        except Exception as e:
            results[model_key] = {
                "available": False,
                "error": f"Prediction failed: {e}"
            }
            continue
            
        threshold = thresh_info['threshold']
        delay = thresh_info['delay']
        
        # Decision logic: 
        # An EOT is fired if the probability is above threshold AND the pause duration is longer than the delay.
        # Otherwise, the model holds.
        fire_eot = (p_eot >= threshold) and (pause_dur >= delay)
        decision = "eot" if fire_eot else "hold"
        
        results[model_key] = {
            "available": True,
            "p_eot": round(p_eot, 4),
            "threshold": round(threshold, 3),
            "delay_ms": round(delay * 1000, 1),
            "pause_duration_ms": round(pause_dur * 1000, 1),
            "decision": decision,
            "conditions": {
                "prob_ok": p_eot >= threshold,
                "delay_ok": pause_dur >= delay
            }
        }
        
    return results

# API: Predict on predefined turn and pause index (supports causal sequence)
@app.route('/api/predict/predefined', methods=['POST'])
def predict_predefined():
    data = request.json or {}
    lang = data.get('language')
    turn_id = data.get('turn_id')
    target_pause_index = data.get('pause_index')
    
    if not lang or not turn_id or target_pause_index is None:
        return jsonify({"error": "Missing parameters (language, turn_id, pause_index)"}), 400
        
    # Find turn records in dataset
    data_dir = DATA_DIR_EN if lang == 'english' else DATA_DIR_HI
    labels_path = os.path.join(data_dir, "labels.csv")
    
    if not os.path.exists(labels_path):
        return jsonify({"error": f"Labels file not found for {lang}"}), 404
        
    pauses_in_turn = []
    audio_file_path = None
    
    # Read the labels and collect all pauses for this turn_id
    try:
        with open(labels_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['turn_id'] == turn_id:
                    audio_file_path = os.path.join(data_dir, row['audio_file'])
                    pauses_in_turn.append({
                        "pause_index": int(row['pause_index']),
                        "pause_start": float(row['pause_start']),
                        "pause_end": float(row['pause_end']),
                        "duration": float(row['pause_end']) - float(row['pause_start']),
                        "label": row['label']
                    })
    except Exception as e:
        return jsonify({"error": f"Error reading dataset: {e}"}), 500
        
    if not pauses_in_turn:
        return jsonify({"error": f"Turn ID {turn_id} not found in {lang} labels"}), 404
        
    # Sort pauses by pause_index
    pauses_in_turn.sort(key=lambda x: x['pause_index'])
    
    # Find the target pause
    target_pause = next((p for p in pauses_in_turn if p['pause_index'] == target_pause_index), None)
    if not target_pause:
        return jsonify({"error": f"Pause index {target_pause_index} not found for turn {turn_id}"}), 404
        
    # Load audio file
    if not os.path.exists(audio_file_path):
        return jsonify({"error": f"Audio file not found: {audio_file_path}"}), 404
        
    try:
        x, sr = sf.read(audio_file_path, dtype="float32", always_2d=False)
    except Exception as e:
        return jsonify({"error": f"Failed to read audio file: {e}"}), 500
        
    # Sequential feature extraction up to target_pause_index to preserve causal speaker_state
    speaker_state = {}
    features = None
    
    for p in pauses_in_turn:
        p_idx = p['pause_index']
        if p_idx > target_pause_index:
            break
            
        try:
            feat = extract_features_v2(x, sr, p['pause_start'], p_idx, speaker_state)
            if p_idx == target_pause_index:
                features = feat
        except Exception as e:
            return jsonify({"error": f"Feature extraction failed at pause index {p_idx}: {e}"}), 500
            
    if features is None:
        return jsonify({"error": "Failed to extract features for target pause"}), 500
        
    # Run predictions on the extracted features
    predictions = make_prediction(features, target_pause['duration'])
    
    # Format features for output
    formatted_features = []
    for idx, meta in enumerate(FEATURE_METADATA):
        val = features[idx]
        formatted_features.append({
            "name": meta['name'],
            "label": meta['label'],
            "desc": meta['desc'],
            "value": None if np.isnan(val) else float(val)
        })
        
    return jsonify({
        "status": "success",
        "turn_id": turn_id,
        "pause_index": target_pause_index,
        "pause_start": target_pause['pause_start'],
        "pause_end": target_pause['pause_end'],
        "pause_duration": target_pause['duration'],
        "ground_truth_label": target_pause['label'],
        "features": formatted_features,
        "predictions": predictions
    })

# API: Predict all pauses in a turn simultaneously (for turn-level summary)
@app.route('/api/predict/turn', methods=['POST'])
def predict_turn():
    data = request.json or {}
    lang = data.get('language')
    turn_id = data.get('turn_id')
    
    if not lang or not turn_id:
        return jsonify({"error": "Missing parameters (language, turn_id)"}), 400
        
    data_dir = DATA_DIR_EN if lang == 'english' else DATA_DIR_HI
    labels_path = os.path.join(data_dir, "labels.csv")
    
    if not os.path.exists(labels_path):
        return jsonify({"error": f"Labels file not found for {lang}"}), 404
        
    pauses_in_turn = []
    audio_file_path = None
    
    try:
        with open(labels_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['turn_id'] == turn_id:
                    audio_file_path = os.path.join(data_dir, row['audio_file'])
                    pauses_in_turn.append({
                        "pause_index": int(row['pause_index']),
                        "pause_start": float(row['pause_start']),
                        "pause_end": float(row['pause_end']),
                        "duration": float(row['pause_end']) - float(row['pause_start']),
                        "label": row['label']
                    })
    except Exception as e:
        return jsonify({"error": f"Error reading dataset: {e}"}), 500
        
    if not pauses_in_turn:
        return jsonify({"error": f"Turn ID {turn_id} not found in {lang} labels"}), 404
        
    pauses_in_turn.sort(key=lambda x: x['pause_index'])
    
    if not os.path.exists(audio_file_path):
        return jsonify({"error": f"Audio file not found: {audio_file_path}"}), 404
        
    try:
        x, sr = sf.read(audio_file_path, dtype="float32", always_2d=False)
    except Exception as e:
        return jsonify({"error": f"Failed to read audio file: {e}"}), 500
        
    speaker_state = {}
    results = []
    
    for p in pauses_in_turn:
        p_idx = p['pause_index']
        try:
            feat = extract_features_v2(x, sr, p['pause_start'], p_idx, speaker_state)
            predictions = make_prediction(feat, p['duration'])
            
            # Format features for output
            formatted_features = []
            for idx, meta in enumerate(FEATURE_METADATA):
                val = feat[idx]
                formatted_features.append({
                    "name": meta['name'],
                    "label": meta['label'],
                    "desc": meta['desc'],
                    "value": None if np.isnan(val) else float(val)
                })
                
            results.append({
                "pause_index": p_idx,
                "pause_start": p['pause_start'],
                "pause_end": p['pause_end'],
                "duration": p['duration'],
                "label": p['label'],
                "features": formatted_features,
                "predictions": predictions
            })
        except Exception as e:
            return jsonify({"error": f"Feature extraction failed at pause index {p_idx}: {e}"}), 500
            
    return jsonify({
        "status": "success",
        "turn_id": turn_id,
        "language": lang,
        "pauses": results
    })

# API: Predict on custom uploaded WAV audio
@app.route('/api/predict/upload', methods=['POST'])

def predict_upload():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
        
    audio_file = request.files['audio']
    pause_start = request.form.get('pause_start', type=float)
    pause_duration = request.form.get('pause_duration', default=0.5, type=float)
    
    # Save file temporarily
    temp_filename = "temp_upload.wav"
    temp_path = os.path.join(app.root_path, temp_filename)
    audio_file.save(temp_path)
    
    try:
        x, sr = sf.read(temp_path, dtype="float32", always_2d=False)
        duration = len(x) / sr
        
        # If pause_start is not specified, default to the end of the audio file
        if pause_start is None or pause_start <= 0:
            pause_start = duration
            
        # Extract features for pause_index = 0 with a clean speaker state
        speaker_state = {}
        features = extract_features_v2(x, sr, pause_start, 0, speaker_state)
        
        # Run predictions
        predictions = make_prediction(features, pause_duration)
        
        # Format features for output
        formatted_features = []
        for idx, meta in enumerate(FEATURE_METADATA):
            val = features[idx]
            formatted_features.append({
                "name": meta['name'],
                "label": meta['label'],
                "desc": meta['desc'],
                "value": None if np.isnan(val) else float(val)
            })
            
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return jsonify({
            "status": "success",
            "audio_duration": duration,
            "pause_start": pause_start,
            "pause_duration": pause_duration,
            "features": formatted_features,
            "predictions": predictions
        })
        
    except Exception as e:
        # Clean up temp file in case of error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": f"Error processing uploaded audio: {e}"}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Use environment port if defined, else 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
