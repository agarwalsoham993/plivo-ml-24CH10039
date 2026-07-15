"""
predict.py — Final inference script for EOT detection.

Usage (matches baseline.py interface exactly):
    python predict.py --data_dir eot_data/english --out predictions.csv

Loads model.joblib and thresholds.json from the same directory as this script.
No refitting — pure inference on unseen data.
"""
import argparse
import csv
import json
import os
import sys

import joblib
import numpy as np
import soundfile as sf
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robust_features import extract_features_v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Directory containing labels.csv and audio/")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model",  default=None,
                    help="Path to model.joblib (default: same dir as predict.py)")
    ap.add_argument("--thresh", default=None,
                    help="Path to thresholds.json (default: same dir as predict.py)")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path  = args.model  or os.path.join(script_dir, "model.joblib")
    thresh_path = args.thresh or os.path.join(script_dir, "thresholds.json")

    # --- Load model + thresholds ---
    if not os.path.exists(model_path):
        raise SystemExit(f"Model not found: {model_path}\n"
                         "Run train_v2.py first to generate model.joblib")
    calibrated = joblib.load(model_path)

    if not os.path.exists(thresh_path):
        raise SystemExit(f"Thresholds not found: {thresh_path}\n"
                         "Run train_v2.py first to generate thresholds.json")
    with open(thresh_path) as f:
        thresh_data = json.load(f)

    print(f"Loaded model from {model_path}")
    print(f"Operating point: T={thresh_data['threshold']:.3f}, "
          f"D={thresh_data['delay']*1000:.0f}ms")

    # --- Load labels ---
    labels_path = os.path.join(args.data_dir, "labels.csv")
    rows = list(csv.DictReader(open(labels_path)))
    print(f"Loaded {len(rows)} pauses from {labels_path}")

    # --- Build records ---
    records = []
    for r in rows:
        records.append({
            "turn_id":     r["turn_id"],
            "pause_index": int(r["pause_index"]),
            "pause_start": float(r["pause_start"]),
            "audio_path":  os.path.join(args.data_dir, r["audio_file"]),
        })

    # --- Extract features (causal: per-turn speaker state) ---
    audio_cache = {}
    X = []
    keys = []

    by_turn = defaultdict(list)
    for rec in records:
        by_turn[rec["turn_id"]].append(rec)

    for tid, turn_recs in by_turn.items():
        state = {}
        for rec in sorted(turn_recs, key=lambda r: r["pause_index"]):
            path = rec["audio_path"]
            if path not in audio_cache:
                audio_cache[path] = sf.read(path, dtype="float32", always_2d=False)
            x, sr = audio_cache[path]
            feat = extract_features_v2(x, sr, rec["pause_start"],
                                       rec["pause_index"], state)
            X.append(feat)
            keys.append((rec["turn_id"], rec["pause_index"]))

    X = np.array(X, dtype=np.float32)
    nan_frac = np.isnan(X).mean()
    print(f"Feature matrix: {X.shape}, NaN fraction: {nan_frac:.3f}")

    # --- Predict ---
    p = calibrated.predict_proba(X)[:, 1]

    # --- Write predictions ---
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), p_eot in zip(keys, p):
            w.writerow([tid, pi, f"{p_eot:.4f}"])

    print(f"Wrote {len(keys)} predictions → {args.out}")


if __name__ == "__main__":
    main()
