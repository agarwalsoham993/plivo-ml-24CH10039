"""
robust_features.py — Language-agnostic prosodic feature extraction for EOT detection.

Design principles:
  - No raw absolute values; only normalized relative features.
  - Shapes over means: only trajectory (slope, curvature) used.
  - Fail loud: quality gate + NaN sentinels instead of silent zeros.
  - SR-adaptive: all windows expressed in ms, converted at runtime.
  - Language-neutral global prior for early-pause normalization.

All features are computed CAUSALLY: only audio strictly before pause_start is used.
"""

import numpy as np
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Global pitch/energy prior (computed from EN + HI training data combined).
# Used for stabilized normalization when per-turn stats are unreliable.
# ---------------------------------------------------------------------------
GLOBAL_PITCH_MEAN = 155.0    # Hz — center for mixed EN+HI speakers
GLOBAL_PITCH_STD  = 45.0     # Hz — wide enough to cover both genders/languages
GLOBAL_ENERGY_MEAN = -30.0   # dB
GLOBAL_ENERGY_STD  = 12.0    # dB
MIN_VOICED_FOR_TRUST = 20    # voiced frames needed before trusting per-turn stats

# Window sizes in milliseconds — converted to frame counts at runtime
WINDOW_ENERGY_MS = 250       # for energy slope
WINDOW_PITCH_MS  = 350       # for pitch slope / voicing density
WINDOW_HESIT_MS  = 150       # for spectral stability (hesitation proxy)
WINDOW_PRIOR_MS  = 600       # "prior" window for voicing density ratio

HOP_MS = 20                  # hop size in ms (20ms = 4.8x faster pyin vs 10ms)
FRAME_MS = 40                # frame size for pitch (longer = better pitch resolution)


# ---------------------------------------------------------------------------
# Step 1: Robust pitch tracker
# ---------------------------------------------------------------------------

def robust_f0_contour(x: np.ndarray, sr: int,
                      frame_ms: int = FRAME_MS, hop_ms: int = HOP_MS) -> np.ndarray:
    """
    Per-frame F0 in Hz. 0.0 = unvoiced/uncertain.

    Primary: librosa.pyin (probabilistic, handles creaky voice & octave ambiguity).
    Fallback: autocorr_f0 from starter features.py if librosa is unavailable.
    """
    try:
        import librosa
        hop_length   = max(1, int(sr * hop_ms   / 1000))
        frame_length = max(1, int(sr * frame_ms / 1000))
        # pyin fmin/fmax wider than autocorr — covers falsetto, deep male voices, Hindi
        f0, voiced_flag, _ = librosa.pyin(
            x.astype(np.float32),
            fmin=librosa.note_to_hz('C2'),   # ~65 Hz
            fmax=librosa.note_to_hz('C7'),   # ~2093 Hz
            sr=sr,
            hop_length=hop_length,
            frame_length=frame_length,
            fill_na=0.0,   # skip expensive NaN-filling; 0.0 = unvoiced
        )
        f0 = np.where(voiced_flag, f0, 0.0)
        return f0.astype(np.float32)
    except ImportError:
        # Fallback: starter autocorr (frame-by-frame)
        hop   = max(1, int(sr * hop_ms   / 1000))
        fl    = max(1, int(sr * frame_ms / 1000))
        if len(x) < fl:
            return np.zeros(0, dtype=np.float32)
        n   = 1 + (len(x) - fl) // hop
        idx = np.arange(fl)[None, :] + hop * np.arange(n)[:, None]
        frms = x[idx]
        return np.array([_autocorr_f0(f, sr) for f in frms], dtype=np.float32)


def _autocorr_f0(frame: np.ndarray, sr: int,
                 fmin: float = 65.0, fmax: float = 2093.0,
                 voicing_thresh: float = 0.30) -> float:
    """Fallback single-frame autocorrelation pitch estimator."""
    frame = frame - np.mean(frame)
    if np.max(np.abs(frame)) < 1e-4:
        return 0.0
    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo = max(1, int(sr / fmax))
    hi = min(int(sr / fmin), len(ac) - 1)
    if hi <= lo:
        return 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] < voicing_thresh:
        return 0.0
    return float(sr / lag)


# ---------------------------------------------------------------------------
# Step 2: Segment quality gate
# ---------------------------------------------------------------------------

def segment_quality(seg: np.ndarray, sr: int,
                    min_voiced_frames: int = 5) -> tuple[bool, float, int]:
    """
    Returns (is_usable, snr_db, n_voiced).

    Gate fails if:
      - Fewer than min_voiced_frames voiced frames detected.
      - SNR (voiced-region energy vs. silent-region energy) < 3 dB.

    If gate fails, callers should return NaN sentinel features.
    HistGradientBoostingClassifier handles NaN natively.
    """
    f0       = robust_f0_contour(seg, sr)
    energies = _frame_energy_db(seg, sr)

    # Align lengths (pyin may produce slightly different count)
    n = min(len(f0), len(energies))
    f0       = f0[:n]
    energies = energies[:n]

    n_voiced = int((f0 > 0).sum())
    voiced_e = energies[f0 > 0]
    silent_e = energies[f0 == 0]

    if len(voiced_e) == 0 or len(silent_e) == 0:
        snr = 0.0
    else:
        snr = float(np.mean(voiced_e) - np.mean(silent_e))

    is_usable = (n_voiced >= min_voiced_frames) and (snr > 3.0)
    return is_usable, snr, n_voiced


def _frame_energy_db(x: np.ndarray, sr: int,
                     frame_ms: int = 25, hop_ms: int = HOP_MS) -> np.ndarray:
    """Short-time energy per frame, in dB."""
    hop = max(1, int(sr * hop_ms   / 1000))
    fl  = max(1, int(sr * frame_ms / 1000))
    if len(x) < fl:
        return np.array([], dtype=np.float32)
    n   = 1 + (len(x) - fl) // hop
    idx = np.arange(fl)[None, :] + hop * np.arange(n)[:, None]
    fr  = x[idx]
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    return (20 * np.log10(rms + 1e-12)).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 3: Stabilized causal speaker normalization
# ---------------------------------------------------------------------------

def stabilized_normalize(value: float,
                         turn_mean: float, turn_std: float, n_samples: int,
                         global_mean: float, global_std: float,
                         min_trust: int = MIN_VOICED_FOR_TRUST) -> float:
    """
    Blend per-turn stats with global prior.
    alpha = 0 → use global prior entirely (early in turn).
    alpha = 1 → use per-turn stats entirely (once enough samples seen).
    """
    alpha = min(1.0, n_samples / min_trust)
    mu    = alpha * turn_mean + (1 - alpha) * global_mean
    std   = alpha * turn_std  + (1 - alpha) * global_std
    return (value - mu) / (std + 1e-6)


def _ms_to_frames(ms: int, hop_ms: int = HOP_MS) -> int:
    return max(1, int(ms / hop_ms))


# ---------------------------------------------------------------------------
# Step 4: Full feature extraction (SR-adaptive, 9 features)
# ---------------------------------------------------------------------------

def extract_features_v2(x: np.ndarray, sr: int,
                        pause_start: float, pause_index: int,
                        speaker_state: dict) -> np.ndarray:
    """
    Extract 9 language-agnostic features from the audio segment strictly
    before pause_start.

    speaker_state is mutated in-place to accumulate per-turn running stats
    (causal — only uses information available at decision time).

    Returns np.float32 array of shape (9,).
    NaN is returned for any feature that cannot be reliably computed.
    HistGradientBoostingClassifier handles NaN natively via its own missing-value
    branch logic.

    Feature layout:
      [0] pitch_slope          — terminal F0 trajectory (normalized, voiced only)
      [1] energy_slope         — energy decay rate (normalized)
      [2] voicing_density_ratio— final/prior voiced fraction
      [3] energy_final         — normalized energy in last frame
      [4] pitch_final          — normalized F0 in last voiced frame
      [5] spectral_stability   — language-neutral hesitation signal
      [6] pause_index          — position of this pause in the turn
      [7] turn_fraction        — soft-normalized position in estimated turn
      [8] n_voiced_fraction    — fraction of voiced frames in last WINDOW_PITCH_MS
    """
    NAN = np.float32(np.nan)
    nan_vec = np.full(9, np.nan, dtype=np.float32)

    # --- Audio segment (causal: strictly before pause_start) ---------------
    end   = int(pause_start * sr)
    start = max(0, end - int(1.5 * sr))   # up to 1.5s of context
    seg   = x[start:end]

    if len(seg) < sr // 10:   # < 100ms — give up entirely
        return nan_vec

    # --- Segment quality gate ----------------------------------------------
    is_usable, snr_db, n_voiced_total = segment_quality(seg, sr)
    if not is_usable:
        # Still compute structure features (pause_index, turn_fraction) —
        # those don't depend on audio quality.
        turn_fraction = pause_start / (pause_start + 2.0)
        partial = nan_vec.copy()
        partial[6] = np.float32(pause_index)
        partial[7] = np.float32(turn_fraction)
        return partial

    # --- Compute full pitch and energy contours ----------------------------
    f0       = robust_f0_contour(seg, sr)
    energies = _frame_energy_db(seg, sr)

    # Align lengths
    n = min(len(f0), len(energies))
    f0       = f0[:n]
    energies = energies[:n]

    voiced_mask = f0 > 0
    voiced_f0   = f0[voiced_mask]

    # --- Update per-turn running stats (causal) ----------------------------
    if 'pitch_values' not in speaker_state:
        speaker_state['pitch_values']  = []
        speaker_state['energy_values'] = []

    speaker_state['pitch_values'].extend(voiced_f0.tolist())
    speaker_state['energy_values'].extend(energies.tolist())

    turn_pitches  = np.array(speaker_state['pitch_values'],  dtype=np.float32)
    turn_energies = np.array(speaker_state['energy_values'], dtype=np.float32)

    turn_pitch_mean  = float(turn_pitches.mean())  if len(turn_pitches)  > 0 else GLOBAL_PITCH_MEAN
    turn_pitch_std   = float(turn_pitches.std())   if len(turn_pitches)  > 1 else GLOBAL_PITCH_STD
    turn_energy_mean = float(turn_energies.mean()) if len(turn_energies) > 0 else GLOBAL_ENERGY_MEAN
    turn_energy_std  = float(turn_energies.std())  if len(turn_energies) > 1 else GLOBAL_ENERGY_STD

    n_voiced_seen = len(turn_pitches)

    # --- Window frame counts (SR-adaptive) ---------------------------------
    n_pitch  = _ms_to_frames(WINDOW_PITCH_MS,  HOP_MS)   # ~35 frames at 16kHz
    n_energy = _ms_to_frames(WINDOW_ENERGY_MS, HOP_MS)   # ~25 frames
    n_hesit  = _ms_to_frames(WINDOW_HESIT_MS,  HOP_MS)   # ~15 frames
    n_prior  = _ms_to_frames(WINDOW_PRIOR_MS,  HOP_MS)   # ~60 frames

    # --- Feature 0: pitch_slope (last WINDOW_PITCH_MS, voiced only) --------
    recent_f0_all = f0[-n_pitch:]
    recent_voiced = recent_f0_all[recent_f0_all > 0]

    pitch_slope = NAN
    if len(recent_voiced) >= 3:
        norm_voiced = np.array([
            stabilized_normalize(v, turn_pitch_mean, turn_pitch_std, n_voiced_seen,
                                 GLOBAL_PITCH_MEAN, GLOBAL_PITCH_STD)
            for v in recent_voiced
        ], dtype=np.float32)
        slope, _, _, _, _ = sp_stats.linregress(np.arange(len(norm_voiced)), norm_voiced)
        pitch_slope = np.float32(slope)

    # --- Feature 1: energy_slope (last WINDOW_ENERGY_MS) ------------------
    recent_e = energies[-n_energy:]
    energy_slope = NAN
    if len(recent_e) >= 3:
        norm_e = np.array([
            stabilized_normalize(v, turn_energy_mean, turn_energy_std, n_voiced_seen,
                                 GLOBAL_ENERGY_MEAN, GLOBAL_ENERGY_STD)
            for v in recent_e
        ], dtype=np.float32)
        slope, _, _, _, _ = sp_stats.linregress(np.arange(len(norm_e)), norm_e)
        energy_slope = np.float32(slope)

    # --- Feature 2: voicing_density_ratio ----------------------------------
    final_voiced  = float(np.mean(voiced_mask[-n_pitch:])) if n >= n_pitch else float(np.mean(voiced_mask))
    prior_voiced  = float(np.mean(voiced_mask[-n_prior:-n_pitch])) if n >= n_prior else final_voiced
    voicing_ratio = np.float32(final_voiced / (prior_voiced + 1e-6))

    # --- Feature 3: energy_final (last frame, normalized) ------------------
    energy_final = NAN
    if len(energies) > 0:
        energy_final = np.float32(
            stabilized_normalize(float(energies[-1]), turn_energy_mean, turn_energy_std,
                                 n_voiced_seen, GLOBAL_ENERGY_MEAN, GLOBAL_ENERGY_STD)
        )

    # --- Feature 4: pitch_final (last voiced frame, normalized) ------------
    pitch_final = NAN
    if len(voiced_f0) > 0:
        pitch_final = np.float32(
            stabilized_normalize(float(voiced_f0[-1]), turn_pitch_mean, turn_pitch_std,
                                 n_voiced_seen, GLOBAL_PITCH_MEAN, GLOBAL_PITCH_STD)
        )

    # --- Feature 5: spectral_stability (language-neutral hesitation) -------
    # High value = stable centroid = held vowel = hesitation signal.
    # Low value = changing spectrum = consonants / phrase boundary.
    # NaN when the segment is silent or too short.
    spectral_stability = NAN
    hesit_samples = int(sr * WINDOW_HESIT_MS / 1000)
    seg_hesit = seg[max(0, len(seg) - hesit_samples):]
    # Only compute if segment has meaningful energy
    if len(seg_hesit) > 0 and np.max(np.abs(seg_hesit)) > 1e-4:
        hop_h = max(1, int(sr * HOP_MS / 1000))
        fl_h  = max(1, int(sr * 25     / 1000))   # 25ms frames for spectral
        if len(seg_hesit) >= fl_h:
            n_fr  = 1 + (len(seg_hesit) - fl_h) // hop_h
            idx   = np.arange(fl_h)[None, :] + hop_h * np.arange(n_fr)[:, None]
            frms  = seg_hesit[idx].astype(np.float64)
            freqs = np.fft.rfftfreq(fl_h, 1.0 / sr)
            if n_fr >= 3:
                centroids = []
                for f in frms:
                    mag = np.abs(np.fft.rfft(f))
                    denom = mag.sum() + 1e-8
                    centroids.append(float(np.sum(mag * freqs) / denom))
                centroids = np.array(centroids)
                var_c = np.var(centroids)
                # Normalize by mean centroid to make it speaker/SR independent
                mean_c = np.mean(centroids) + 1e-2
                spectral_stability = np.float32(mean_c / (var_c + 1e-4))

    # --- Feature 6: pause_index (turn structure) ---------------------------
    feat_pause_index = np.float32(pause_index)

    # --- Feature 7: turn_fraction (soft-normalized position) ---------------
    turn_fraction = np.float32(pause_start / (pause_start + 2.0))

    # --- Feature 8: n_voiced_fraction (last WINDOW_PITCH_MS) --------------
    if n >= n_pitch:
        n_voiced_frac = np.float32(float(voiced_mask[-n_pitch:].sum()) / n_pitch)
    else:
        n_voiced_frac = np.float32(float(voiced_mask.sum()) / max(1, len(voiced_mask)))

    return np.array([
        pitch_slope,
        energy_slope,
        voicing_ratio,
        energy_final,
        pitch_final,
        spectral_stability,
        feat_pause_index,
        turn_fraction,
        n_voiced_frac,
    ], dtype=np.float32)
