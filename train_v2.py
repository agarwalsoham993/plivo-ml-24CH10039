"""
train_v2.py — Hindi-weighted training with Platt calibration and threshold stability.

Usage:
    python train_v2.py \
        --train_en references/eot_handout/eot_data/english \
        --train_hi references/eot_handout/eot_data/hindi \
        --out_model model.joblib \
        --out_thresh thresholds.json

    # English-only (for cross-language validation):
    python train_v2.py \
        --train_en references/eot_handout/eot_data/english \
        --out_model model_en_only.joblib \
        --out_thresh thresh_en.json

Outputs:
    model.joblib      — CalibratedClassifierCV (HistGBM + Platt sigmoid)
    thresholds.json   — Operating point (T, D), fold stability stats
    train_v2_en.csv   — In-sample predictions for English (for score.py)
    train_v2_hi.csv   — In-sample predictions for Hindi (if --train_hi given)
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(__file__))
from robust_features import extract_features_v2


# ---------------------------------------------------------------------------
# Scoring helpers (mirrors score.py logic, no file I/O)
# ---------------------------------------------------------------------------
TIMEOUT_S  = 1.6
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 3)
DELAYS     = np.round(np.arange(0.10, 1.65, 0.05), 3)


def _evaluate(pauses, threshold, delay):
    turns_cut = set()
    turn_ids  = set()
    latencies = []
    for pz in pauses:
        turn_ids.add(pz["turn_id"])
        fires = pz["p"] >= threshold
        if pz["label"] == "hold":
            if fires and delay < pz["dur"]:
                turns_cut.add(pz["turn_id"])
        else:
            latencies.append(delay if fires else TIMEOUT_S)
    cutoff_rate = len(turns_cut) / max(1, len(turn_ids))
    return cutoff_rate, float(np.mean(latencies)) if latencies else TIMEOUT_S


def find_best_threshold(pauses_with_p, budget=0.05):
    """
    Returns (best_T, best_D, best_latency) given pauses list with 'p' field set.
    """
    best = None
    for t in THRESHOLDS:
        for d in DELAYS:
            cut, lat = _evaluate(pauses_with_p, t, d)
            if cut <= budget and (best is None or lat < best[2]):
                best = (float(t), float(d), float(lat))
    if best is None:
        best = (1.0, TIMEOUT_S, TIMEOUT_S)
    return best


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir, lang_prefix):
    """
    Load all pauses from a data directory.
    Returns list of dicts with keys: turn_id, pause_index, pause_start,
    pause_dur, label, audio_path, lang.
    """
    labels_path = os.path.join(data_dir, "labels.csv")
    rows = list(csv.DictReader(open(labels_path)))
    records = []
    for r in rows:
        records.append({
            "turn_id":    r["turn_id"],
            "pause_index": int(r["pause_index"]),
            "pause_start": float(r["pause_start"]),
            "pause_dur":   float(r["pause_end"]) - float(r["pause_start"]),
            "label":       r["label"],
            "audio_path":  os.path.join(data_dir, r["audio_file"]),
            "lang":        lang_prefix,
        })
    return records


FEATURE_CACHE_PATH = ".feature_cache.npz"


def extract_all_features(records, use_cache=True):
    """
    Extract features for all records, maintaining causal per-turn state.
    Caches results to .feature_cache.npz to avoid re-running librosa.pyin.
    Returns (X, y, groups, pauses_meta).
    """
    # Try loading cache — keyed by (turn_id, pause_index)
    if use_cache and os.path.exists(FEATURE_CACHE_PATH):
        print("  Loading features from cache...")
        cache_data = np.load(FEATURE_CACHE_PATH, allow_pickle=True)
        X_cached   = cache_data["X"]
        y_cached   = cache_data["y"]
        groups_c   = list(cache_data["groups"])
        meta_c     = list(cache_data["meta"])
        if len(X_cached) == len(records):
            print(f"  Cache hit: {len(X_cached)} records")
            return X_cached.astype(np.float32), y_cached, groups_c, meta_c
        print("  Cache stale (record count mismatch), re-extracting...")

    audio_cache = {}
    X, y, groups, meta = [], [], [], []

    # Group by turn_id to maintain causal state
    by_turn = defaultdict(list)
    for rec in records:
        by_turn[rec["turn_id"]].append(rec)

    n_done = 0
    for tid, turn_recs in by_turn.items():
        state = {}   # per-turn speaker state
        for rec in sorted(turn_recs, key=lambda r: r["pause_index"]):
            path = rec["audio_path"]
            if path not in audio_cache:
                audio_cache[path] = sf.read(path, dtype="float32", always_2d=False)
            x, sr = audio_cache[path]
            feat = extract_features_v2(x, sr, rec["pause_start"],
                                       rec["pause_index"], state)
            X.append(feat)
            y.append(1 if rec["label"] == "eot" else 0)
            groups.append(rec["turn_id"])
            meta.append({
                "turn_id":     rec["turn_id"],
                "pause_index": rec["pause_index"],
                "label":       rec["label"],
                "dur":         rec["pause_dur"],
            })
            n_done += 1
            if n_done % 50 == 0:
                print(f"  ... {n_done}/{len(records)} pauses")

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y)
    np.savez(FEATURE_CACHE_PATH,
             X=X_arr, y=y_arr,
             groups=np.array(groups),
             meta=np.array(meta, dtype=object))
    print(f"  Saved feature cache → {FEATURE_CACHE_PATH}")
    return X_arr, y_arr, groups, meta


# ---------------------------------------------------------------------------
# Sample weights (Hindi = 2x)
# ---------------------------------------------------------------------------

def build_sample_weights(groups, lang_map):
    """lang_map: turn_id -> lang prefix ('en' or 'hi')."""
    weights = np.ones(len(groups), dtype=np.float32)
    for i, g in enumerate(groups):
        if lang_map.get(g, "en") == "hi":
            weights[i] = 2.0
    return weights


# ---------------------------------------------------------------------------
# Cross-validation loop
# ---------------------------------------------------------------------------

def train_with_cv(X, y, groups, meta, lang_map, n_splits=5):
    """
    GroupKFold CV: fit HistGBM + Platt per fold, collect thresholds.
    meta is guaranteed parallel to X (same order from extract_all_features).
    Returns (fold_results, T_per_fold, D_per_fold).
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []
    T_per_fold = []
    D_per_fold = []

    weights_all = build_sample_weights(groups, lang_map)

    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
        w_tr = weights_all[tr_idx]

        # Fit HistGBM
        clf = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=8,
            random_state=42,
        )
        clf.fit(X_tr, y_tr, sample_weight=w_tr)

        # Platt calibration (2 params — safe for small folds)
        # sklearn 1.9+: wrap already-fitted clf in FrozenEstimator
        frozen = FrozenEstimator(clf)
        calibrated = CalibratedClassifierCV(
            estimator=frozen,
            method='sigmoid',
        )
        calibrated.fit(X_va, y_va)

        # Predict on validation fold
        p_va = calibrated.predict_proba(X_va)[:, 1]

        # Build pauses list for threshold search (meta is parallel to X)
        va_pauses = []
        for i, idx in enumerate(va_idx):
            m = meta[idx] if isinstance(meta[idx], dict) else dict(meta[idx])
            va_pauses.append({
                "turn_id": groups[idx],
                "label":   "eot" if y_va[i] == 1 else "hold",
                "dur":     float(m.get("dur", 0.5)),
                "p":       float(p_va[i]),
            })

        best_T, best_D, best_lat = find_best_threshold(va_pauses)
        T_per_fold.append(best_T)
        D_per_fold.append(best_D)

        # AUC (manual Mann-Whitney)
        order = np.argsort(p_va)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(p_va) + 1)
        n1, n0 = y_va.sum(), len(y_va) - y_va.sum()
        auc = ((ranks[y_va == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")

        print(f"  fold {fold_i+1}: T={best_T:.2f}, D={best_D*1000:.0f}ms, "
              f"val_lat={best_lat*1000:.0f}ms, AUC={auc:.3f}")
        fold_results.append({
            "fold": fold_i + 1,
            "T": best_T, "D": best_D,
            "val_latency": best_lat,
            "auc": float(auc),
        })

    return fold_results, T_per_fold, D_per_fold


# ---------------------------------------------------------------------------
# Full refit and save
# ---------------------------------------------------------------------------

def refit_and_save(X, y, groups, lang_map, T_per_fold, D_per_fold,
                   model_path, thresh_path):
    """
    Refit on all data, calibrate on a 20% hold-out, save model + thresholds.
    """
    T_std  = float(np.std(T_per_fold))
    T_mean = float(np.mean(T_per_fold))
    D_mean = float(np.mean(D_per_fold))

    if T_std > 0.15:
        final_T  = float(np.percentile(T_per_fold, 10))
        strategy = "conservative_p10"
        print(f"  WARN: T unstable (std={T_std:.3f}). Using conservative T={final_T:.3f}")
    else:
        final_T  = T_mean
        strategy = "mean"
        print(f"  OK: T stable (std={T_std:.3f}). Using T={final_T:.3f}")

    # 80/20 split for final calibration (stratified on label)
    rng   = np.random.default_rng(42)
    eot_idx  = np.where(y == 1)[0]
    hold_idx = np.where(y == 0)[0]
    n_cal_eot  = max(1, len(eot_idx)  // 5)
    n_cal_hold = max(1, len(hold_idx) // 5)
    cal_idx  = np.concatenate([
        rng.choice(eot_idx,  n_cal_eot,  replace=False),
        rng.choice(hold_idx, n_cal_hold, replace=False),
    ])
    fit_idx = np.setdiff1d(np.arange(len(y)), cal_idx)

    weights_all = build_sample_weights(groups, lang_map)

    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05,
        max_depth=4, min_samples_leaf=8, random_state=42,
    )
    clf.fit(X[fit_idx], y[fit_idx], sample_weight=weights_all[fit_idx])

    # sklearn 1.9+: FrozenEstimator pattern for pre-fit calibration
    frozen = FrozenEstimator(clf)
    calibrated = CalibratedClassifierCV(
        estimator=frozen, method='sigmoid',
    )
    calibrated.fit(X[cal_idx], y[cal_idx])

    joblib.dump(calibrated, model_path)
    print(f"  Saved model → {model_path}")

    thresh_data = {
        "threshold":  final_T,
        "delay":      D_mean,
        "T_per_fold": [float(t) for t in T_per_fold],
        "D_per_fold": [float(d) for d in D_per_fold],
        "T_std":      T_std,
        "T_mean":     T_mean,
        "T_strategy": strategy,
    }
    with open(thresh_path, "w") as f:
        json.dump(thresh_data, f, indent=2)
    print(f"  Saved thresholds → {thresh_path}")

    return calibrated, final_T, D_mean


# ---------------------------------------------------------------------------
# Prediction helpers (used for in-sample evaluation after training)
# ---------------------------------------------------------------------------

def predict_and_write(calibrated, X, y, groups, meta, out_csv):
    p = calibrated.predict_proba(X)[:, 1]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for i, m in enumerate(meta):
            w.writerow([m["turn_id"], m["pause_index"], f"{p[i]:.4f}"])
    print(f"  Wrote {len(meta)} predictions → {out_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_en", required=True,  help="English data dir")
    ap.add_argument("--train_hi", default=None,   help="Hindi data dir (optional)")
    ap.add_argument("--out_model",  default="model.joblib")
    ap.add_argument("--out_thresh", default="thresholds.json")
    ap.add_argument("--n_splits",   type=int, default=5)
    args = ap.parse_args()

    # --- Load data ---
    print("Loading English data...")
    en_records = load_data(args.train_en, lang_prefix="en")

    hi_records = []
    if args.train_hi:
        print("Loading Hindi data...")
        hi_records = load_data(args.train_hi, lang_prefix="hi")

    all_records = en_records + hi_records
    print(f"Total pauses: {len(all_records)} "
          f"(EN={len(en_records)}, HI={len(hi_records)})")

    # --- Extract features ---
    print("Extracting features...")
    X, y, groups, meta = extract_all_features(all_records)

    # Rebuild records in the same order as extract_all_features output
    # (by_turn dict order → sorted by pause_index)
    # We need records_ordered parallel to X for sample weights
    by_turn = defaultdict(list)
    for r in all_records:
        by_turn[r["turn_id"]].append(r)
    records_ordered = []
    for tid in dict.fromkeys(r["turn_id"] for r in all_records):  # preserve order
        for r in sorted(by_turn[tid], key=lambda x: x["pause_index"]):
            records_ordered.append(r)

    # Build lang_map: turn_id -> 'en'|'hi'
    lang_map = {}
    for r in all_records:
        lang_map[r["turn_id"]] = r["lang"]

    print(f"Feature matrix: {X.shape}, NaN fraction: {np.isnan(X).mean():.3f}")
    print(f"Label balance: {y.mean():.3f} EOT rate")

    # --- Cross-validation ---
    print(f"\nRunning {args.n_splits}-fold GroupKFold CV...")
    fold_results, T_per_fold, D_per_fold = train_with_cv(
        X, y, groups, meta, lang_map, n_splits=args.n_splits
    )

    print(f"\nThreshold stats: mean={np.mean(T_per_fold):.3f}, "
          f"std={np.std(T_per_fold):.3f}")

    # --- Refit and save ---
    print("\nRefitting on full dataset...")
    calibrated, final_T, final_D = refit_and_save(
        X, y, groups, lang_map, T_per_fold, D_per_fold,
        args.out_model, args.out_thresh
    )

    # --- In-sample predictions for score.py ---
    print("\nWriting in-sample predictions...")
    if args.train_hi:
        # Write EN and HI separately for separate scoring
        en_mask = np.array([g.startswith("en") for g in groups])
        hi_mask = ~en_mask

        predict_and_write(calibrated,
                          X[en_mask], y[en_mask],
                          [g for g, m in zip(groups, en_mask) if m],
                          [m for m, mask in zip(meta, en_mask) if mask],
                          "train_v2_en.csv")
        predict_and_write(calibrated,
                          X[hi_mask], y[hi_mask],
                          [g for g, m in zip(groups, hi_mask) if m],
                          [m for m, mask in zip(meta, hi_mask) if mask],
                          "train_v2_hi.csv")
    else:
        predict_and_write(calibrated, X, y, groups, meta, "train_v2_en.csv")

    print(f"\nDone. Operating point: T={final_T:.3f}, D={final_D*1000:.0f}ms")
    print("Run score.py on train_v2_en.csv and train_v2_hi.csv to verify.")


if __name__ == "__main__":
    main()
