# NOTES — Error Analysis & Hindi-Specific Observations

## Key Finding: EN-only Model Fails on Hindi

AUC = 0.412 on Hindi when trained only on English. Below-random performance means the model's predictions are **inverted** for Hindi — it confidently predicts EOT for HOLD pauses and vice versa.

This is consistent with the criticisms in `criticize.md`:
- The autocorr pitch tracker's fmin=60, fmax=400 range misses Hindi male voices (closer to 110Hz mean, but variably lower)
- The English normalization prior (150Hz) shifts normalized Hindi pitch in the wrong direction
- The spectral stability feature may behave differently for Hindi vowel-heavy filler sounds

## Feature Behavior Observations

### pitch_slope
- Voiced frames work well for English: clear terminal F0 drop before EOT
- Hindi: more variable, pyin handles octave ambiguity better than autocorr

### spectral_stability
- Currently returns NaN for silent segments (correct behavior)
- For voiced segments, the `mean_c / (var_c + 1e-4)` normalization makes it speaker-independent
- High value (stable centroid) = held vowel = hesitation signal

### voicing_density_ratio
- Works well for both languages — dimensionless ratio is genuinely language-neutral
- Final window having lower voiced fraction than prior window is a reliable EOT signal

## Data Observations

### Hindi Data Characteristics
- 42.6% of hold pauses < 300ms (vs 29.7% for English)
- Hindi speakers tend to have shorter hold pauses — this is why 850ms is the Hindi baseline
- At 100ms delay, the model fires very quickly when p_eot > 0.45, achieving 355ms
- The 355ms result with only 3% interruptions (vs 5% budget) suggests there is still room to push lower

### English Data Characteristics  
- 638ms in-sample (up from 1190ms baseline) — clear improvement
- The 29.7% holds < 300ms means we can't go below ~350ms delay without budget violations

## Physical Floor Analysis

```
Hindi:  5th percentile hold duration = 100ms → floor is 150ms delay
        42.6% holds < 300ms → at 300ms delay, would cut 42.6% of turns (>>5% budget)
        At 100ms delay + careful threshold → 355ms achievable
English: 5th pct = 100ms, but 29.7% < 300ms → 350ms delay is near the floor
```

## Potential Improvements (if time allows)

1. **Error analysis on worst cases**: Load the 5 pauses with highest confidence wrong predictions
   ```python
   # Find where model is most wrong (confident EOT prediction on HOLD)
   # These are the false cutoffs
   ```

2. **Feature importance check**: HistGBM provides `feature_importances_` — verify pitch_slope and voicing_density_ratio are dominant

3. **Additional features to try**:
   - Final syllable lengthening (duration of last voiced stretch vs. speaker mean)
   - Speaking rate estimate (syllable count / turn duration)
   - Energy rise at turn boundary (some Hindi speakers have rising energy at the end)

4. **Ensemble**: HI-only model + EN-only model + combined model weighted average
   - Risk: small folds make HI-only model unreliable (already saw std(T)=0.191)
   - Only try if combined model clearly underperforms on one language on held-out data
