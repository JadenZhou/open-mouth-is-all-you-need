# Condition Thresholds

All thresholds calibrated on the training video `20260107-1` (n≈all labeled frames).

## Training distribution summary

| Feature | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| brightness | 0.0 | 114.2 | 118.3 | 122.3 | 187.3 |
| motion_mean | 0.0 | 3.1 | 5.3 | 8.9 | 117.3 |
| yaw (°) | -45.5 | -22.9 | -11.6 | 9.1 | 52.5 |

## Chosen thresholds

| Condition | Threshold | Split on 20260107-1 | Rationale |
|---|---|---|---|
| Pose: angled | \|yaw\| ≥ 15° | ~62% angled | Standard frontal/angled boundary in pose literature |
| Lighting: low | brightness ≤ 115 | ~25% low | p25 of training — "low" = bottom quartile; ≤90 only captured 2% |
| Motion: high | motion_mean ≥ 5.3 | ~50% high | Median split for balanced groups |

## Design rationale

Thresholds are **absolute** (not per-video relative) so that condition labels mean the same thing across
all test videos. This allows cross-video comparison: a frame labeled "low brightness" in one video
represents the same real-world lighting level as in another.

A per-video median split would guarantee balanced groups but would make "high brightness" mean
different things across recordings — undermining the generalization analysis.
