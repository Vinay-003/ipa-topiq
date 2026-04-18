# TOPIQ Delta Benchmark Summary

Repository: `https://github.com/Vinay-003/ipa-topiq`

## Experiment Goal
Improve cross-dataset performance on **TID2013** while training on **KonIQ-10k**, using a lightweight distortion-statistics branch + adaptive fusion gate added to TOPIQ-NR.

## KonIQ Validation During Training (Our Run)
From training log at epoch 7, iter 3200:

- Validation SRCC (KonIQ-10k): `0.9286`
- Validation PLCC (KonIQ-10k): `0.9425`

Repository baseline values for TOPIQ-NR on KonIQ-10k:

- Baseline SRCC: `0.9299`
- Baseline PLCC: `0.9436`

Comparison:

| Metric | Baseline | Ours   | Delta (Ours - Base) |
|---|---:|---:|---:|
| SRCC | 0.9299 | 0.9286 | -0.0013 |
| PLCC | 0.9436 | 0.9425 | -0.0011 |

Interpretation: this is effectively baseline-matched behavior on KonIQ.

## TID2013 Final Test (Our Run)

- SRCC: `0.4542`
- PLCC: `0.5650`
- KRCC: `0.3197`

Repository baseline values for TOPIQ-NR on TID2013:

- Baseline SRCC: `0.4452`
- Baseline PLCC: `0.5625`
- Baseline KRCC: `0.3143`

Comparison:

| Metric | Baseline | Ours   | Delta (Ours - Base) |
|---|---:|---:|---:|
| SRCC | 0.4452 | 0.4542 | +0.0090 |
| PLCC | 0.5625 | 0.5650 | +0.0025 |
| KRCC | 0.3143 | 0.3197 | +0.0054 |

## High-Level Conclusion

- Source-domain behavior (KonIQ validation) remains essentially unchanged.
- Target-domain cross-dataset behavior (TID2013) improves across all 3 metrics.
- The result supports the value of the delta addition under strict architecture constraints.
