# ReactFlow-Delta M1 failure atlas v1

Evidence status: `DEVELOPMENT_CONSUMED_DIAGNOSTIC_ONLY`. This is a diagnostic over consumed development outcomes, not confirmation.

## Decision

- Route: `LRSO_REMAINS_ELIGIBLE_FOR_CONTROLLED_M2_SCREEN`.
- Selected-rank CRPS gain over rank zero: +0.001796 (15/20 puzzles positive).
- Of that gain, Shapley mean contribution is +0.001226; scale contribution is +0.000570.
- Rank-zero signed-delta MAE gain vs WT: +0.012209 (20/20 puzzles positive).
- Selected-rank signed-delta MAE gain vs WT: +0.012978 (20/20 puzzles positive).
- Low-rank signed-delta MAE gain over rank zero: +0.000769 (13/20 puzzles positive).
- Low-rank residual energy / rank-zero predicted-delta energy: 0.064979.

## Response-magnitude strata

| stratum | positions | CRPS gain | mean part | scale part | signed-delta gain vs WT | low-rank delta-MAE gain |
|---|---:|---:|---:|---:|---:|---:|
| medium_0.20_0.50 | 234626 | +0.002896 | +0.002405 | +0.000491 | +0.026068 | +0.002054 |
| near_zero_le_0.05 | 508650 | -0.001333 | -0.000933 | -0.000400 | -0.014478 | -0.002745 |
| small_0.05_0.20 | 437458 | +0.000379 | +0.000390 | -0.000011 | +0.001489 | +0.001394 |
| tail_gt_0.50 | 204966 | +0.009634 | +0.006370 | +0.003263 | +0.084014 | +0.006257 |

## Per-puzzle heterogeneity

| puzzle | CRPS gain | mean part | scale part | rankpos delta gain vs WT | low-rank delta gain | residual energy ratio |
|---|---:|---:|---:|---:|---:|---:|
| P01 | +0.010174 | +0.006275 | +0.003899 | +0.026134 | +0.005790 | 0.154693 |
| P02 | -0.000896 | -0.000392 | -0.000504 | +0.015927 | -0.000238 | 0.018639 |
| P03 | +0.002500 | +0.001354 | +0.001146 | +0.017933 | +0.001695 | 0.022861 |
| P04 | +0.003388 | +0.002989 | +0.000399 | +0.011127 | +0.002254 | 0.104650 |
| P05 | +0.003536 | +0.000870 | +0.002666 | +0.023635 | -0.000530 | 0.027193 |
| P06 | +0.000251 | +0.000480 | -0.000229 | +0.018003 | +0.000083 | 0.022083 |
| P07 | +0.001684 | +0.001684 | +0.000000 | +0.004323 | +0.001752 | 0.063055 |
| P08 | +0.005163 | +0.004630 | +0.000533 | +0.012428 | +0.004125 | 0.234248 |
| P09 | +0.001770 | +0.001572 | +0.000198 | +0.020847 | +0.000944 | 0.044134 |
| P10 | -0.000799 | -0.001808 | +0.001009 | +0.021498 | -0.002272 | 0.025137 |
| P11 | +0.006687 | +0.004164 | +0.002523 | +0.013570 | +0.002082 | 0.184130 |
| P12 | +0.000228 | +0.000520 | -0.000292 | +0.007978 | +0.000347 | 0.048891 |
| P13 | +0.001939 | +0.001515 | +0.000424 | +0.007495 | +0.000714 | 0.061942 |
| P14 | +0.003780 | +0.003156 | +0.000624 | +0.009017 | +0.002142 | 0.177387 |
| P15 | -0.004576 | -0.003680 | -0.000896 | +0.005490 | -0.003761 | 0.114224 |
| P16 | -0.001020 | -0.000391 | -0.000630 | +0.009011 | -0.000441 | 0.041152 |
| P17 | -0.000770 | -0.000540 | -0.000231 | +0.009148 | -0.001292 | 0.021231 |
| P18 | +0.000410 | +0.000317 | +0.000093 | +0.006414 | -0.000334 | 0.072625 |
| P19 | +0.001252 | +0.001071 | +0.000181 | +0.011472 | +0.001749 | 0.050969 |
| P20 | +0.001215 | +0.000733 | +0.000481 | +0.008116 | +0.000573 | 0.064151 |

## Interpretation boundary

The atlas may identify where the existing model fails and which component contributes to CRPS. It cannot establish prospective performance, external replication, mechanism, or SOTA. The structure probe is a separate pre-frozen LOPO analysis.
