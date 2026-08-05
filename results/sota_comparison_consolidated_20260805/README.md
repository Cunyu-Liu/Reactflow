# M0-X Published-SOTA Horizontal Comparison (横向对比表)

Changer-detection task, publication split (3516 train / 548 val), study-macro AUPRC. Test SEALED.

| Method | Category | AUPRC |
|---|---|---|
| Our improved changer classifier (EPRO_DEV_10) | Ours (trained) | 0.7435 |
| EPRO_DEV_06 structure-aware changer | Ours (trained) | 0.7353 |
| p2_paired baseline | Internal baseline | 0.6936 |
| wt_only baseline | Internal baseline | 0.6748 |
| RNAformer (Nat. Mach. Intell. 2023) | Published SOTA (DL) | 0.5554 |
| EternaFold (Nat. Commun. 2022) | Published SOTA (ML) | 0.4632 |
| ViennaRNA Turner-rules physics | Published physics | 0.4534 |

**Result:** our improved classifier (EPRO_DEV_10, 0.7435) outperforms published SOTA RNAformer (0.5554) by +0.188 and EternaFold (0.4632) by +0.280 study-macro AUPRC on the same validation changer-detection task.
