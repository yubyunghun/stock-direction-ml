# Model Card — Stock Direction (LogReg)
**Date:** 2025-10-09  
**Ticker:** AAPL | **Period:** 2015-02-06 → 2025-10-09  
**Labeling:** from NB5 settings  
**Features (8):** macd, macd_signal, ret1, ret10, ret5, rsi14, vol10, volz

## Metrics (Val/Test summary)
| model | split | AUC | PR_AUC | Brier | LogLoss | PosRate |
| --- | --- | --- | --- | --- | --- | --- |
| LR | val | 0.4729 | 0.4983 | 0.2537 | 0.7007 | 0.5140 |
| LR | test | 0.4537 | 0.5128 | 0.2530 | 0.6992 | 0.5475 |
| XGB | val | 0.4958 | 0.5134 | 0.2652 | 0.7266 | 0.5140 |
| XGB | test | 0.4548 | 0.5122 | 0.2759 | 0.7501 | 0.5475 |

**Val-chosen threshold:** 0.400

## Stability (test by year)
| year | AUC | Brier | PosRate |
| --- | --- | --- | --- |
| 2023 | 0.4465 | 0.2506 | 0.5761 |
| 2024 | 0.4866 | 0.2493 | 0.5635 |
| 2025 | 0.4149 | 0.2590 | 0.5130 |

## Artifacts
- `artifacts/scaler.joblib`, `artifacts/lr.joblib`, `artifacts/threshold.json`, `artifacts/feature_list.json`
- Curves: `reports/figures/roc_nb05.png`, `pr_nb05.png`, `reliability_lr_nb05.png`, `equity_curve_lr.png`
- Importance: `reports/figures/lr_feature_importance_topk.png`
