# Model Card — Stock Direction (LogReg)
**Date:** 2025-10-13  
**Ticker:** AAPL | **Period:** 2015-02-06 → 2025-10-10  
**Labeling:** tau=0.500, dead_zone=False  
**Features (16):** close, high, low, macd, macd_signal, mkt_ret1, mkt_ret5, open, ret1, ret10, ret5, rsi14 ...

## Metrics (Val/Test summary)
| model | split | AUC | PR_AUC | Brier | LogLoss | PosRate |
| --- | --- | --- | --- | --- | --- | --- |
| LR | val | 0.4583 | 0.4883 | 0.2619 | 0.7181 | 0.5140 |
| LR | test | 0.4790 | 0.5202 | 0.2547 | 0.7025 | 0.5483 |
| XGB | val | 0.5113 | 0.5285 | 0.2771 | 0.7582 | 0.5140 |
| XGB | test | 0.4745 | 0.5228 | 0.2871 | 0.7758 | 0.5483 |

**Val-chosen threshold:** 0.050

## Stability (test by year)
| year | AUC | Brier | PosRate |
| --- | --- | --- | --- |
| 2023 | 0.4877 | 0.2520 | 0.5761 |
| 2024 | 0.5309 | 0.2482 | 0.5635 |
| 2025 | 0.4106 | 0.2643 | 0.5155 |

## Artifacts
- `artifacts/scaler.joblib`, `artifacts/lr.joblib`, `artifacts/threshold.json`, `artifacts/feature_list.json`
- Curves: `reports/figures/roc_nb05.png`, `pr_nb05.png`, `reliability_lr_nb05.png`, `equity_curve_lr.png`
- Importance: `reports/figures/lr_feature_importance_topk.png`
