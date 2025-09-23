# Model Card — Stock Direction (LogReg)
**Date:** 2025-09-23
**Ticker:** UNK | **Period:** 2015-02-20 → 2023-12-28
**Labeling:** tau=None, dead_zone=None
**Features (8):** macd, macd_signal, ret1, ret10, ret5, rsi14, vol10, volz

## Metrics (Val/Test summary)
| model | split | AUC | PR_AUC | Brier | LogLoss | PosRate |
| --- | --- | --- | --- | --- | --- | --- |
| LR | val | 0.4920 | 0.5179 | 0.2630 | 0.7219 | 0.5325 |
| LR | test | 0.4546 | 0.4910 | 0.2746 | 0.7459 | 0.5252 |
| XGB | val | 0.5200 | 0.5508 | 0.2660 | 0.7310 | 0.5325 |
| XGB | test | 0.5226 | 0.5374 | 0.2650 | 0.7286 | 0.5252 |

**Val-chosen threshold (max-F1):** 0.290

## Stability (test by year)
| year | AUC | Brier | PosRate |
| --- | --- | --- | --- |
| 2022 | 0.4456 | 0.2839 | 0.4706 |
| 2023 | 0.4246 | 0.2670 | 0.5696 |

## Artifacts
- `artifacts/scaler.joblib`, `artifacts/lr.joblib`, `artifacts/threshold.json`, `artifacts/feature_list.json`
- Curves: `reports/figures/roc_nb05.png`, `pr_nb05.png`, `reliability_lr_nb05.png`, `equity_curve_lr.png`
- Importance: `reports/figures/lr_feature_importance_topk.png`
