# Model Card — Stock Direction (LogReg)
**Date:** 2025-10-14  
**Ticker:** AAPL | **Period:** 2015-02-06 → 2025-10-10  
**Labeling:** tau=0.5, dead_zone=False  
**Features (16):** close, high, low, macd, macd_signal, mkt_ret1, mkt_ret5, open, ret1, ret10, ret5, rsi14 ...

## Metrics (Val/Test summary)
| model | split | AUC | PR_AUC | Brier | LogLoss | PosRate |
| --- | --- | --- | --- | --- | --- | --- |
| LR | val | 0.4583 | 0.4883 | 0.2619 | 0.7181 | 0.5140 |
| LR | test | 0.4790 | 0.5202 | 0.2547 | 0.7025 | 0.5483 |
| XGB | val | 0.5113 | 0.5285 | 0.2771 | 0.7582 | 0.5140 |
| XGB | test | 0.4745 | 0.5228 | 0.2871 | 0.7758 | 0.5483 |

**Chosen threshold (val-based):** 0.447

## Stability (Test by year)
| year | AUC | Brier | PosRate |
| --- | --- | --- | --- |
| 2023 | 0.4877 | 0.2520 | 0.5761 |
| 2024 | 0.5309 | 0.2482 | 0.5635 |
| 2025 | 0.4106 | 0.2643 | 0.5155 |

## Backtest (Test set, long-only @ threshold)
| CAGR | Vol | Sharpe | MaxDD | HitRate | Trades | TurnoverYr |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0990 | 0.2300 | 0.4307 | -0.3269 | 0.4796 | 101.0000 | 47.3086 |

## Artifacts
- `artifacts/scaler.joblib`, `artifacts/lr.joblib`, `artifacts/threshold.json`, `artifacts/feature_list.json`
- Curves: `reports/figures/roc_nb05.png`, `pr_nb05.png`, `reliability_lr_nb05.png`, `equity_curve_lr.png`
- Importance: `reports/figures/lr_feature_importance_topk.png`
