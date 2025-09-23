# Model Card — Stock Direction (LogReg)
**Date:** 2025-09-23
**Ticker:** AAPL | **Period:** 2015-01-01 → 2023-12-31
**Labeling:** tau=0.001, dead_zone=True
**Features (8):** ret1, ret5, ret10, vol10, volz, rsi14, macd, macd_signal

## Metrics
|model|AUC|Brier|
|---|---|---|
|lr|0.4741|0.2535|
|xgb|0.4546|0.2954|

**Val-chosen threshold (max-F1):** 0.415

## Stability (test by year)
|index|AUC|Brier|PosRate|
|---|---|---|---|
|2022|0.4245|0.2586|0.4390|
|2023|0.4750|0.2517|0.5696|

## Artifacts
- `artifacts/scaler.joblib`, `artifacts/lr.joblib`, `artifacts/threshold.json`, `artifacts/feature_list.json`
- Curves: `reports/figures/roc_*.png`, `pr_*.png`, `equity_curve_lr.png`
