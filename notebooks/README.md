# Stock/Crypto Direction Classifier

An end-to-end ML project that predicts the **next-day direction** (up/down) for equities and crypto, with:
- Data prep & features (NB01–NB04)
- Model training + eval (Logistic Regression) (NB05–NB06, NB13)
- Backtest + threshold parity (NB07)
- Monitoring + promotion checks (NB10–NB12, NB21)
- Daily runner (NB14)
- Streamlit demo app (NB15–NB16, NB18–NB21)
- Crypto artifacts (NB20)

> **Not financial advice.** This is a demo research system.

## Quickstart

### Run locally
```bash
# from the repo folder that contains app/
python -m pip install -r notebooks/requirements.txt
python -m streamlit run app/streamlit_app.py
```

### Streamlit Cloud
- Main file path: `app/streamlit_app.py`
- Secrets are not required (uses Yahoo Finance for public data).

## App Features
- **Data source:** Repo file or live fetch (Yahoo) for equities & crypto.
- **Metrics:** ROC AUC, PR AUC, Brier, LogLoss (when labels exist).
- **Charts:** Equity vs Buy & Hold, ROC/PR, Calibration, τ-sweep.
- **Controls:** Decision threshold (τ) & fee (bps). Per-ticker τ defaults.
- **Monitoring tab:** 60-day KPIs, drift summary (PSI/KS), backtest parity, promotion recommendation.

### Screenshots (placeholders)
Add screenshots here (save under `reports/figures/`):
- `reports/figures/screenshot_model.png` – Model tab
- `reports/figures/screenshot_monitor.png` – Monitoring tab

## Repo Layout
```
app/
  config.py
  lib_artifacts.py
  lib_features.py
  lib_fetch.py
  lib_eval.py
  lib_monitor.py
  streamlit_app.py
artifacts/                 # equity artifacts (features, scaler, lr, threshold, tau_map)
artifacts_crypto/          # crypto artifacts (NB20)
data/                      # df_nb02.* and signals.csv
reports/
  figures/
  demo/                    # small prediction CSVs for sharing
notebooks/                 # NB01–NB22
```

## How It Works (short)
1. **Features:** Returns (1/5/10), realized vol (10d), z-vol, RSI(14), MACD+signal, market ret (SPY/BTC), VIX or BTC-vol proxy.
2. **Model:** Logistic Regression on standardized features.
3. **Threshold (τ):** Picked by final equity in backtest; can be per-ticker (`artifacts/*/tau_map.json`).
4. **Equity curve:** Long-only (1 = long, 0 = cash). Fee applied per position flip.
5. **Monitoring:** 60-day KPIs + drift + backtest parity → promotion PASS/HOLD.

## Repro & Daily
- **Daily runner:** NB14 (updates data/paper trade).
- **Monitoring snapshot:** NB11 outputs `artifacts/monitor_snapshot.json`.
- **Backtest summary:** NB07 writes `artifacts/backtest_summary.json`.

## Limitations
- AUC is modest (≈0.5–0.55 on splits); demonstration focus.
- Long-only, 1-day horizon; no risk sizing.
- Yahoo data used for convenience.

## License
MIT (or your choice)
