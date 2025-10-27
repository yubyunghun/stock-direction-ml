# app/streamlit_app.py — model & monitoring tabs
import numpy as np, pandas as pd, matplotlib.pyplot as plt, streamlit as st

from app.config import ROOT, DEFAULT_TAU
from app.lib_artifacts import load_artifacts
from app.lib_fetch import load_repo_df, fetch_equity_df, fetch_crypto_df
from app.lib_features import make_dataset
from app.lib_eval import predict_proba, metrics_all, tau_sweep
from app.lib_monitor import load_monitoring, summarize_monitor, summarize_paper, summarize_backtest, promotion_checks

st.set_page_config(page_title="Direction Classifier", layout="wide")
st.title("📈 Direction Classifier — Any Ticker (Equities & Crypto)")

# ---- Sidebar: data source ----
with st.sidebar:
    st.header("Data source")
    src = st.radio("Choose", ["Repo file","Fetch (Yahoo)"], index=0)
    asset_class = st.selectbox("Asset class", ["equity","crypto"], index=0)

    if src == "Repo file":
        df = load_repo_df()
        if "ticker" in df.columns:
            ticks = sorted(df["ticker"].dropna().unique().tolist())
            default = df["ticker"].value_counts().idxmax()
            ticker = st.selectbox("Ticker", ticks, index=max(0, ticks.index(default)))
            df = df.loc[df["ticker"]==ticker].copy()
            st.caption(f"Ticker: **{ticker}**  •  Rows: {len(df)}")
        else:
            ticker = None
            st.caption("No 'ticker' column; using all rows.")
        if "date" in df.columns:
            dmin, dmax = df["date"].min(), df["date"].max()
            start, end = st.date_input("Date range", value=(dmin.date(), dmax.date()),
                                       min_value=dmin.date(), max_value=dmax.date())
            df = df.loc[df["date"].dt.date.between(start, end)].copy()
    else:
        ticker = st.text_input("Ticker", value=("AAPL" if asset_class=="equity" else "BTC-USD"))
        dates = st.date_input("Fetch range (UTC)", value=(pd.to_datetime("2023-01-01").date(), pd.Timestamp.today().date()))
        btn = st.button("Fetch data")
        if not btn:
            st.stop()
        try:
            if asset_class=="equity":
                df = fetch_equity_df(ticker, dates[0], dates[1])
            else:
                df = fetch_crypto_df(ticker, dates[0], dates[1])
            st.success(f"Fetched {len(df)} rows for {ticker}")
        except Exception as e:
            st.error(f"Fetch failed: {e}"); st.stop()

# ---- Artifacts (with crypto folder support for tau_map) ----
try:
    feature_list, scaler, model, tau_art, tau_map = load_artifacts(asset_class=("equity" if asset_class=="equity" else "crypto"))
except Exception:
    st.warning("Artifacts not found for selected asset class; falling back to equity artifacts.")
    feature_list, scaler, model, tau_art, tau_map = load_artifacts("equity")
default_tau = float(tau_map.get(ticker, tau_art if tau_art is not None else DEFAULT_TAU)) if ticker else (tau_art or DEFAULT_TAU)

# ---- Dataset & predictions ----
X, y, retn, idx, used_cols = make_dataset(df, feature_list)
if len(X)==0: st.error("No usable rows after feature alignment/NA drop."); st.stop()
Xs = scaler.transform(X)
p  = np.clip(predict_proba(model, Xs), 1e-6, 1-1e-6)

# ---- Tabs ----
tab_model, tab_monitor = st.tabs(["🔮 Model", "🛡️ Monitoring & Promotion"])

with tab_model:
    with st.sidebar:
        tau     = st.slider("Decision threshold (τ)", 0.00, 1.00, value=float(round(default_tau,2)), step=0.01)
        fee_bps = st.number_input("Fee (bps) per position flip", value=5, min_value=0, max_value=100, step=1)

    # Metrics
    c1,c2,c3,c4 = st.columns(4)
    if y is not None and len(y)==len(p):
        m = metrics_all(y, p)
        c1.metric("ROC AUC", f"{m['auc']:.3f}" if np.isfinite(m['auc']) else "n/a")
        c2.metric("PR AUC",  f"{m['ap']:.3f}" if np.isfinite(m['ap']) else "n/a")
        c3.metric("Brier",    f"{m['brier']:.4f}" if np.isfinite(m['brier']) else "n/a")
        c4.metric("Log Loss", f"{m['logloss']:.4f}" if np.isfinite(m['logloss']) else "n/a")
    else:
        for c in (c1,c2,c3,c4): c.metric("—","—")
        st.info("Labels not available; showing predictions/equity only.")

    # Equity vs B&H
    sig = (p >= tau).astype(int)
    flips = np.zeros_like(sig)
    if len(flips)>1: flips[1:] = (sig[1:] != sig[:-1]).astype(int)
    fee = flips * (fee_bps/10000.0)
    eq  = np.cumprod(1 + (retn*sig - fee))
    bh  = np.cumprod(1 + retn)

    dates_axis = (df.iloc[idx]["date"].values if "date" in df.columns else df.index.values)
    st.subheader("Equity Curve vs. Buy & Hold")
    fig, ax = plt.subplots()
    ax.plot(dates_axis, bh,  label="Buy & Hold")
    ax.plot(dates_axis, eq,  label=f"Strategy (τ={tau:.2f}, fee={fee_bps}bps)")
    ax.set_xlabel("Date" if "date" in df.columns else "Index"); ax.set_ylabel("Equity (×)")
    ax.legend(); st.pyplot(fig)

    # τ-sweep
    with st.expander("τ-sweep (F1 & Final Equity)"):
        if y is not None and len(y)==len(p):
            grid, f1s, finals = tau_sweep(y, p, retn, fee_bps=fee_bps)
            best_f1_tau = float(grid[int(np.nanargmax(f1s))])
            best_eq_tau = float(grid[int(np.nanargmax(finals))])
            st.write({"best_f1_tau":best_f1_tau, "best_final_equity_tau":best_eq_tau})
            f, axf = plt.subplots(); axf.plot(grid, f1s, label="F1 vs τ"); axf.set_xlabel("τ"); axf.set_ylabel("F1"); axf.legend(); st.pyplot(f)
        else:
            st.info("Labels not available; τ-sweep (F1) disabled.")

    # Tail & CSV
    pred_df = pd.DataFrame({"date": dates_axis, "proba": p, "signal": sig})
    if "close" in df.columns: pred_df["close"] = df.iloc[idx]["close"].values
    st.subheader("Latest predictions (tail)")
    st.dataframe(pred_df.tail(min(12, len(pred_df))))
    st.download_button("Download predictions CSV",
        data=pred_df.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv", mime="text/csv")

with tab_monitor:
    st.subheader("Monitoring snapshot & Promotion readiness")
    mon_raw, paper_raw, back_raw = load_monitoring(ROOT)
    mon = summarize_monitor(mon_raw) if mon_raw else {}
    pap = summarize_paper(paper_raw) if paper_raw else {}
    bak = summarize_backtest(back_raw) if back_raw else {}

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Winrate (60d)", f"{mon.get('winrate_60d'):.2%}" if isinstance(mon.get('winrate_60d'), (int,float)) else "n/a")
    c2.metric("Trades (60d)",  f"{mon.get('trades_60d')}" if mon.get('trades_60d') is not None else "n/a")
    c3.metric("PSI max",       f"{mon.get('psi_max'):.3f}" if isinstance(mon.get('psi_max'), (int,float)) else "n/a")
    c4.metric("KS max",        f"{mon.get('ks_max'):.3f}" if isinstance(mon.get('ks_max'), (int,float)) else "n/a")

    c5,c6,c7 = st.columns(3)
    c5.metric("Paper final equity", f"{pap.get('final_equity'):.3f}" if isinstance(pap.get('final_equity'), (int,float)) else "n/a")
    c6.metric("Backtest AUC",       f"{bak.get('auc'):.3f}" if isinstance(bak.get('auc'), (int,float)) else "n/a")
    c7.metric("Backtest parity",    "OK" if bak.get('parity_ok') else ("n/a" if bak.get('parity_ok') is None else "FAIL"))

    checks = promotion_checks(mon, bak)
    st.markdown(f"**Promotion status:** {'🟢 PASS' if checks['status']=='PASS' else ('🟡 UNKNOWN' if checks['status']=='UNKNOWN' else '🟠 HOLD')}")
    if checks["rules"]:
        st.write(pd.DataFrame(checks["rules"]))

    with st.expander("Raw artifacts"):
        colA, colB, colC = st.columns(3)
        colA.download_button("monitor_snapshot.json", data=(str(mon_raw).encode('utf-8') if mon_raw else b''), file_name="monitor_snapshot.json")
        colB.download_button("paper_trade.json", data=(str(paper_raw).encode('utf-8') if paper_raw else b''), file_name="paper_trade.json")
        colC.download_button("backtest_summary.json", data=(str(back_raw).encode('utf-8') if back_raw else b''), file_name="backtest_summary.json")

    if not mon_raw:  st.info("No artifacts/monitor_snapshot.json found — NB11 creates it.")
    if not paper_raw: st.info("No artifacts/paper_trade.json found — NB10/NB14 create it.")
    if not back_raw:  st.info("No artifacts/backtest_summary.json found — NB7 writes it.")

st.caption("Monitoring shows 60d KPIs, drift, backtest parity, and a promotion recommendation.")
