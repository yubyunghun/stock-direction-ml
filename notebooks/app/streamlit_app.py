# app/streamlit_app.py — modular, fetch-enabled
import numpy as np, pandas as pd, matplotlib.pyplot as plt, streamlit as st
from app.config import ROOT, DEFAULT_TAU
from app.lib_artifacts import load_artifacts
from app.lib_fetch import load_repo_df, fetch_equity_df, fetch_crypto_df
from app.lib_features import make_dataset
from app.lib_eval import predict_proba, metrics_all, tau_sweep

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

# ---- Artifacts ----
feature_list, scaler, model, tau_art, tau_map = load_artifacts(asset_class=("equity" if asset_class=="equity" else "crypto"))
default_tau = float(tau_map.get(ticker, tau_art if tau_art is not None else DEFAULT_TAU)) if ticker else (tau_art or DEFAULT_TAU)

# ---- Dataset ----
X, y, retn, idx, used_cols = make_dataset(df, feature_list)
if len(X)==0: st.error("No usable rows after feature alignment/NA drop."); st.stop()
Xs = scaler.transform(X)
p  = np.clip(predict_proba(model, Xs), 1e-6, 1-1e-6)

with st.sidebar:
    tau     = st.slider("Decision threshold (τ)", 0.00, 1.00, value=float(round(default_tau,2)), step=0.01)
    fee_bps = st.number_input("Fee (bps) per position flip", value=5, min_value=0, max_value=100, step=1)

# ---- Metrics ----
c1,c2,c3,c4 = st.columns(4)
if y is not None and len(y)==len(p):
    m = metrics_all(y, p)
    c1.metric("ROC AUC", f"{m['auc']:.3f}" if np.isfinite(m['auc']) else "n/a")
    c2.metric("PR AUC",  f"{m['ap']:.3f}" if np.isfinite(m['ap']) else "n/a")
    c3.metric("Brier",    f"{m['brier']:.4f}" if np.isfinite(m['brier']) else "n/a")
    c4.metric("Log Loss", f"{m['logloss']:.4f}" if np.isfinite(m['logloss']) else "n/a")
else:
    for c in (c1,c2,c3,c4): c.metric("—","—")
    st.info("Labels not available for this selection; showing predictions/equity only.")

# ---- Equity vs B&H ----
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

# ---- τ-sweep ----
with st.expander("τ-sweep (F1 & Final Equity)"):
    if y is not None and len(y)==len(p):
        grid, f1s, finals = tau_sweep(y, p, retn, fee_bps=fee_bps)
        best_f1_tau = float(grid[int(np.nanargmax(f1s))])
        best_eq_tau = float(grid[int(np.nanargmax(finals))])
        st.write({"best_f1_tau":best_f1_tau, "best_final_equity_tau":best_eq_tau})
        f, axf = plt.subplots(); axf.plot(grid, f1s, label="F1 vs τ"); axf.set_xlabel("τ"); axf.set_ylabel("F1"); axf.legend(); st.pyplot(f)
    else:
        st.info("Labels not available; τ-sweep (F1) disabled.")

# ---- Tail & CSV ----
pred_df = pd.DataFrame({"date": dates_axis, "proba": p, "signal": sig})
if "close" in df.columns: pred_df["close"] = df.iloc[idx]["close"].values
st.subheader("Latest predictions (tail)")
st.dataframe(pred_df.tail(min(12, len(pred_df))))
st.download_button("Download predictions CSV",
    data=pred_df.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv", mime="text/csv")

st.caption("This UI runs your trained LR on repo data or live Yahoo fetch. Crypto uses BTC-based proxies. Not financial advice.")
