
# Streamlit demo for Stock/Crypto Direction Classifier (NB15)
# - Loads df_nb02 (csv or parquet) and artifacts (feature_list.json, scaler.joblib, lr.joblib, threshold.json)
# - UI: set threshold (tau) and fee_bps
# - Metrics: AUC, AP (area under PR), Brier, LogLoss on the selected date range
# - Plots: Equity vs Buy&Hold, ROC, PR, Calibration
import os, json, pathlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_curve,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve

import streamlit as st

# ----------------------
# Paths & loaders
# ----------------------
HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent  # repo root (assuming this file lives in app/)

def load_df(root: Path) -> pd.DataFrame:
    data_dir = root / "data"
    csv_path = data_dir / "df_nb02.csv"
    pq_path  = data_dir / "df_nb02.parquet"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif pq_path.exists():
        df = pd.read_parquet(pq_path)
    else:
        st.error("Missing data file: expected data/df_nb02.csv or data/df_nb02.parquet")
        st.stop()

    # Parse date if present
    for c in ["date", "Date", "timestamp", "ts"]:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c])
            except Exception:
                pass
            if c != "date":
                df["date"] = df[c]
            break
    return df

def load_artifacts(root: Path):
    art = root / "artifacts"
    with open(art / "feature_list.json", "r", encoding="utf-8") as f:
        feature_list = json.load(f)
    scaler = joblib.load(art / "scaler.joblib")
    model  = joblib.load(art / "lr.joblib")
    tau_file = art / "threshold.json"
    tau_val = None
    if tau_file.exists():
        try:
            t = json.load(open(tau_file, "r", encoding="utf-8"))
            tau_val = t.get("tau") or t.get("threshold") or t.get("value")
        except Exception:
            tau_val = None
    return feature_list, scaler, model, tau_val

def infer_target(df: pd.DataFrame):
    # Prefer explicit label; else ret_next>0; else derive from close
    candidates = ["y", "label", "target", "y_bin", "direction", "is_up", "class", "cls"]
    for c in candidates:
        if c in df.columns:
            y = df[c].astype(int).clip(0, 1).values
            return y, c
    if "ret_next" in df.columns:
        y = (df["ret_next"].astype(float) > 0).astype(int).values
        return y, "ret_next>0"
    if "close" in df.columns:
        ret_next = df["close"].astype(float).pct_change().shift(-1).fillna(0.0)
        df["ret_next"] = ret_next
        y = (ret_next > 0).astype(int).values
        return y, "ret_next_from_close>0"
    return None, None

def make_dataset(df: pd.DataFrame, features: list):
    cols = [c for c in features if c in df.columns]
    if not cols:
        raise ValueError("None of the expected features are present in df_nb02. Check artifacts/feature_list.json vs data columns.")
    tmp = df[cols].replace([np.inf, -np.inf], np.nan)
    y_vals, y_name = infer_target(df)
    if y_vals is None:
        return None, None, None, None, None
    if "ret_next" in df.columns:
        retn = df["ret_next"].astype(float).values
    elif "close" in df.columns:
        retn = df["close"].astype(float).pct_change().shift(-1).fillna(0.0).values
    else:
        retn = np.zeros(len(df), dtype=float)
    tmp["__y__"] = y_vals
    tmp["__ret_next__"] = retn
    tmp = tmp.dropna()
    X  = tmp[cols].to_numpy()
    y  = tmp["__y__"].astype(int).to_numpy()
    retn = tmp["__ret_next__"].astype(float).to_numpy()
    idx = tmp.index
    return X, y, retn, idx, y_name

def predict_proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] == 2:
            return p[:, 1]
        if p.ndim == 1:
            return p
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1 / (1 + np.exp(-s))
    pred = model.predict(X)
    return np.clip(pred.astype(float), 0.0, 1.0)

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Direction Classifier Demo", layout="wide")
st.title("📈 Direction Classifier — Streamlit Demo (NB15)")

df = load_df(ROOT)
feature_list, scaler, model, tau_art = load_artifacts(ROOT)

with st.sidebar:
    st.header("Settings")
    if "date" in df.columns:
        dmin, dmax = df["date"].min(), df["date"].max()
        start, end = st.date_input(
            "Date range",
            value=(dmin.date(), dmax.date()),
            min_value=dmin.date(),
            max_value=dmax.date()
        )
        mask = df["date"].dt.date.between(start, end)
        df_view = df.loc[mask].copy()
    else:
        df_view = df.copy()
        st.caption("No 'date' column found; using all rows.")

    default_tau = float(tau_art) if tau_art is not None else 0.59
    tau = st.slider("Decision threshold (τ)", 0.00, 1.00, value=float(round(default_tau, 2)), step=0.01)
    fee_bps = st.number_input("Fee (bps) per position flip", value=5, min_value=0, max_value=100, step=1)
    st.caption("A flip is any change in position (enter/exit). Fee applied per flip.")

try:
    X, y, retn, idx, y_name = make_dataset(df_view, feature_list)
except Exception as e:
    st.error(str(e))
    st.stop()
if X is None:
    st.error("Could not infer a binary target; ensure your df_nb02 has one of the supported label columns or ret_next/close to derive one.")
    st.stop()

Xs = scaler.transform(X)
proba = predict_proba(model, Xs)
proba = np.clip(proba, 1e-6, 1 - 1e-6)

# Metrics
import numpy as np
metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
try:
    auc = roc_auc_score(y, proba)
except Exception:
    auc = float("nan")
try:
    ap = average_precision_score(y, proba)
except Exception:
    ap = float("nan")
try:
    brier = brier_score_loss(y, proba)
except Exception:
    brier = float("nan")
try:
    ll = log_loss(y, proba)
except Exception:
    ll = float("nan")

metrics_col1.metric("ROC AUC", f"{auc:.3f}" if np.isfinite(auc) else "n/a")
metrics_col2.metric("Average Precision (PR AUC)", f"{ap:.3f}" if np.isfinite(ap) else "n/a")
metrics_col3.metric("Brier Score", f"{brier:.4f}" if np.isfinite(brier) else "n/a")
metrics_col4.metric("Log Loss", f"{ll:.4f}" if np.isfinite(ll) else "n/a")

# Strategy equity vs Buy&Hold
sig = (proba >= tau).astype(int)
flips = np.zeros_like(sig)
if len(flips) > 1:
    flips[1:] = (sig[1:] != sig[:-1]).astype(int)
fee = flips * (fee_bps / 10000.0)
strategy_ret = (retn * sig) - fee

eq_strategy = np.cumprod(1.0 + strategy_ret)
eq_bh = np.cumprod(1.0 + retn)

# Align dates
if "date" in df_view.columns:
    dates = df_view.iloc[idx]["date"].values
else:
    dates = df_view.index.values

st.subheader("Equity Curve vs. Buy & Hold")
fig1, ax1 = plt.subplots()
ax1.plot(dates, eq_bh, label="Buy & Hold")
ax1.plot(dates, eq_strategy, label=f"Strategy (τ={tau:.2f}, fee={fee_bps}bps)")
ax1.set_xlabel("Date" if "date" in df_view.columns else "Index")
ax1.set_ylabel("Equity (×)")
ax1.legend()
st.pyplot(fig1)

# ROC
st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y, proba)
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f"AUC={auc:.3f}" if np.isfinite(auc) else "AUC=n/a")
ax2.plot([0, 1], [0, 1], linestyle="--")
ax2.set_xlabel("FPR")
ax2.set_ylabel("TPR")
ax2.legend()
st.pyplot(fig2)

# PR
st.subheader("Precision–Recall Curve")
prec, rec, _ = precision_recall_curve(y, proba)
fig3, ax3 = plt.subplots()
ax3.plot(rec, prec, label=f"AP={ap:.3f}" if np.isfinite(ap) else "AP=n/a")
ax3.set_xlabel("Recall")
ax3.set_ylabel("Precision")
ax3.legend()
st.pyplot(fig3)

# Calibration
st.subheader("Calibration")
prob_true, prob_pred = calibration_curve(y, proba, n_bins=10, strategy="uniform")
fig4, ax4 = plt.subplots()
ax4.plot(prob_pred, prob_true, marker="o", label="Model")
ax4.plot([0, 1], [0, 1], linestyle="--")
ax4.set_xlabel("Predicted probability")
ax4.set_ylabel("Observed frequency")
ax4.legend()
st.pyplot(fig4)

# Tail preview
st.subheader("Latest predictions (tail)")
tail_n = min(12, len(proba))
preview = {}
if "date" in df_view.columns:
    preview["date"] = list(dates[-tail_n:])
if "close" in df_view.columns:
    preview["close"] = list(df_view.iloc[idx]["close"].values[-tail_n:])
preview["proba"] = list(proba[-tail_n:])
preview["signal"] = list(sig[-tail_n:])
st.dataframe(pd.DataFrame(preview))
st.caption("Signals are long-only (1=long, 0=cash); flips incur fee in equity curve.")
