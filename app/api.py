from typing import Optional, Literal, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Local libs
from app.lib_artifacts import load_artifacts
from app.lib_features import make_dataset
from app.lib_eval import predict_proba
from app.lib_fetch import load_repo_df, fetch_equity_df, fetch_crypto_df

DEFAULT_TAU = 0.59

def _choose_tau(user_tau: Optional[float], tau_art: Optional[float], tau_map: Optional[dict], ticker: Optional[str]) -> float:
    if user_tau is not None:
        return float(user_tau)
    if tau_map and ticker and ticker in tau_map:
        try:
            return float(tau_map[ticker])
        except Exception:
            pass
    if tau_art is not None:
        try:
            return float(tau_art)
        except Exception:
            pass
    return DEFAULT_TAU

def predict_core(asset_class: str, source: str, ticker: Optional[str] = None,
                 start: Optional[str] = None, end: Optional[str] = None,
                 fee_bps: int = 5, tau: Optional[float] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Core prediction used by API/CLI/Notebooks. Returns (preds, meta)."""
    ac = "equity" if asset_class == "equity" else "crypto"

    # 1) Load data
    if source == "repo":
        df = load_repo_df()
        if ticker is None and "ticker" in df.columns and not df.empty:
            ticker = str(df["ticker"].iloc[0])
        if ticker is not None and "ticker" in df.columns:
            try:
                df = df[df["ticker"] == ticker].copy()
            except Exception:
                pass
    elif source == "fetch":
        if not ticker:
            raise ValueError("When source='fetch', you must provide 'ticker'.")
        if ac == "equity":
            df = fetch_equity_df(ticker, start=start, end=end)
        else:
            df = fetch_crypto_df(ticker, start=start, end=end)
    else:
        raise ValueError("source must be 'repo' or 'fetch'")

    # 2) Load artifacts (support legacy 4-tuple or new 5-tuple w/ tau_map)
    try:
        feature_list, scaler, model, tau_art, tau_map = load_artifacts(ac)
    except TypeError:
        feature_list, scaler, model, tau_art = load_artifacts(ac)  # type: ignore
        tau_map = None

    # 3) Dataset
    X, y, retn, idx, used_cols = make_dataset(df, feature_list)
    idx_arr = np.asarray(idx)

    # 4) Probabilities
    Xs = scaler.transform(X)
    p = np.clip(predict_proba(model, Xs), 1e-6, 1-1e-6)

    # 5) Threshold
    tau_used = _choose_tau(tau, tau_art, tau_map, ticker)

    # 6) Signals
    sig = (p >= tau_used).astype(int)

    # 7) Dates
    if "date" in df.columns:
        dates = pd.to_datetime(df.iloc[idx_arr]["date"], errors="coerce").dt.strftime("%Y-%m-%d").to_numpy()
    else:
        dates = idx_arr.astype(str)

    # 8) Close — robust to duplicate 'close' columns
    if "close" in df.columns:
        _c = df["close"]
        if getattr(_c, "ndim", 1) > 1:
            _c = _c.iloc[:, 0]
        close_vec = pd.to_numeric(_c, errors="coerce").to_numpy()
        close = close_vec[idx_arr]
    else:
        close = np.full(len(idx_arr), np.nan, dtype=float)

    # 9) Pack
    out_df = pd.DataFrame({
        "date": dates,
        "ticker": ticker if ticker is not None else (df["ticker"].iloc[0] if "ticker" in df.columns and not df.empty else None),
        "close": close,
        "proba": p,
        "signal": sig,
        "tau_used": np.full(len(sig), float(tau_used)),
    })
    preds = out_df.to_dict(orient="records")
    meta = {
        "asset_class": ac,
        "source": source,
        "ticker": ticker,
        "tau_used": float(tau_used),
        "fee_bps": int(fee_bps),
        "used_features": list(used_cols),
        "n_rows": int(len(sig)),
    }
    return preds, meta

# ---------------- API wiring ----------------
class PredictIn(BaseModel):
    asset_class: Literal["equity","crypto"]
    source: Literal["repo","fetch"]
    ticker: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    fee_bps: int = 5
    tau: Optional[float] = None

app = FastAPI(title="Direction Classifier API", version="0.2.0")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(inp: PredictIn):
    try:
        preds, meta = predict_core(
            asset_class=inp.asset_class,
            source=inp.source,
            ticker=inp.ticker,
            start=inp.start,
            end=inp.end,
            fee_bps=inp.fee_bps,
            tau=inp.tau,
        )
        return {"preds": preds, "meta": meta}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- selective override wrapper (auto-added by NB25) ---

try:
    _sel_wrapper_applied  # guard
except NameError:
    _orig_predict_core = predict_core

    def predict_core(asset_class="equity", source="repo", ticker=None,
                     start=None, end=None, fee_bps=5, tau=None):
        pred, meta = _orig_predict_core(asset_class=asset_class, source=source, ticker=ticker,
                                        start=start, end=end, fee_bps=fee_bps, tau=tau)

        import json, numpy as _np
        import pandas as _pd
        from pathlib import Path as _P

        # Locate selective config
        try:
            base = ROOT  # provided by app.config
        except NameError:
            # fall back: api.py is under app/  ⇒ parent.parent is repo root
            base = _P(__file__).resolve().parent.parent
        cfgp = (base / "artifacts" / "selective_config.json")
        sel = {}
        if cfgp.exists():
            try:
                sel = json.loads(cfgp.read_text(encoding="utf-8"))
            except Exception:
                sel = {}

        invert = bool(sel.get("invert_proba", False))
        tau_default = float(sel.get("global_tau", meta.get("tau_used", 0.59)))
        tau_used = float(tau) if tau is not None else tau_default

        # Normalize return to DataFrame for post-processing
        if isinstance(pred, list):
            dfp = _pd.DataFrame(pred)
            ret_type = "list"
        else:
            dfp = pred.copy()
            ret_type = "df"

        # Apply inversion and re-threshold if proba present
        if "proba" in dfp.columns:
            proba = _pd.to_numeric(dfp["proba"], errors="coerce").to_numpy()
            proba = _np.clip(proba, 1e-6, 1-1e-6)
            if invert:
                proba = 1.0 - proba
                dfp["proba"] = proba
            dfp["signal"] = (proba >= tau_used).astype(int)

        dfp["tau_used"] = float(tau_used)
        meta["tau_used"] = tau_used

        if ret_type == "list":
            return dfp.to_dict(orient="records"), meta
        else:
            return dfp, meta

    _sel_wrapper_applied = True
