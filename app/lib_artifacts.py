# app/lib_artifacts.py
import json, joblib
from .config import ART_DIR_EQUITY, ART_DIR_CRYPTO, DEFAULT_TAU

def _safe_tau(v, default):
    try: return float(v)
    except Exception: return default

def load_artifacts(asset_class="equity"):
    art = ART_DIR_EQUITY if asset_class == "equity" else ART_DIR_CRYPTO
    feature_list = json.loads((art/"feature_list.json").read_text(encoding="utf-8"))
    scaler = joblib.load(art/"scaler.joblib")
    model  = joblib.load(art/"lr.joblib")

    
    # NB26: prefer calibrated model if present
    try:
        import os, joblib, json
        from pathlib import Path as _P
        base = _P(art)
        cand = base/"lr_calibrated.joblib"
        if cand.exists():
            model = joblib.load(cand)
    except Exception:
        pass
tau_art = DEFAULT_TAU
    tfile = art / "threshold.json"
    if tfile.exists():
        try:
            t = json.loads(tfile.read_text(encoding="utf-8"))
            tau_art = _safe_tau(t.get("tau") or t.get("threshold") or t.get("value"), DEFAULT_TAU)
        except Exception:
            pass

    # NEW: always try per-folder tau_map.json (equity or crypto)
    tau_map = {}
    tmap = art / "tau_map.json"
    if tmap.exists():
        try:
            tau_map = json.loads(tmap.read_text(encoding="utf-8"))
        except Exception:
            tau_map = {}

    return feature_list, scaler, model, tau_art, tau_map
