# app/lib_artifacts.py
import json, joblib
from .config import ART_DIR_EQUITY, ART_DIR_CRYPTO, DEFAULT_TAU, TAU_MAP_PATH

def _safe_tau(v, default):
    try: return float(v)
    except Exception: return default

def load_artifacts(asset_class="equity"):
    art = ART_DIR_EQUITY if asset_class == "equity" else ART_DIR_CRYPTO
    feature_list = json.loads((art/"feature_list.json").read_text(encoding="utf-8"))
    scaler = joblib.load(art/"scaler.joblib")
    model  = joblib.load(art/"lr.joblib")

    tau_art = DEFAULT_TAU
    tfile = art / "threshold.json"
    if tfile.exists():
        try:
            t = json.loads(tfile.read_text(encoding="utf-8"))
            tau_art = _safe_tau(t.get("tau") or t.get("threshold") or t.get("value"), DEFAULT_TAU)
        except Exception:
            pass

    tau_map = {}
    if asset_class == "equity" and TAU_MAP_PATH.exists():
        try:
            tau_map = json.loads(TAU_MAP_PATH.read_text(encoding="utf-8"))
        except Exception:
            tau_map = {}

    return feature_list, scaler, model, tau_art, tau_map
