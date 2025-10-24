# app/lib_eval.py
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss, f1_score
)

def predict_proba(model, X):
    if hasattr(model,"predict_proba"):
        p = model.predict_proba(X)
        return p[:,1] if p.ndim==2 else p
    if hasattr(model,"decision_function"):
        s = model.decision_function(X); return 1/(1+np.exp(-s))
    return np.clip(model.predict(X).astype(float), 0, 1)

def metrics_all(y, p):
    def safe(fn,*a):
        try: return float(fn(*a))
        except: return float("nan")
    return dict(
        auc     = safe(roc_auc_score, y, p),
        ap      = safe(average_precision_score, y, p),
        brier   = safe(brier_score_loss, y, p),
        logloss = safe(log_loss, y, p),
    )

def tau_sweep(y, p, retn, fee_bps=5, grid=None):
    import numpy as np
    if grid is None: grid = np.linspace(0.05, 0.95, 91)
    f1s, finals = [], []
    for t in grid:
        sig = (p >= t).astype(int)
        f1s.append(_safe_f1(y, sig))
        finals.append(_final_equity(retn, sig, fee_bps))
    return grid, np.array(f1s), np.array(finals)

def _safe_f1(y, sig):
    try: return f1_score(y, sig)
    except: return float("nan")

def _final_equity(retn, sig, fee_bps):
    import numpy as np
    flips = np.zeros_like(sig)
    if len(flips)>1: flips[1:] = (sig[1:] != sig[:-1]).astype(int)
    fee = flips * (fee_bps/10000.0)
    eq  = np.cumprod(1 + (retn*sig - fee))
    return float(eq[-1]) if len(eq) else float("nan")
