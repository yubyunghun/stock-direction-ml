# src/predict.py
import argparse, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ART = Path("artifacts")
MODEL = ART / "lr.joblib"
SCALER = ART / "scaler.joblib"
FEAT = ART / "feature_list.json"
TAU = ART / "threshold.json"

def _fail_if_missing(p: Path, what: str):
    if not p.exists():
        raise FileNotFoundError(f"Missing {what}: {p}")

def conform(df: pd.DataFrame, cols):
    """Ensure required columns exist (missing -> 0.0), order them, coerce numeric."""
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X

def main():
    ap = argparse.ArgumentParser(description="Score rows with saved LR model; emit prob_up (+ optional signal).")
    ap.add_argument("--csv", required=True, help="Input CSV with feature columns.")
    ap.add_argument("--out", default="probs.csv", help="Output CSV path.")
    ap.add_argument("--emit-signal", action="store_true", help="Also output a binary signal column.")
    ap.add_argument("--threshold", type=float, default=None, help="Override τ (else read artifacts/threshold.json, else 0.5).")
    ap.add_argument("--id-cols", nargs="*", default=None, help="Optional columns to carry through unchanged (e.g., date ticker).")
    args = ap.parse_args()

    _fail_if_missing(MODEL, "model")
    _fail_if_missing(FEAT, "feature list")

    model = joblib.load(MODEL)
    scaler = joblib.load(SCALER) if SCALER.exists() else None
    feat = json.load(open(FEAT, "r"))

    # choose threshold
    if args.threshold is not None:
        tau = float(args.threshold)
    elif TAU.exists():
        try:
            tau = float(json.load(open(TAU, "r"))["LR"]["tau"])
        except Exception:
            tau = 0.5
    else:
        tau = 0.5

    df_in = pd.read_csv(args.csv)
    X = conform(df_in, feat)
    Xv = scaler.transform(X.values) if scaler is not None else X.values

    p = model.predict_proba(Xv)[:, 1]
    out = pd.DataFrame({"prob_up": p})

    if args.emit_signal:
        out["signal"] = (p > tau).astype(int)

    if args.id_cols:
        keep = [c for c in args.id_cols if c in df_in.columns]
        if keep:
            out = pd.concat([df_in[keep].reset_index(drop=True), out], axis=1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(out)} rows)"
          + (f" | threshold={tau:.2f}" if args.emit_signal else ""))

if __name__ == "__main__":
    main()
