# src/predict.py
import argparse, json, joblib, pandas as pd
from pathlib import Path

ART = Path("artifacts")
MODEL = ART/"lr.joblib"
SCALER = ART/"scaler.joblib"
FEAT = ART/"feature_list.json"
TAU = ART/"threshold.json"

def conform(df, cols):
    for c in cols:
        if c not in df.columns: df[c] = 0.0
    return df[cols].copy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="probs.csv")
    args = ap.parse_args()

    model = joblib.load(MODEL)
    scaler = joblib.load(SCALER) if SCALER.exists() else None
    feat = json.load(open(FEAT))
    tau = json.load(open(TAU))["LR"]["tau"] if TAU.exists() else 0.55

    X = pd.read_csv(args.csv)
    X = conform(X, feat)
    Xv = scaler.transform(X.values) if scaler is not None else X.values
    p = model.predict_proba(Xv)[:,1]
    pd.DataFrame({"prob_up": p, "signal": (p>tau).astype(int)}).to_csv(args.out, index=False)
    print(f"Wrote {args.out}  (τ={tau:.2f})")

if __name__ == "__main__":
    main()
