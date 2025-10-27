# scripts/predict_cli.py — programmatic predictions via CLI
import argparse
from pathlib import Path
import pandas as pd
from app.api import predict_core  # reuse API core

def main():
    ap = argparse.ArgumentParser(description="Direction Classifier CLI")
    ap.add_argument("--asset", choices=["equity","crypto"], default="equity")
    ap.add_argument("--source", choices=["repo","fetch"], default="repo")
    ap.add_argument("--ticker", type=str, help="Ticker (required for fetch; optional for repo)")
    ap.add_argument("--start", type=str, default=None, help="YYYY-MM-DD (fetch only)")
    ap.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (fetch only)")
    ap.add_argument("--fee-bps", type=int, default=5)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--out", type=str, default=None, help="Output CSV path (default under reports/demo)")
    args = ap.parse_args()

    start = pd.to_datetime(args.start).date() if args.start else None
    end   = pd.to_datetime(args.end).date() if args.end else None

    df, meta = predict_core(
        asset_class=args.asset,
        source=args.source,
        ticker=args.ticker,
        start=start,
        end=end,
        fee_bps=args.fee_bps,
        tau=args.tau,
    )

    out = Path(args.out) if args.out else (Path("reports")/"demo"/f"preds_{args.asset}_{(args.ticker or 'UNKNOWN').replace('-','')}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("Wrote:", out.resolve())
    print("Meta:", meta)

if __name__ == "__main__":
    main()
