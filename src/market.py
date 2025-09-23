# src/market.py
# -----------------------------------------------------------------------------
# Adds broad-market context features (SPY, VIX) to your per-ticker dataframe.
# Robust to MultiIndex, index-on-date, or 'Date' vs 'date' column naming.
# Merge is done on a plain 'date' column to avoid pandas "different levels" errors.
# -----------------------------------------------------------------------------

from __future__ import annotations
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except Exception:
    yf = None


def _download(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily data for a ticker and return columns: ['date', 'close'].
    Uses yfinance with auto_adjust=True (dividends/splits adjusted).
    """
    if yf is None:
        raise ImportError(
            "yfinance is required to fetch market context.\n"
            "Install it with:  pip install yfinance"
        )

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker} in range {start}..{end}")

    # yfinance returns Date index + 'Close' column (capitalized)
    out = df.reset_index().rename(columns={"Date": "date", "Close": "close"})
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out.loc[:, ["date", "close"]]


def add_market_context(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """
    Merge SPY & VIX context features by a plain 'date' column.
    Adds columns:
      - spy_close
      - mkt_ret1, mkt_ret5
      - vix_close
      - vix_chg1

    Parameters
    ----------
    df : pd.DataFrame
        Your per-ticker frame. Can have date in index (any type) or as 'date'/'Date' column.
    start, end : str (YYYY-MM-DD), optional
        Fetch window. If omitted, inferred from min/max date in `df`.

    Returns
    -------
    pd.DataFrame
        Input dataframe with market columns merged on 'date'.
    """
    # ---- 1) Flatten any index -> plain columns (avoids "different levels" merge errors) ----
    left = df.copy()
    left = left.reset_index()  # unconditional: brings ALL index levels out

    # ---- 2) Ensure we have a single 'date' column (rename if needed) ----
    # Prefer an existing 'date'; otherwise accept 'Date'.
    if "date" not in left.columns:
        if "Date" in left.columns:
            left = left.rename(columns={"Date": "date"})
        else:
            raise ValueError("No 'date' column found after reset_index().")

    # Normalize 'date' to naive datetime
    left["date"] = pd.to_datetime(left["date"]).dt.tz_localize(None)

    # ---- 3) Determine span if not provided ----
    if start is None:
        start = str(left["date"].min().date())
    if end is None:
        end = str(left["date"].max().date())

    # ---- 4) Download SPY & VIX series, normalize ----
    spy = _download("SPY", start, end).rename(columns={"close": "spy_close"})
    vix = _download("^VIX", start, end).rename(columns={"close": "vix_close"})
    spy["date"] = pd.to_datetime(spy["date"]).dt.tz_localize(None)
    vix["date"] = pd.to_datetime(vix["date"]).dt.tz_localize(None)

    # ---- 5) Merge on *column* 'date' (NOT index) to avoid level mismatches ----
    out = (
        left.merge(spy, on="date", how="left", sort=False)
            .merge(vix, on="date", how="left", sort=False)
    )

    # ---- 6) Context features ----
    out["mkt_ret1"] = out["spy_close"].pct_change(1)
    out["mkt_ret5"] = out["spy_close"].pct_change(5)
    out["vix_chg1"] = out["vix_close"].pct_change(1)

    return out


# Optional: export the names of features added by this module
MARKET_FEATURES = ["spy_close", "mkt_ret1", "mkt_ret5", "vix_close", "vix_chg1"]
