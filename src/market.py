# src/market.py
from __future__ import annotations
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except Exception:
    yf = None


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join(map(str, c)).strip() if isinstance(c, tuple) else str(c)
            for c in df.columns
        ]
    else:
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
    return df


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a tz-naive datetime64[ns] column called 'date'."""
    df = df.reset_index().copy()  # always bring index out
    # find a datetime-like column to rename to 'date'
    dt_col = None
    for cand in ["date", "Date", "Datetime", "datetime", "index"]:
        if cand in df.columns:
            # try to coerce to datetime to verify
            maybe = pd.to_datetime(df[cand], errors="coerce")
            if maybe.notna().any():
                df[cand] = maybe
                dt_col = cand
                break
    if dt_col is None:
        # last resort: first column that can be parsed as datetime
        for c in df.columns:
            maybe = pd.to_datetime(df[c], errors="coerce")
            if maybe.notna().any():
                df[c] = maybe
                dt_col = c
                break
    if dt_col is None:
        raise ValueError("Could not locate a datetime column for 'date'.")

    df = df.rename(columns={dt_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    return _flatten_columns(df)


def _download(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily data and return ['date','close'] with a guaranteed 'date' column."""
    if yf is None:
        raise ImportError("yfinance is required. Install: pip install yfinance")

    r = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if r is None or r.empty:
        raise ValueError(f"No data returned for {ticker} in range {start}..{end}")

    # Ensure index is DatetimeIndex, tz-naive, then make it a column named 'date'
    if isinstance(r.index, pd.DatetimeIndex):
        try:
            # only tz_localize if tz-aware
            if r.index.tz is not None:
                r.index = r.index.tz_localize(None)
        except Exception:
            # some pandas versions store tz info differently; coerce via to_datetime
            r.index = pd.to_datetime(r.index, errors="coerce").tz_localize(None)
    else:
        # coerce whatever the index is into datetime (best effort)
        r.index = pd.to_datetime(r.index, errors="coerce").tz_localize(None)

    r = r.reset_index()
    # whatever the index column is called ('Date', 'Datetime', 'index', etc.), rename it to 'date'
    r = r.rename(columns={r.columns[0]: "date"})

    # choose a close-like column robustly
    close_col = None
    for cand in ["Adj Close", "Close", "close", "adj_close"]:
        if cand in r.columns:
            close_col = cand
            break
    if close_col is None:
        # case-insensitive fallback
        for c in r.columns:
            if c.lower().replace(" ", "") in ("close", "adjclose"):
                close_col = c
                break
    if close_col is None:
        raise ValueError("Could not find a close/adj close column in downloaded data.")

    out = r.loc[:, ["date", close_col]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    out = out.dropna(subset=["date"]).rename(columns={close_col: "close"}).reset_index(drop=True)
    return out

def add_market_context(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """
    Merge SPY & VIX context features by a plain 'date' column.
    Adds: spy_close, mkt_ret1, mkt_ret5, vix_close, vix_chg1
    """
    left = _ensure_date_column(_flatten_columns(df))

    if start is None:
        start = str(left["date"].min().date())
    if end is None:
        end = str(left["date"].max().date())

    spy = _download("SPY", start, end).rename(columns={"close": "spy_close"})
    vix = _download("^VIX", start, end).rename(columns={"close": "vix_close"})

    out = _safe_merge_on_date(left, spy, suffix="_spy")
    out = _safe_merge_on_date(out, vix, suffix="_vix")

    # context features
    out["mkt_ret1"] = out["spy_close"].pct_change(1)
    out["mkt_ret5"] = out["spy_close"].pct_change(5)
    out["vix_chg1"] = out["vix_close"].pct_change(1)

    return out


MARKET_FEATURES = ["spy_close", "mkt_ret1", "mkt_ret5", "vix_close", "vix_chg1"]
