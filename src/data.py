# src/data.py
import pandas as pd
import yfinance as yf

def _flatten_col(c):
    """Turn a column (possibly a tuple) into a clean lowercase string."""
    if isinstance(c, tuple):
        parts = [str(p).strip().lower() for p in c if str(p).strip() != ""]
        return "_".join(parts)
    return str(c).strip().lower()

def _pick_cols(cols):
    """
    Given a list of flattened column names, pick canonical OHLCV.
    Robust to names like: 'open', 'open_aapl', 'aapl_open', 'adj close', 'adj_close', etc.
    Returns a dict mapping canonical -> actual column name in df.
    """
    cols_l = [c.lower() for c in cols]
    pick = {}

    # date
    for c in cols_l:
        if c == "date" or c.endswith("_date"):
            pick["date"] = c
            break

    # adj close first (preferred for 'close' if present)
    adj_close_name = None
    for c in cols_l:
        if "adj close" in c or "adj_close" in c or "adjusted close" in c:
            adj_close_name = c
            break

    # close
    close_name = None
    for c in cols_l:
        # avoid picking adj close again as 'close'
        if ("close" in c) and (adj_close_name is None or c != adj_close_name):
            close_name = c
            break

    # open/high/low/volume
    def find_like(token):
        for c in cols_l:
            if token in c:
                return c
        return None

    open_name   = find_like("open")
    high_name   = find_like("high")
    low_name    = find_like("low")
    volume_name = find_like("volume")

    # Build mapping, prefer adj close when available
    if adj_close_name is not None:
        pick["close"] = adj_close_name
    elif close_name is not None:
        pick["close"] = close_name

    if open_name:   pick["open"]   = open_name
    if high_name:   pick["high"]   = high_name
    if low_name:    pick["low"]    = low_name
    if volume_name: pick["volume"] = volume_name

    return pick

def get_data(ticker="AAPL", start="2012-01-01", end=None):
    """
    Download daily OHLCV data from Yahoo Finance and return a DataFrame with:
    ['date','open','high','low','close','volume']
    (Uses Adjusted Close as Close when available.)
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="column",        # help avoid multiindex, but we still handle it if it appears
        progress=False,
    )

    # Ensure Date is a column, not an index
    df = df.reset_index()

    # Flatten/normalize column names
    flat_cols = [_flatten_col(c) for c in df.columns]
    df.columns = flat_cols

    # Pick actual columns present
    picks = _pick_cols(df.columns)

    # Helpful error if something is missing
    needed = ["date", "open", "high", "low", "close", "volume"]
    missing = [k for k in needed if k not in picks]
    if missing:
        raise KeyError(
            f"Could not find columns: {missing}\n"
            f"Available columns after flattening:\n{df.columns.tolist()}\n"
            f"Auto-picks so far: {picks}"
        )

    # Slice and rename to canonical
    out = df[[picks["date"], picks["open"], picks["high"], picks["low"], picks["close"], picks["volume"]]].copy()
    out.columns = ["date", "open", "high", "low", "close", "volume"]

    # Types
    out["date"] = pd.to_datetime(out["date"])
    return out

if __name__ == "__main__":
    t = get_data("AAPL", "2015-01-01", "2023-12-31")
    print(t.head())
    print(t.columns.tolist())
