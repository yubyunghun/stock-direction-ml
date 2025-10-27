# app/lib_fetch.py
import warnings
import pandas as pd
from .config import DATA_CSV, DATA_PQ
from .lib_features import add_nb02_features

# Optional: quiet pandas PerformanceWarning on some merges/sorts
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

def ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["date", "Date", "timestamp", "ts"]:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c])
            except Exception:
                pass
            if c != "date":
                df["date"] = df[c]
            break
    return df

def load_repo_df() -> pd.DataFrame:
    if DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
    elif DATA_PQ.exists():
        df = pd.read_parquet(DATA_PQ)
    else:
        raise FileNotFoundError("Missing data/df_nb02.csv or .parquet")
    return ensure_date(df)

def fetch_equity_df(ticker: str, start, end) -> pd.DataFrame:
    import yfinance as yf
    px = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if px.empty:
        raise ValueError(f"No data for {ticker}")

    df = (
        px.rename_axis("date").reset_index()
          .assign(date=lambda x: pd.to_datetime(x["date"]))
          .rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
          .loc[:, ["date","open","high","low","close","volume"]]
    )
    df["ticker"] = ticker
    df = add_nb02_features(df)

    # Market & risk
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    vix = yf.download("^VIX", start=start, end=end, progress=False)

    spy = (
        spy.rename_axis("date").reset_index()[["date","Close"]]
           .rename(columns={"Close":"spy_close"})
    )
    vix = (
        vix.rename_axis("date").reset_index()[["date","Close"]]
           .rename(columns={"Close":"vix_close"})
    )

    spy["date"] = pd.to_datetime(spy["date"])
    vix["date"] = pd.to_datetime(vix["date"])

    df = df.sort_values("date").reset_index(drop=True)
    spy = spy.sort_values("date").reset_index(drop=True)
    vix = vix.sort_values("date").reset_index(drop=True)

    df = df.merge(spy, on="date", how="left").merge(vix, on="date", how="left")

    # Fill early gaps to avoid NaNs at the head
    df[["close","spy_close","vix_close"]] = df[["close","spy_close","vix_close"]].astype(float)
    df[["spy_close","vix_close"]] = df[["spy_close","vix_close"]].ffill()

    df["mkt_ret1"] = df["spy_close"].pct_change(1)
    df["mkt_ret5"] = df["spy_close"].pct_change(5)
    df["vix_chg1"] = df["vix_close"].pct_change(1)

    df["ret_next"] = df["close"].pct_change().shift(-1)
    df["y"] = (df["ret_next"] > 0).astype(int)
    return df

def fetch_crypto_df(ticker: str, start, end) -> pd.DataFrame:
    """
    Crypto: use BTC-USD as the 'market proxy' to populate spy_close & mkt_ret*,
    and compute a VIX-like proxy from BTC returns (30d rolling annualized vol).
    """
    import yfinance as yf
    px = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if px.empty:
        raise ValueError(f"No data for {ticker}")

    df = (
        px.rename_axis("date").reset_index()
          .assign(date=lambda x: pd.to_datetime(x["date"]))
          .rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
          .loc[:, ["date","open","high","low","close","volume"]]
    )
    df["ticker"] = ticker
    df = add_nb02_features(df)

    # Market proxy = BTC-USD (even if ticker is BTC-USD, this still works)
    btc = yf.download("BTC-USD", start=start, end=end, auto_adjust=True, progress=False)
    btc = (
        btc.rename_axis("date").reset_index()[["date","Close"]]
           .rename(columns={"Close":"btc_close"})
    )
    btc["date"] = pd.to_datetime(btc["date"])

    df  = df.sort_values("date").reset_index(drop=True)
    btc = btc.sort_values("date").reset_index(drop=True)
    df = df.merge(btc, on="date", how="left").rename(columns={"btc_close":"spy_close"})

    # Ensure numeric + fill any leading gaps on proxy
    df[["close","spy_close"]] = df[["close","spy_close"]].astype(float)
    df["spy_close"] = df["spy_close"].ffill()

    # Market returns from proxy
    df["mkt_ret1"] = df["spy_close"].pct_change(1)
    df["mkt_ret5"] = df["spy_close"].pct_change(5)

    # VIX-like proxy from proxy returns: 30d rolling stdev (annualized %) 
    ret = df["spy_close"].pct_change()
    vix_proxy = ret.rolling(30, min_periods=5).std() * (365 ** 0.5) * 100.0
    df["vix_close"] = vix_proxy.astype(float)
    df["vix_chg1"] = df["vix_close"].pct_change(1)

    df["ret_next"] = df["close"].pct_change().shift(-1)
    df["y"] = (df["ret_next"] > 0).astype(int)
    return df
