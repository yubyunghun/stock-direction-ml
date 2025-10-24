# app/lib_fetch.py
import pandas as pd
from .config import DATA_CSV, DATA_PQ
from .lib_features import add_nb02_features

def ensure_date(df):
    for c in ["date","Date","timestamp","ts"]:
        if c in df.columns:
            try: df[c] = pd.to_datetime(df[c])
            except: pass
            if c != "date": df["date"] = df[c]
            return df
    return df

def load_repo_df():
    if DATA_CSV.exists(): df = pd.read_csv(DATA_CSV)
    elif DATA_PQ.exists(): df = pd.read_parquet(DATA_PQ)
    else: raise FileNotFoundError("Missing data/df_nb02.csv or .parquet")
    return ensure_date(df)

def fetch_equity_df(ticker, start, end):
    import yfinance as yf
    px = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if px.empty: raise ValueError(f"No data for {ticker}")
    df = px.rename_axis("date").reset_index()
    df["date"]=pd.to_datetime(df["date"])
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    df = df[["date","open","high","low","close","volume"]]; df["ticker"]=ticker
    df = add_nb02_features(df)

    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False).rename_axis("date").reset_index()[["date","Close"]].rename(columns={"Close":"spy_close"})
    vix = yf.download("^VIX", start=start, end=end, progress=False).rename_axis("date").reset_index()[["date","Close"]].rename(columns={"Close":"vix_close"})
    spy["date"]=pd.to_datetime(spy["date"]); vix["date"]=pd.to_datetime(vix["date"])

    df = df.merge(spy,on="date",how="left").merge(vix,on="date",how="left")
    df["mkt_ret1"]=df["spy_close"].pct_change(1); df["mkt_ret5"]=df["spy_close"].pct_change(5)
    df["vix_chg1"]=df["vix_close"].pct_change(1)
    df["ret_next"]=df["close"].pct_change().shift(-1); df["y"]=(df["ret_next"]>0).astype(int)
    return df

def fetch_crypto_df(ticker, start, end):
    import yfinance as yf
    px = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if px.empty: raise ValueError(f"No data for {ticker}")
    df = px.rename_axis("date").reset_index()
    df["date"]=pd.to_datetime(df["date"])
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    df = df[["date","open","high","low","close","volume"]]; df["ticker"]=ticker
    df = add_nb02_features(df)

    btc = yf.download("BTC-USD", start=start, end=end, auto_adjust=True, progress=False).rename_axis("date").reset_index()[["date","Close"]].rename(columns={"Close":"btc_close"})
    btc["date"]=pd.to_datetime(btc["date"])
    df = df.merge(btc,on="date",how="left").rename(columns={"btc_close":"spy_close"})
    df["mkt_ret1"]=df["spy_close"].pct_change(1); df["mkt_ret5"]=df["spy_close"].pct_change(5)

    ret = df["spy_close"].pct_change()
    vix_proxy = ret.rolling(30).std() * (365 ** 0.5) * 100.0
    df["vix_close"]=vix_proxy; df["vix_chg1"]=df["vix_close"].pct_change(1)

    df["ret_next"]=df["close"].pct_change().shift(-1); df["y"]=(df["ret_next"]>0).astype(int)
    return df
