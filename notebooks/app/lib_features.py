# app/lib_features.py
import math, numpy as np, pandas as pd

def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def rsi_wilder(close, length=14):
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100/(1+rs))

def macd_and_signal(close, fast=12, slow=26, sig=9):
    macd = ema(close, fast) - ema(close, slow)
    signal = ema(macd, sig)
    return macd, signal

def add_nb02_features(df):
    df = df.copy()
    ret1  = df["close"].pct_change(1)
    ret5  = df["close"].pct_change(5)
    ret10 = df["close"].pct_change(10)
    vol10 = ret1.rolling(10).std()
    volz  = (vol10 - vol10.rolling(100).mean()) / (vol10.rolling(100).std() + 1e-12)
    rsi14 = rsi_wilder(df["close"], 14)
    macd, macd_signal = macd_and_signal(df["close"], 12, 26, 9)

    df["ret1"]=ret1; df["ret5"]=ret5; df["ret10"]=ret10
    df["vol10"]=vol10; df["volz"]=volz
    df["rsi14"]=rsi14; df["macd"]=macd; df["macd_signal"]=macd_signal
    return df

def add_market_features_equity(df, spy_series, vix_series):
    m = df.merge(spy_series, on="date", how="left").merge(vix_series, on="date", how="left")
    m["mkt_ret1"] = m["spy_close"].pct_change(1)
    m["mkt_ret5"] = m["spy_close"].pct_change(5)
    m["vix_chg1"] = m["vix_close"].pct_change(1)
    return m

def add_market_features_crypto(df, bench_close):
    m = df.merge(bench_close, on="date", how="left").rename(columns={"btc_close":"spy_close"})
    m["mkt_ret1"]  = m["spy_close"].pct_change(1)
    m["mkt_ret5"]  = m["spy_close"].pct_change(5)
    btc_ret = m["spy_close"].pct_change()
    vix_proxy = btc_ret.rolling(30).std() * math.sqrt(365) * 100.0
    m["vix_close"] = vix_proxy
    m["vix_chg1"]  = m["vix_close"].pct_change(1)
    return m

def infer_target(df):
    for c in ["y","label","target","y_bin","direction","is_up","class","cls"]:
        if c in df.columns: return df[c].astype(int).clip(0,1).values, c
    if "ret_next" in df.columns:
        y = (df["ret_next"].astype(float) > 0).astype(int).values
        return y, "ret_next>0"
    if "close" in df.columns:
        rn = df["close"].astype(float).pct_change().shift(-1).fillna(0.0)
        df["ret_next"] = rn
        return (rn > 0).astype(int).values, "ret_next_from_close>0"
    return None, None

def make_dataset(df, feature_list):
    cols = [c for c in feature_list if c in df.columns]
    if not cols: raise ValueError("No overlap between feature_list.json and data columns.")
    tmp = df[cols].replace([np.inf,-np.inf], np.nan)
    y_vals, _ = infer_target(df)
    retn = df["ret_next"].astype(float).values if "ret_next" in df.columns else np.zeros(len(df))
    tmp["__y__"] = y_vals if y_vals is not None else np.nan
    tmp["__ret_next__"] = retn
    tmp = tmp.dropna()
    X = tmp[cols].to_numpy()
    y = (tmp["__y__"].astype(int).to_numpy() if y_vals is not None else None)
    retn = tmp["__ret_next__"].astype(float).to_numpy()
    idx = tmp.index
    return X, y, retn, idx, cols
