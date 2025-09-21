# src/features.py
import pandas as pd
import numpy as np
import ta  # pip install ta

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add returns, volatility, and technical indicators.
    """
    df = df.copy()
    
    # Returns
    df["ret1"] = df["close"].pct_change()
    df["ret5"] = df["close"].pct_change(5)
    df["ret10"] = df["close"].pct_change(10)
    
    # Rolling volatility
    df["vol10"] = df["ret1"].rolling(10).std()
    
    # Volume z-score
    df["volz"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()
    
    # Technical indicators
    df["rsi14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    
    # Drop NaN
    df = df.dropna().reset_index(drop=True)
    return df
