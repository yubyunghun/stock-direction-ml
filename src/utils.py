# src/utils.py
import pandas as pd

def make_labels(df: pd.DataFrame, tau: float = 0.0) -> pd.DataFrame:
    """
    Create binary labels: 1 if next-day return > tau, else 0.
    """
    df = df.copy()
    df["ret_next"] = df["close"].shift(-1) / df["close"] - 1
    df["y"] = (df["ret_next"] > tau).astype(int)
    df = df.dropna().reset_index(drop=True)
    return df
