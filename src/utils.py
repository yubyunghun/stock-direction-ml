# src/utils.py
import numpy as np

def make_labels(df, tau=0.0, dead_zone=False):
    """
    Create next-day direction label.

    Args:
        df: DataFrame with a 'close' column.
        tau: threshold for returns (e.g., 0.001 = 0.1%).
        dead_zone: if True and tau>0, drop tiny moves between [-tau, +tau].

    Returns:
        DataFrame with added columns:
            - ret_next: next-day simple return
            - y: binary label (1 up / 0 down). If dead_zone=True, rows with |ret_next| <= tau are dropped.
    """
    df = df.copy()
    # next-day simple return (no leakage)
    df["ret_next"] = df["close"].shift(-1) / df["close"] - 1

    if dead_zone and tau > 0:
        # 1 if > +tau, 0 if < -tau, else drop
        y_raw = np.where(df["ret_next"] >  tau, 1,
                 np.where(df["ret_next"] < -tau, 0, np.nan))
        df["y"] = y_raw
        df = df.dropna(subset=["y"])
        df["y"] = df["y"].astype(int)
    else:
        df["y"] = (df["ret_next"] > tau).astype(int)

    # drop the last row (shift created NaN) and reindex
    return df.dropna().reset_index(drop=True)
