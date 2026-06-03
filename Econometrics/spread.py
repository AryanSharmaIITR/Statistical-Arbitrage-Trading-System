from __future__ import annotations

import numpy as np
import pandas as pd


def build_spread(log_prices: pd.DataFrame, beta: np.ndarray) -> pd.Series:
    """spread_t = β' logP_t  (a single stationary series)."""
    beta = np.asarray(beta, float).ravel()
    return pd.Series(log_prices.values @ beta, index=log_prices.index,
                     name="spread")


def rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
    """Rolling z-score of the spread (mean/std over a trailing window)."""
    mu = spread.rolling(window, min_periods=window // 2).mean()
    sd = spread.rolling(window, min_periods=window // 2).std()
    z = (spread - mu) / sd.replace(0.0, np.nan)
    return z.rename("zscore")


def half_life(spread: pd.Series) -> float:
    """Mean-reversion half-life (bars) from an AR(1) fit on the spread."""
    s = spread.dropna().values
    if len(s) < 5:
        return float("inf")
    s_lag = s[:-1]
    ds = np.diff(s)
    X = np.column_stack([s_lag, np.ones_like(s_lag)])
    coef, *_ = np.linalg.lstsq(X, ds, rcond=None)
    lam = coef[0]
    if lam >= 0:
        return float("inf")
    return float(-np.log(2) / np.log(1 + lam))


def dollar_weights(beta: np.ndarray, prices: np.ndarray,
                   target_gross: float = 1.0) -> np.ndarray:
    w = np.asarray(beta, float).ravel()
    gross = np.abs(w).sum()
    if gross == 0:
        return w
    return w / gross * target_gross
