from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM, select_order


@dataclass
class VECMFit:
    beta: np.ndarray            # k x r cointegrating vectors (hedge ratios)
    alpha: np.ndarray           # k x r speed-of-adjustment matrix
    rank: int
    k_ar_diff: int
    assets: list[str]
    half_lives: np.ndarray      # per-relation half-life of mean reversion (bars)

    def hedge_ratios(self, idx: int = 0) -> pd.Series:
        """β for the idx-th relation as a labelled Series."""
        return pd.Series(self.beta[:, idx], index=self.assets, name="beta")

    def adjustment_speeds(self, idx: int = 0) -> pd.Series:
        """α for the idx-th relation as a labelled Series."""
        return pd.Series(self.alpha[:, idx], index=self.assets, name="alpha")

    def summary(self) -> str:
        lines = [f"VECM(rank={self.rank}, k_ar_diff={self.k_ar_diff})"]
        for j in range(self.rank):
            hl = self.half_lives[j]
            lines.append(f"  relation {j}: half-life ≈ {hl:.1f} bars")
            b = self.hedge_ratios(j).round(3).to_dict()
            lines.append(f"    β: {b}")
        return "\n".join(lines)


def select_lag_order(log_prices: pd.DataFrame, maxlags: int = 10,
                     det_order: int = 0) -> int:
    """Pick the number of lagged differences (k_ar_diff) by AIC."""
    sel = select_order(log_prices.values, maxlags=maxlags,
                        deterministic="ci" if det_order == 0 else "nc")
    # select_order reports the VAR order in levels; VECM uses one fewer diff.
    return max(1, int(sel.aic) - 1) if sel.aic else 1


def _half_life(series: np.ndarray) -> float:
    """Half-life of mean reversion from an AR(1) fit: Δs_t = λ s_{t-1} + c."""
    s = np.asarray(series, float)
    s_lag = s[:-1]
    ds = np.diff(s)
    X = np.column_stack([s_lag, np.ones_like(s_lag)])
    # OLS coefficient on the lagged level.
    coef, *_ = np.linalg.lstsq(X, ds, rcond=None)
    lam = coef[0]
    if lam >= 0:                      # no mean reversion detected
        return float("inf")
    return float(-np.log(2) / np.log(1 + lam))


def fit_vecm(
    log_prices: pd.DataFrame,
    rank: int,
    k_ar_diff: int = 1,
    det_order: int = 0,
) -> VECMFit:
    """Fit a VECM of the given rank and return β, α and per-relation half-lives."""
    deterministic = "ci" if det_order == 0 else ("co" if det_order == 1 else "n")
    model = VECM(
        log_prices.values,
        k_ar_diff=k_ar_diff,
        coint_rank=rank,
        deterministic=deterministic,
    )
    res = model.fit()

    beta = np.asarray(res.beta)         # k x r
    alpha = np.asarray(res.alpha)       # k x r

    # Half-life of each cointegrating relation's spread.
    spreads = log_prices.values @ beta  # T x r
    half_lives = np.array([_half_life(spreads[:, j]) for j in range(rank)])

    return VECMFit(
        beta=beta,
        alpha=alpha,
        rank=rank,
        k_ar_diff=k_ar_diff,
        assets=list(log_prices.columns),
        half_lives=half_lives,
    )
