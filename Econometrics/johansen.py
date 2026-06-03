from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen


@dataclass
class JohansenResult:
    rank: int                 # selected cointegration rank
    eigenvalues: np.ndarray   # eigenvalues, descending
    eigenvectors: np.ndarray  # columns are cointegrating vectors (β), k x k
    trace_stat: np.ndarray    # trace statistics per null hypothesis
    trace_crit: np.ndarray    # critical values at the chosen significance level
    assets: list[str]
    signif_label: str

    def cointegrating_vector(self, idx: int = 0) -> np.ndarray:
        beta = self.eigenvectors[:, idx].astype(float).copy()
        if beta[np.argmax(np.abs(beta))] < 0:
            beta = -beta
        l1 = np.abs(beta).sum()
        return beta / l1 if l1 > 0 else beta

    def summary(self) -> str:
        lines = [
            f"Johansen trace test  ({self.signif_label} critical values)",
            f"  universe (k={len(self.assets)}): {', '.join(self.assets)}",
            "  H0          trace      crit       reject?",
        ]
        for i in range(len(self.trace_stat)):
            rej = self.trace_stat[i] > self.trace_crit[i]
            lines.append(
                f"  rank<={i:<2d}   {self.trace_stat[i]:9.2f}  "
                f"{self.trace_crit[i]:9.2f}   {'YES' if rej else 'no'}"
            )
        lines.append(f"  => selected cointegration rank r = {self.rank}")
        return "\n".join(lines)


def run_johansen(
    log_prices: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
    signif_idx: int = 1,
) -> JohansenResult:

    labels = {0: "90%", 1: "95%", 2: "99%"}
    res = coint_johansen(log_prices.values, det_order, k_ar_diff)

    trace_stat = res.lr1
    trace_crit = res.cvt[:, signif_idx]

    # Sequential trace test: accept the largest r still rejected.
    rank = 0
    for i in range(len(trace_stat)):
        if trace_stat[i] > trace_crit[i]:
            rank = i + 1
        else:
            break

    # Sort eigenvectors by eigenvalue magnitude (descending) for stability.
    order = np.argsort(res.eig)[::-1]
    return JohansenResult(
        rank=rank,
        eigenvalues=res.eig[order],
        eigenvectors=res.evec[:, order],
        trace_stat=trace_stat,
        trace_crit=trace_crit,
        assets=list(log_prices.columns),
        signif_label=labels.get(signif_idx, str(signif_idx)),
    )
