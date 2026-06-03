from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .johansen import run_johansen
from .spread import half_life, build_spread


@dataclass
class Relation:
    beta: np.ndarray      # normalised cointegrating vector (L1 = 1)
    rank: int             # rolling Johansen rank on the window
    half_life: float      # mean-reversion half-life of the chosen spread (bars)
    eig_index: int        # which eigenvector was chosen


def select_relation(
    log_window: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
    signif_idx: int = 1,
    method: str = "shortest_half_life",
    n_candidates: int = 3,
) -> Relation:
    jr = run_johansen(log_window, det_order, k_ar_diff, signif_idx)

    # Candidate eigenvectors: the validated relations if any, else the top few.
    n = max(jr.rank, n_candidates) if jr.rank > 0 else n_candidates
    n = min(n, jr.eigenvectors.shape[1])

    if method == "dominant":
        pick = 0
    else:
        best_hl, pick = np.inf, 0
        for j in range(n):
            sp = build_spread(log_window, jr.cointegrating_vector(j))
            hl = half_life(sp)
            if hl < best_hl:
                best_hl, pick = hl, j

    beta = jr.cointegrating_vector(pick)
    spread = build_spread(log_window, beta)
    return Relation(beta=beta, rank=jr.rank,
                    half_life=half_life(spread), eig_index=pick)
