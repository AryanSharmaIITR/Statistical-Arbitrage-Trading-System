"""
Cointegration-breakdown protocol.

A cointegrating relationship is not permanent — a structural break (earnings
shock, index reconstitution, M&A) can destroy the equilibrium. Trading a spread
that no longer mean-reverts is how stat-arb books blow up. This monitor re-runs
the Johansen test on a rolling window and flags a breakdown when **both**:

  * the cointegration rank collapses below ``min_rank`` (no validated relation), and
  * the traded spread's half-life blows out beyond ``max_half_life``,

and stays that way for ``confirm`` consecutive checks (debounce). On a confirmed
breakdown the engine liquidates the book and pauses new entries until the
relationship re-validates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from Econometrics import run_johansen, build_spread, half_life


@dataclass
class BreakdownState:
    rank: int
    half_life: float
    breakdown: bool          # confirmed breakdown -> liquidate / pause
    consecutive_bad: int


class BreakdownMonitor:
    def __init__(self, min_rank: int = config.MIN_RANK,
                 max_half_life: float = config.MAX_HALF_LIFE,
                 confirm: int = config.BREAKDOWN_CONFIRM,
                 det_order: int = config.JOHANSEN_DET_ORDER,
                 k_ar_diff: int = config.K_AR_DIFF,
                 signif_idx: int = config.SIGNIF_IDX):
        self.min_rank = min_rank
        self.max_half_life = max_half_life
        self.confirm = confirm
        self.det_order = det_order
        self.k_ar_diff = k_ar_diff
        self.signif_idx = signif_idx
        self._bad = 0

    def check(self, log_window: pd.DataFrame, beta: np.ndarray) -> BreakdownState:
        jr = run_johansen(log_window, self.det_order, self.k_ar_diff,
                          self.signif_idx)
        spread = build_spread(log_window, beta)
        hl = half_life(spread)

        rank_bad = jr.rank < self.min_rank
        hl_bad = (not np.isfinite(hl)) or hl > self.max_half_life
        if rank_bad and hl_bad:
            self._bad += 1
        else:
            self._bad = 0

        breakdown = self._bad >= self.confirm
        return BreakdownState(rank=jr.rank, half_life=hl,
                              breakdown=breakdown, consecutive_bad=self._bad)
