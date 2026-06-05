from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import config

@dataclass
class CostBreakdown:
    fees: float
    impact: float
    borrow: float

    @property
    def total(self) -> float:
        return self.fees + self.impact + self.borrow


def fee_cost(trade_notional: np.ndarray, taker: bool = True) -> float:
    bps = (config.TAKER_FEE_BPS if taker else config.MAKER_FEE_BPS) \
        + config.COMMISSION_BPS
    return float(np.abs(trade_notional).sum() * bps * 1e-4)


def borrow_cost(positions_notional: np.ndarray, borrow_rate: np.ndarray,
                days: int = 1) -> float:
    shorts = np.maximum(-positions_notional, 0.0)
    daily_rate = borrow_rate / config.TRADING_DAYS
    return float((shorts * daily_rate * days).sum())


def market_impact(trade_notional: np.ndarray, adv_notional: np.ndarray,
                  sigma_daily: np.ndarray, coef: float = config.IMPACT_COEF
                  ) -> float:
    q = np.abs(trade_notional)
    adv = np.maximum(adv_notional, 1.0)
    participation = q / adv
    impact_frac = coef * sigma_daily * np.sqrt(participation)
    return float((impact_frac * q).sum())


def total_costs(trade_notional, positions_notional, adv_notional, sigma_daily,
                borrow_rate, taker: bool = True, days: int = 1) -> CostBreakdown:
    return CostBreakdown(
        fees=fee_cost(trade_notional, taker),
        impact=market_impact(trade_notional, adv_notional, sigma_daily),
        borrow=borrow_cost(positions_notional, borrow_rate, days),
    )
