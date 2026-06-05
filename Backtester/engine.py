from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import config
from Econometrics import select_relation, build_spread
from Risk import optimize_hedge, BreakdownMonitor
from . import microstructure as ms


@dataclass
class BacktestResult:
    timeline: pd.DataFrame
    trades: int
    aum: float
    metadata: dict = field(default_factory=dict)

    @property
    def net_returns(self) -> pd.Series:
        return self.timeline["net_ret"]

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.timeline["net_ret"]).cumprod().rename("equity")


def _target_weights(beta: np.ndarray, direction: int) -> np.ndarray:
    g = np.abs(beta).sum()
    w = beta / g * config.TARGET_GROSS if g > 0 else beta
    w = np.clip(w, -config.MAX_WEIGHT, config.MAX_WEIGHT)
    return direction * w


def run_backtest(
    panels: dict,
    aum: float = 1e7,
    halts: dict[str, set] | None = None,
    hard_to_borrow: set[str] | None = None,
    use_hedge_optimizer: bool = True,
    verbose: bool = False,
) -> BacktestResult:
    prices = panels["prices"]
    log_prices = panels["log_prices"]
    adv = panels["adv"]
    assets = list(prices.columns)
    k = len(assets)
    rets = prices.pct_change().fillna(0.0)
    halts = halts or {}
    hard_to_borrow = hard_to_borrow or set()

    borrow_rate = np.array([
        config.HARD_TO_BORROW_RATE if a in hard_to_borrow
        else config.DEFAULT_BORROW_RATE for a in assets
    ])

    monitor = BreakdownMonitor()
    n = len(prices)
    start = config.TRAIN_WINDOW

    w_prev = np.zeros(k)
    beta = None
    beta_candidate = None
    rank, half_life = 0, np.inf
    breakdown = False
    direction = 0
    n_trades = 0
    rows = []

    for t in range(start, n):
        ts = prices.index[t]
        r_t = rets.iloc[t].values
        gross_ret = float(w_prev @ r_t)

        log_win = log_prices.iloc[t - config.TRAIN_WINDOW:t]

        if beta_candidate is None or (t - start) % config.REFIT_EVERY == 0:
            rel = select_relation(log_win, method=config.SELECT_BY)
            beta_candidate, rank, half_life = rel.beta, rel.rank, rel.half_life

        if direction == 0:
            beta = beta_candidate

        if (t - start) % config.BREAKDOWN_CHECK_EVERY == 0:
            st = monitor.check(log_win, beta)
            rank, half_life = st.rank, st.half_life
            breakdown = st.breakdown

        spread = build_spread(log_prices.iloc[t - config.ZSCORE_WINDOW:t + 1], beta)
        cur, hist = spread.iloc[-1], spread.iloc[:-1]
        sd = hist.std()
        z = float((cur - hist.mean()) / sd) if sd and np.isfinite(sd) else 0.0

        hl_ok = (not config.HALF_LIFE_FILTER) or \
            (config.MIN_HALF_LIFE <= half_life <= config.MAX_HALF_LIFE)
        if breakdown:
            direction = 0                                   # liquidate & pause
        elif direction == 0:
            if hl_ok and z > config.ENTRY_Z:
                direction = -1
            elif hl_ok and z < -config.ENTRY_Z:
                direction = +1
        else:
            if abs(z) < config.EXIT_Z or abs(z) > config.STOP_Z:
                direction = 0

        w_target = (_target_weights(beta, direction) if direction != 0
                    else np.zeros(k))

        halted_mask = np.array([ts in halts.get(a, set()) for a in assets])
        constrained = halted_mask.any() or bool(hard_to_borrow)
        if use_hedge_optimizer and direction != 0 and constrained:
            cov = rets.iloc[t - config.ADV_WINDOW:t].cov().values * config.TRADING_DAYS
            sol = optimize_hedge(w_target, cov, halted=halted_mask,
                                 borrow_rate=borrow_rate,
                                 target_gross=config.TARGET_GROSS)
            w_target = sol.weights
        elif halted_mask.any():
            w_target = np.where(halted_mask, w_prev, w_target)  # can't trade halted

        trade_notional = (w_target - w_prev) * aum
        positions_notional = w_prev * aum
        sigma_daily = rets.iloc[t - config.ADV_WINDOW:t].std().values
        adv_notional = adv.iloc[t].values
        costs = ms.total_costs(trade_notional, positions_notional, adv_notional,
                               sigma_daily, borrow_rate, taker=True, days=1)
        cost_ret = costs.total / aum
        net_ret = gross_ret - cost_ret

        if np.any(np.abs(w_target - w_prev) > 1e-9):
            n_trades += 1

        rows.append({
            "date": ts, "z": z, "direction": direction, "rank": rank,
            "half_life": half_life, "breakdown": breakdown,
            "n_halted": int(halted_mask.sum()),
            "gross_ret": gross_ret, "net_ret": net_ret,
            "fees": costs.fees, "impact": costs.impact, "borrow": costs.borrow,
            "gross_exposure": float(np.abs(w_prev).sum()),
        })
        w_prev = w_target

    timeline = pd.DataFrame(rows).set_index("date")
    if verbose:
        print(f"AUM ${aum:,.0f}: {n_trades} rebalances, "
              f"net ann. ret {timeline['net_ret'].mean()*252:.2%}")
    return BacktestResult(timeline=timeline, trades=n_trades, aum=aum,
                          metadata={"assets": assets})
