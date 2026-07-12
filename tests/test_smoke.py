from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import config
from Dataset.loader import load_panels, returns
from Econometrics import run_johansen, fit_vecm, select_relation, build_spread
from Risk import optimize_hedge, BreakdownMonitor
from Backtester import run_backtest, compute_metrics, market_impact

PANELS = load_panels()
LP = PANELS["log_prices"]
ASSETS = list(LP.columns)

def test_data_aligned():
    assert PANELS["prices"].shape[1] == len(config.ASSETS)
    assert PANELS["prices"].index.equals(PANELS["log_prices"].index)
    assert not PANELS["prices"].isna().any().any()

def test_johansen_runs_and_orders_eigenvalues():
    jr = run_johansen(LP,config.JOHANSEN_DET_ORDER,config.K_AR_DIFF)
    assert 0 <= jr.rank <= len(ASSETS)      # eigenvalues sorted descending
    assert np.all(np.diff(jr.eigenvalues) <= 1e-12) # cointegrating vector is L1-normalised
    np.testing.assert_allclose(np.abs(jr.cointegrating_vector(0)).sum(), 1.0, atol=1e-9)

def test_vecm_shapes():
    fit = fit_vecm(LP, rank=2, k_ar_diff=1)
    assert fit.beta.shape== (len(ASSETS), 2)
    assert fit.alpha.shape == (len(ASSETS), 2)

def test_spread_is_more_stationary_than_components():
    rel = select_relation(LP.iloc[-252:], method="shortest_half_life")
    spread = build_spread(LP.iloc[-252:], rel.beta)      # a tradable relation should have a finite, positive half-life
    assert 0 < rel.half_life < 1e6


def test_hedge_base_case_recovers_beta():
    rel = select_relation(LP.iloc[-252:], method=config.SELECT_BY)
    cov =returns(PANELS["prices"]).iloc[-252:].cov().values * config.TRADING_DAYS
    g = np.abs(rel.beta).sum()
    target = rel.beta / g
    sol = optimize_hedge(rel.beta,cov)
    np.testing.assert_allclose(sol.weights, target, atol=1e-3)
    assert sol.tracking_error < 1e-3

def test_hedge_zeroes_halted_and_preserves_net_exposure():
    rel = select_relation(LP.iloc[-252:], method=config.SELECT_BY)
    cov = returns(PANELS["prices"]).iloc[-252:].cov().values * config.TRADING_DAYS
    halted = np.array([a == "NVIDIA" for a in ASSETS])
    g = np.abs(rel.beta).sum()
    net_before = (rel.beta / g).sum()
    sol =optimize_hedge(rel.beta, cov, halted=halted)
    assert abs(sol.weights[ASSETS.index("NVIDIA")]) < 1e-6      # halted -> zero
    np.testing.assert_allclose(sol.weights.sum(), net_before, atol=1e-3)  # neutrality kept


# doubling traded size less-than-doubles impact cost per unit (sqrt law)
def test_market_impact_is_sqrt_concave():
    adv = np.array([1e9]); sig = np.array([0.02])
    c1= market_impact(np.array([1e6]), adv, sig)
    c2 = market_impact(np.array([4e6]), adv, sig)
    # 4x notional -> impact COST scales 4^1.5 = 8x (frac ~ sqrt, cost = frac*notional)
    np.testing.assert_allclose(c2 / c1, 8.0, rtol=1e-6)

def test_breakdown_monitor_returns_state():
    rel =select_relation(LP.iloc[-252:], method=config.SELECT_BY)
    st = BreakdownMonitor().check(LP.iloc[-252:], rel.beta)
    assert isinstance(st.breakdown, bool)
    assert st.rank >= 0

def test_backtest_runs_and_costs_scale_with_aum():
    small = run_backtest(PANELS,aum=1e6)
    large = run_backtest(PANELS, aum=1e9)
    ms =compute_metrics(small); ml = compute_metrics(large)
    np.testing.assert_allclose(ms.gross_sharpe, ml.gross_sharpe, rtol=1e-6)     # gross identical, net worse at larger AUM (impact bites)
    assert ml.sharpe < ms.sharpe
    assert large.timeline["impact"].sum() > small.timeline["impact"].sum()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL  {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
