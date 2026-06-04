"""
Dynamic risk & hedging engine (convex optimisation).

The naive way to hold the cointegrating portfolio is to take positions exactly
proportional to β. But in the real world an asset can suddenly **halt** (no
trading) or become **hard-to-borrow** (shorting it is punitively expensive). We
must then reconstruct the *economic exposure* of the spread using the remaining
N-1 tradable assets, while staying market-neutral and respecting position caps.

We solve a convex (QP) tracking problem with CVXPY:

    minimise   (w - β)' Σ (w - β)        # replicate β's risk exposure
             + γ · Σ borrow_i · (-w_i)_+ # penalise expensive shorts
    s.t.       m' w = m' β                # preserve net market exposure
               w_i = 0  ∀ i ∈ halted      # cannot trade halted names
               |w_i| ≤ w_max              # concentration limit
               ||w||_1 ≤ L                # gross-leverage budget

Two points worth stressing:

* A Johansen cointegrating vector β is a *stationary* combination, not a
  dollar-neutral one, so we do **not** force ``1'w = 0``. Instead we preserve
  the *net market exposure* ``m'β`` of the original spread (with ``m`` the
  per-name market betas, defaulting to 1 = dollar exposure). When nothing is
  halted the optimum is simply ``w = β`` (zero tracking error); when a name
  drops out the remaining weights adjust so the book's market neutrality is
  unchanged — exactly the requirement in the brief.
* Minimising tracking error in *return-covariance* space (not raw weight space)
  means the optimiser substitutes a halted/expensive name with the basket of
  its most-correlated tradable peers — the correct economic hedge.

A SciPy SLSQP fallback is provided if CVXPY is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception:  # pragma: no cover
    _HAS_CVXPY = False

from scipy.optimize import minimize

import config


@dataclass
class HedgeSolution:
    weights: np.ndarray         # optimal per-name dollar weights
    tracking_error: float       # sqrt((w-β)'Σ(w-β))
    gross: float                # Σ|w|
    status: str
    solver: str


def optimize_hedge(
    beta: np.ndarray,
    cov: np.ndarray,
    halted: np.ndarray | None = None,
    borrow_rate: np.ndarray | None = None,
    market_beta: np.ndarray | None = None,
    max_weight: float = config.MAX_WEIGHT,
    max_gross: float = 3.0,
    borrow_penalty: float = 5.0,
    target_gross: float = config.TARGET_GROSS,
) -> HedgeSolution:
    """Re-optimise hedge weights for the tradable sub-universe.

    Parameters
    ----------
    beta        : target cointegrating exposure (length k), gross-normalised.
    cov         : k x k return covariance matrix (risk metric for tracking).
    halted      : boolean mask, True where the asset cannot be traded.
    borrow_rate : per-name annualised short-borrow cost (length k).
    market_beta : per-name market betas ``m`` to preserve net exposure ``m'β``.
                  Defaults to ones (preserve net dollar exposure).
    """
    k = len(beta)
    beta = np.asarray(beta, float).ravel()
    cov = np.asarray(cov, float)
    halted = np.zeros(k, bool) if halted is None else np.asarray(halted, bool)
    borrow_rate = (np.full(k, config.DEFAULT_BORROW_RATE)
                   if borrow_rate is None else np.asarray(borrow_rate, float))
    market_beta = (np.ones(k) if market_beta is None
                   else np.asarray(market_beta, float).ravel())

    # Only the borrow cost *in excess* of cheap general-collateral matters:
    # normal shorts are ~free, so the base case reduces to w = β exactly, while
    # hard-to-borrow names (on special) are penalised away from being shorted.
    excess_borrow = np.maximum(borrow_rate - config.DEFAULT_BORROW_RATE, 0.0)

    # Normalise target to the desired gross so tracking error is comparable.
    g = np.abs(beta).sum()
    target = beta / g * target_gross if g > 0 else beta
    net_target = float(market_beta @ target)   # net market exposure to preserve

    if _HAS_CVXPY:
        return _solve_cvxpy(target, cov, halted, excess_borrow, market_beta,
                            net_target, max_weight, max_gross, borrow_penalty)
    return _solve_scipy(target, cov, halted, excess_borrow, market_beta,
                        net_target, max_weight, max_gross, borrow_penalty)


def _solve_cvxpy(target, cov, halted, excess_borrow, market_beta, net_target,
                 max_weight, max_gross, borrow_penalty) -> HedgeSolution:
    k = len(target)
    w = cp.Variable(k)
    diff = w - target
    cov_psd = cp.psd_wrap(cov + 1e-10 * np.eye(k))      # PSD-safe
    tracking = cp.quad_form(diff, cov_psd)
    short_cost = borrow_penalty * (excess_borrow @ cp.pos(-w))
    constraints = [
        market_beta @ w == net_target,        # preserve net market exposure
        cp.norm1(w) <= max_gross,             # leverage budget
        w <= max_weight,
        w >= -max_weight,
    ]
    if halted.any():
        constraints.append(w[halted] == 0)    # cannot trade halted names

    prob = cp.Problem(cp.Minimize(tracking + short_cost), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    wv = np.zeros(k) if w.value is None else np.asarray(w.value).ravel()
    wv[np.abs(wv) < 1e-6] = 0.0
    te = float(np.sqrt(max((wv - target) @ cov @ (wv - target), 0.0)))
    return HedgeSolution(wv, te, float(np.abs(wv).sum()),
                         prob.status, "cvxpy/OSQP")


def _solve_scipy(target, cov, halted, excess_borrow, market_beta, net_target,
                 max_weight, max_gross, borrow_penalty) -> HedgeSolution:
    k = len(target)

    def obj(w):
        d = w - target
        return d @ cov @ d + borrow_penalty * (excess_borrow @ np.maximum(-w, 0))

    cons = [{"type": "eq", "fun": lambda w: market_beta @ w - net_target},
            {"type": "ineq", "fun": lambda w: max_gross - np.abs(w).sum()}]
    bounds = []
    for i in range(k):
        bounds.append((0.0, 0.0) if halted[i] else (-max_weight, max_weight))

    res = minimize(obj, target * (~halted), method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-10})
    wv = res.x
    wv[np.abs(wv) < 1e-6] = 0.0
    te = float(np.sqrt(max((wv - target) @ cov @ (wv - target), 0.0)))
    return HedgeSolution(wv, te, float(np.abs(wv).sum()),
                         "optimal" if res.success else "failed", "scipy/SLSQP")
