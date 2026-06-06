from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import config
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 9})
_ACCENT = "#7CFC00"

def plot_prices(prices: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or config.VIZ_DIR / "Closing_Price.png"
    fig, ax = plt.subplots(figsize=(11, 5))
    (prices / prices.iloc[0] * 100).plot(ax=ax, lw=1)
    ax.set_title("Normalised closing prices (start = 100)")
    ax.set_ylabel("Index"); ax.legend(ncol=4, fontsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path

def plot_equity_drawdown(timeline: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or config.RESULTS_DIR / "equity_drawdown.png"
    eq = (1 + timeline["net_ret"]).cumprod()
    eqg = (1 + timeline["gross_ret"]).cumprod()
    dd = eq / eq.cummax() - 1
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
    a1.plot(eqg.index, eqg.values, color="grey", lw=1, label="gross")
    a1.plot(eq.index, eq.values, color=_ACCENT, lw=1.6, label="net of costs")
    a1.set_title("VECM-ARB equity curve"); a1.set_ylabel("Growth of $1")
    a1.legend()
    a2.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.5)
    a2.set_ylabel("Drawdown"); a2.set_xlabel("Date")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path

def plot_spread_zscore(timeline: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or config.RESULTS_DIR / "spread_zscore.png"
    z = timeline["z"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(z.index, z.values, color="steelblue", lw=0.9)
    for lvl, st in [(config.ENTRY_Z, "--"), (-config.ENTRY_Z, "--"),
                    (config.EXIT_Z, ":"), (-config.EXIT_Z, ":"),
                    (config.STOP_Z, "-"), (-config.STOP_Z, "-")]:
        ax.axhline(lvl, color="grey", ls=st, lw=0.8)
    pos = timeline["direction"] != 0
    ax.fill_between(z.index, z.min(), z.max(), where=pos, color=_ACCENT,
                    alpha=0.12, label="in position")
    ax.set_title("Spread z-score with entry/exit/stop bands")
    ax.set_ylabel("z-score"); ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path

def plot_capacity(cap_df: pd.DataFrame, capacity_aum: float,
                  path: Path | None = None) -> Path:
    path = path or config.RESULTS_DIR / "capacity_curve.png"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.semilogx(cap_df.index, cap_df["net_sharpe"], "o-", color=_ACCENT,
                label="net Sharpe")
    a1.axhline(cap_df["gross_sharpe"].iloc[0], color="grey", ls="--",
               label="gross Sharpe")
    a1.axhline(config.SHARPE_FLOOR, color="crimson", ls=":", label="viability floor")
    if np.isfinite(capacity_aum):
        a1.axvline(capacity_aum, color="black", ls="-", lw=0.8)
    a1.set_xlabel("AUM ($)"); a1.set_ylabel("Sharpe")
    a1.set_title("Capacity: Sharpe vs AUM"); a1.legend(fontsize=7)
    a2.semilogx(cap_df.index, cap_df["cost_drag_bps"], "s-", color="darkorange",
                label="total cost drag")
    a2.semilogx(cap_df.index, cap_df["impact_share"] * 100, "^-",
                color="purple", label="impact % of cost")
    a2.set_xlabel("AUM ($)"); a2.set_ylabel("bps  /  %")
    a2.set_title("Cost decomposition vs AUM"); a2.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path

def plot_cost_decomposition(timeline: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or config.RESULTS_DIR / "cost_decomposition.png"
    cum = timeline[["fees", "impact", "borrow"]].cumsum()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.stackplot(cum.index, cum["fees"], cum["impact"], cum["borrow"],
                 labels=["fees", "market impact", "borrow"],
                 colors=["#4C72B0", "#C44E52", "#55A868"], alpha=0.8)
    ax.set_title("Cumulative transaction-cost decomposition ($)")
    ax.set_ylabel("Cumulative cost ($)"); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path

def plot_cointegration(jr, path: Path | None = None) -> Path:
    path = path or config.RESULTS_DIR / "cointegration_rank.png"
    n = len(jr.trace_stat)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, jr.trace_stat, 0.4, label="trace statistic", color=_ACCENT)
    ax.bar(x + 0.2, jr.trace_crit, 0.4, label=f"crit ({jr.signif_label})",
           color="grey")
    ax.set_xticks(x); ax.set_xticklabels([f"r≤{i}" for i in range(n)])
    ax.set_title(f"Johansen trace test — selected rank r = {jr.rank}")
    ax.set_ylabel("statistic"); ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path