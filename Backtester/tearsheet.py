from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import config
from .engine import run_backtest, BacktestResult

@dataclass
class Metrics:
    total_return: float
    cagr: float
    ann_return: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    n_rebalances: int
    avg_cost_drag_bps: float
    gross_sharpe: float

    def as_dict(self) -> dict:
        return asdict(self)


def _max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def compute_metrics(result: BacktestResult) -> Metrics:
    tl = result.timeline
    net = tl["net_ret"]
    gross = tl["gross_ret"]
    eq = (1 + net).cumprod()
    ann = TRADING = config.TRADING_DAYS

    ann_ret = net.mean() * ann
    ann_vol = net.std() * np.sqrt(ann)
    downside = net[net < 0].std() * np.sqrt(ann)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    sortino = ann_ret / downside if downside > 0 else 0.0
    mdd = _max_drawdown(eq)
    n_years = len(net) / ann
    cagr = eq.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else 0.0
    gross_ann = gross.mean() * ann
    gross_vol = gross.std() * np.sqrt(ann)
    cost_drag = (tl["fees"] + tl["impact"] + tl["borrow"]).sum() / result.aum \
        / n_years * 1e4

    return Metrics(
        total_return=float(eq.iloc[-1] - 1),
        cagr=float(cagr),
        ann_return=float(ann_ret),
        ann_vol=float(ann_vol),
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown=mdd,
        calmar=float(ann_ret / abs(mdd)) if mdd < 0 else 0.0,
        hit_rate=float((net > 0).mean()),
        n_rebalances=result.trades,
        avg_cost_drag_bps=float(cost_drag),
        gross_sharpe=float(gross_ann / gross_vol) if gross_vol > 0 else 0.0,
    )


def format_tearsheet(metrics: Metrics, aum: float, assets: list[str]) -> str:
    m = metrics
    lines = [
        "=" * 60,
        "  VECM-ARB  —  PERFORMANCE TEAR SHEET",
        "=" * 60,
        f"  Universe (k={len(assets)}) : {', '.join(assets)}",
        f"  Notional AUM         : ${aum:,.0f}",
        "-" * 60,
        f"  Total return         : {m.total_return:>10.2%}",
        f"  CAGR                 : {m.cagr:>10.2%}",
        f"  Annualised return    : {m.ann_return:>10.2%}",
        f"  Annualised vol       : {m.ann_vol:>10.2%}",
        f"  Sharpe (net)         : {m.sharpe:>10.2f}",
        f"  Sharpe (gross)       : {m.gross_sharpe:>10.2f}",
        f"  Sortino              : {m.sortino:>10.2f}",
        f"  Max drawdown         : {m.max_drawdown:>10.2%}",
        f"  Calmar               : {m.calmar:>10.2f}",
        f"  Hit rate (daily)     : {m.hit_rate:>10.2%}",
        f"  Rebalances           : {m.n_rebalances:>10d}",
        f"  Cost drag (ann.)     : {m.avg_cost_drag_bps:>9.1f}bps",
        "=" * 60,
    ]
    return "\n".join(lines)


def capacity_analysis(panels: dict, aum_grid: list[float] | None = None,
                      sharpe_floor: float = config.SHARPE_FLOOR, **bt_kwargs) -> pd.DataFrame:
    grid = aum_grid or config.CAPACITY_AUM_GRID
    out = []
    for aum in grid:
        res = run_backtest(panels, aum=aum, **bt_kwargs)
        met = compute_metrics(res)
        tl = res.timeline
        total_cost = (tl["fees"] + tl["impact"] + tl["borrow"]).sum()
        impact_share = tl["impact"].sum() / total_cost if total_cost > 0 else 0.0
        out.append({
            "aum": aum,
            "net_sharpe": met.sharpe,
            "gross_sharpe": met.gross_sharpe,
            "net_ann_return": met.ann_return,
            "cost_drag_bps": met.avg_cost_drag_bps,
            "impact_share": impact_share,
            "viable": met.sharpe >= sharpe_floor,
        })
    df = pd.DataFrame(out).set_index("aum")
    return df


def capacity_point(cap_df: pd.DataFrame, sharpe_floor: float = config.SHARPE_FLOOR) -> float:
    viable = cap_df[cap_df["net_sharpe"] >= sharpe_floor]
    return float(viable.index.max()) if len(viable) else float("nan")
