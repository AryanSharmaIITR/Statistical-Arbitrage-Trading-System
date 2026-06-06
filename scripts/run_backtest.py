from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd


_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config
from Dataset.loader import load_panels
from Econometrics import run_johansen, fit_vecm, select_relation
from Backtester import (run_backtest, compute_metrics, format_tearsheet,
                        capacity_analysis, capacity_point)
from Visualization import plots

REFERENCE_AUM = 1e7

def main(stress: bool = False):
    print("Loading data ...")
    panels = load_panels()
    prices, log_prices = panels["prices"], panels["log_prices"]
    assets = list(prices.columns)

    print("\n[1] Johansen cointegration test (full sample)")
    jr = run_johansen(log_prices, config.JOHANSEN_DET_ORDER,
                      config.K_AR_DIFF, config.SIGNIF_IDX)
    print(jr.summary())

    rank_for_vecm = max(jr.rank, 1)
    fit = fit_vecm(log_prices, rank_for_vecm, config.K_AR_DIFF,
                   config.JOHANSEN_DET_ORDER)
    print("\n[2] VECM fit")
    print(fit.summary())

    rel = select_relation(log_prices.iloc[-config.TRAIN_WINDOW:],
                          method=config.SELECT_BY)

    kwargs = {}
    if stress:
        idx = prices.index
        kwargs = {"halts": {"NVIDIA": set(idx[800:830])},
                  "hard_to_borrow": {"NVIDIA"}}
        print("\n[3] STRESS backtest (NVIDIA halt + hard-to-borrow)")
    else:
        print("\n[3] Walk-forward backtest")
    result = run_backtest(panels, aum=REFERENCE_AUM, verbose=True, **kwargs)
    metrics = compute_metrics(result)
    sheet = format_tearsheet(metrics, REFERENCE_AUM, assets)
    print("\n" + sheet)

    print("\n[4] Capacity analysis across AUM grid ...")
    cap = capacity_analysis(panels, **kwargs)
    cap_aum = capacity_point(cap)
    print(cap.round(3).to_string())
    print(f"\nStrategy capacity (net Sharpe >= {config.SHARPE_FLOOR}): "
          f"{'$%.0f' % cap_aum if np.isfinite(cap_aum) else 'below grid'}")

    print("\n[5] Writing deliverables ...")
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "tearsheet.txt").write_text(
        sheet + "\n\n" + jr.summary() + "\n\n" + fit.summary() + "\n")
    result.timeline.to_csv(config.RESULTS_DIR / "backtest_timeline.csv")
    cap.to_csv(config.RESULTS_DIR / "capacity_analysis.csv")

    model = {
        "assets": assets,
        "johansen_rank_full_sample": int(jr.rank),
        "trace_stat": [float(x) for x in jr.trace_stat],
        "trace_crit": [float(x) for x in jr.trace_crit],
        "latest_beta": dict(zip(assets, [float(x) for x in rel.beta])),
        "latest_rank": int(rel.rank),
        "latest_half_life": float(rel.half_life),
        "metrics_at_reference_aum": metrics.as_dict(),
        "reference_aum": REFERENCE_AUM,
        "capacity_aum": None if not np.isfinite(cap_aum) else cap_aum,
    }
    (config.MODELS_DIR / "vecm_arb_model.json").write_text(json.dumps(model, indent=2))

    plots.plot_prices(prices)
    plots.plot_equity_drawdown(result.timeline)
    plots.plot_spread_zscore(result.timeline)
    plots.plot_cost_decomposition(result.timeline)
    plots.plot_capacity(cap, cap_aum)
    plots.plot_cointegration(jr)
    print(f"  -> tear sheet, CSVs, model JSON and 6 plots written to "
          f"{config.RESULTS_DIR.relative_to(_ROOT)}/ and Models/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stress", action="store_true",
                    help="run the halt + hard-to-borrow stress scenario")
    args = ap.parse_args()
    main(stress=args.stress)
