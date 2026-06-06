from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import config 

ACCENT = "#7CFC00"
DARK = "#1b1b1b"


def _box(ax, xy, w, h, text, fc="#2b2b2b", ec=ACCENT, tc="white", fs=9):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w/2, y+h/2, text, ha="center", va="center",
            color=tc, fontsize=fs, weight="bold", wrap=True)
    return (x+w/2,y)  


def _arrow(ax, p0, p1, color=ACCENT, label=None):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=1.4, shrinkA=2, shrinkB=2))
    if label:
        mx, my = (p0[0]+p1[0])/2, (p0[1]+p1[1]) / 2
        ax.text(mx, my + 0.05, label, ha="center", va="bottom",
                fontsize=7, color="#cfcfcf", style="italic")


def build(path: Path |None = None) -> Path:
    path = path or config.DOCS_DIR/ "architecture.png"
    fig, ax = plt.subplots(figsize=(13, 8.5))
    fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
    ax.set_xlim(0, 13); ax.set_ylim(0, 9); ax.axis("off")
    ax.text(6.5, 8.6, "VECM-ARB  ·  Distributed Cross-Asset Statistical Arbitrage Engine",
            ha="center", color=ACCENT, fontsize=14, weight="bold")

    src = []
    for i, s in enumerate(["Yahoo Finance\n(daily adj close)",
                           "Binance / Bybit\n(crypto HFT)",
                           "Alpaca / Polygon\n(minute ETFs)"]):
        src.append(_box(ax, (0.6+i*4.3,7.4),3.4,0.9, s,
                        fc="#23303a", ec="#5fa8d3", fs=8))

    feeds = []
    for i in range(3):
        feeds.append(_box(ax, (0.6+i * 4.3, 6.0), 3.4, 0.8,
                          f"Feed service {i+1}\n(async publisher)",
                          fc="#2b2b2b", ec="#9b9b9b", fs=8))
    for s, f in zip(src, feeds):
        _arrow(ax, s, (f[0], f[1] +0.8))

    bus = _box(ax, (2.5, 4.7), 8.0, 0.8,
               "ZeroMQ / Redis  PUB-SUB bus   (topic: md.<ASSET>, non-blocking, HWM-bounded)",
               fc="#1f2d1f", ec=ACCENT, fs=9)
    for f in feeds:
        _arrow(ax, (f[0], f[1]), (f[0] , 5.5))

    eng = _box(ax, (0.6, 3.2), 5.2, 0.95,
               "MATH ENGINE  (asyncio subscriber)\nJohansen rank · VECM β,α · spread z-score",
               fc="#2b2b2b", ec=ACCENT, fs=8)
    bd = _box(ax, (6.2, 3.2), 2.7, 0.95,
              "Breakdown\nprotocol\n(rolling Johansen)", fc="#3a2323", ec="#d36b5f", fs=8)
    risk = _box(ax, (9.3, 3.2), 3.1, 0.95,
                "Dynamic-risk\nhedge optimiser\n(CVXPY · halts/HTB)",
                fc="#23303a", ec="#5fa8d3", fs=8)
    _arrow(ax, (6.5, 4.7), (eng[0], 4.15))
    _arrow(ax, (eng[0] + 2.0, 4.15), (bd[0], 4.15), label="β, spread")
    _arrow(ax, (bd[0] + 1.0, 4.15), (risk[0] - 0.8, 4.15), label="rank ok?")

    sig = _box(ax, (2.5, 2.0), 8.0, 0.7,
               "Signal bus  (topic: sig)  ->  Portfolio allocator (target weights)",
               fc="#1f2d1f", ec=ACCENT, fs=9)
    _arrow(ax, (eng[0], 3.2), (4.5, 2.7))
    _arrow(ax, (risk[0], 3.2), (8.5, 2.7))

    # execution / backtester
    bt = _box(ax, (1.5, 0.5), 5.0, 0.9,
              "Microstructure backtester\nfees · borrow · √-impact (ADV)",
              fc="#2b2b2b", ec="#9b9b9b", fs=8)
    ts = _box(ax, (7.0, 0.5), 4.5, 0.9,
              "Performance tear sheet\nSharpe · maxDD · capacity",
              fc="#2b2b2b", ec="#9b9b9b", fs=8)
    _arrow(ax, (4.5, 2.0), (bt[0], 1.4))
    _arrow(ax, (6.5, 0.95), (7.0, 0.95))

    fig.tight_layout()
    fig.savefig(path, facecolor=DARK)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("architecture diagram ->", build())
