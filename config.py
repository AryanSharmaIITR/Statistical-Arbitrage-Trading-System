"""
Central configuration for the VECM-ARB distributed statistical-arbitrage engine.

Every tunable parameter — econometric, trading, microstructure and messaging —
lives here so that the math engine, backtester and live services all read from a
single source of truth.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "Dataset"
RESULTS_DIR = ROOT / "Results"
MODELS_DIR = ROOT / "Models"
VIZ_DIR = ROOT / "Visualization"
DOCS_DIR = ROOT / "docs"

for _d in (RESULTS_DIR, MODELS_DIR, VIZ_DIR, DOCS_DIR):
    _d.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
# The k-dimensional basket. These large-cap US tech names share common
# macro / sector factors, which is exactly the setting where a high-dimensional
# cointegration relationship (an equilibrium "fair value" of the basket) tends
# to exist.
ASSETS = [
    "Apple",
    "Microsoft",
    "Google",
    "Amazon",
    "NVIDIA",
    "Meta",
    "Adobe",
    "Salesforce",
]

# --------------------------------------------------------------------------- #
# Econometrics (Johansen / VECM)
# --------------------------------------------------------------------------- #
# det_order: -1 no deterministic term, 0 constant in the cointegration relation,
#            1 linear trend. 0 is the standard choice for log price levels.
JOHANSEN_DET_ORDER = 0
# Number of lagged differences in the VECM (p-1). Selected empirically / by IC.
K_AR_DIFF = 1
# Significance column for Johansen critical values: 0=90%, 1=95%, 2=99%.
SIGNIF_IDX = 1
SIGNIF_LABEL = "95%"
# We trade the dominant (largest-eigenvalue) cointegrating vector as the spread.
TRADE_VECTOR = 0

# Use log prices for the econometrics (price ratios -> additive spreads).
USE_LOG_PRICES = True

# --------------------------------------------------------------------------- #
# Trading / signal generation
# --------------------------------------------------------------------------- #
TRAIN_WINDOW = 252          # bars used to (re)estimate the cointegration system
ZSCORE_WINDOW = 30          # rolling window for spread mean / std
REFIT_EVERY = 63            # re-estimate beta every N bars (adaptive equilibrium)

# We trade the cointegrating vector with the SHORTEST mean-reversion half-life
# among the validated relations — the most tradable one, not just the dominant.
SELECT_BY = "shortest_half_life"   # or "dominant"
MAX_HALF_LIFE = 40.0        # skip relations slower than this (bars)
MIN_HALF_LIFE = 2.0         # skip relations faster than this (noise / overfit)
# Only open a trade when the spread is genuinely mean-reverting, i.e. its
# half-life sits inside [MIN_HALF_LIFE, MAX_HALF_LIFE].
HALF_LIFE_FILTER = True

ENTRY_Z = 2.0               # |z| above which we open a mean-reversion position
EXIT_Z = 0.5                # |z| below which we close
STOP_Z = 4.0                # |z| above which we hard-stop (regime / blow-out)

TARGET_GROSS = 1.0          # gross exposure as a fraction of equity (per leg book)
MAX_WEIGHT = 0.40           # cap on any single-name weight (concentration limit)

# --------------------------------------------------------------------------- #
# Microstructure cost model
# --------------------------------------------------------------------------- #
MAKER_FEE_BPS = 0.0         # passive rebate/fee (bps of notional)
TAKER_FEE_BPS = 1.0         # aggressive fee (bps of notional)
COMMISSION_BPS = 0.5        # broker commission (bps of notional)

# Annualised short-borrow / hard-to-borrow financing cost (fraction per year).
DEFAULT_BORROW_RATE = 0.005             # 50 bps general collateral
HARD_TO_BORROW_RATE = 0.15              # 15% for a name on special
# Names flagged hard-to-borrow in the stress scenarios.
HARD_TO_BORROW = {"NVIDIA"}

# Square-root market-impact model:  impact_frac = IMPACT_COEF * sigma * sqrt(Q/ADV)
IMPACT_COEF = 0.5
ADV_WINDOW = 21             # bars used to estimate average daily $ volume
TRADING_DAYS = 252

# --------------------------------------------------------------------------- #
# Cointegration-breakdown protocol
# --------------------------------------------------------------------------- #
BREAKDOWN_CHECK_EVERY = 21  # rolling Johansen re-validation cadence (bars)
MIN_RANK = 1                # the relationship is "validated" while rank >= this
BREAKDOWN_WINDOW = 252      # window for the rolling Johansen re-test
# Liquidate only after the rank stays collapsed AND the spread half-life blows
# out for this many consecutive checks — avoids whipsawing on a single reading.
BREAKDOWN_CONFIRM = 2

# --------------------------------------------------------------------------- #
# Messaging (ZeroMQ pub/sub)
# --------------------------------------------------------------------------- #
# ZeroMQ is brokerless, so the demo runs with zero external infrastructure.
# Swap these endpoints for a Redis pub/sub URL in production if preferred.
MARKETDATA_ADDR = "tcp://127.0.0.1:5556"   # publishers -> math engine
SIGNAL_ADDR = "tcp://127.0.0.1:5557"       # math engine -> allocator
MARKETDATA_TOPIC = "md"                    # topic prefix for order-book/bar msgs
SIGNAL_TOPIC = "sig"                       # topic prefix for trade signals
LIVE_REPLAY_SPEED = 200.0                  # bars/sec when replaying history live

# --------------------------------------------------------------------------- #
# Capacity analysis
# --------------------------------------------------------------------------- #
# AUM grid ($) over which we measure how square-root impact erodes net alpha.
CAPACITY_AUM_GRID = [
    1e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9, 5e9, 1e10,
]
# Net-Sharpe floor defining usable strategy capacity (≈ 70% of gross Sharpe).
SHARPE_FLOOR = 0.4

RANDOM_SEED = 42
