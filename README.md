# Statistical-Arbitrage-Trading-System

**Author:** Aryan Sharma

A research-grade **statistical arbitrage** system for high-dimensional cointegration trading using:
- **Johansen cointegration** (rank/trace test)
- **VECM** fitting for equilibrium dynamics
- A tradable **spread** built from the selected cointegrating vector
- **Rolling validation / breakdown handling**
- **Transaction cost + square-root market impact** microstructure model
- Optional **live replay** using **ZeroMQ** pub/sub

---

## Quick start

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Download dataset (Yahoo Finance)
This repo expects one CSV per asset in `Dataset/` with a schema that the loader reads:
`Date, Open, High, Low, Close, Volume, Dividends, Stock Splits`

Run:
```bash
python Dataset/download_dataset.py --period 5y
```

### 3) Run a backtest
Walk-forward backtest (default):
```bash
python scripts/run_backtest.py
```

Stress scenario:
```bash
python scripts/run_backtest.py --stress
```

Outputs are written to `Results/` and a fitted model snapshot to `Models/`.

### 4) Run a live replay (ZeroMQ pipeline)
```bash
python scripts/run_live.py --bars 60 --speed 200
```

Stress “NVIDIA halt” feed flag (last 5 bars):
```bash
python scripts/run_live.py --bars 60 --speed 200 --halt-nvidia
```

---

## What this system does (high level)

1. **Load aligned panels** for the asset universe:
   - `prices` (Close)
   - `log_prices` (natural log)
   - `volume` (share volume)
   - `adv` (rolling average dollar volume)

2. **Estimate the cointegration structure**:
   - Johansen test on the full sample (for reporting)
   - Walk-forward refits during backtest using `TRAIN_WINDOW` and `REFIT_EVERY`

3. **Select a tradable relationship**:
   - By default: `SELECT_BY="shortest_half_life"`
   - The system can also trade the dominant relation.

4. **Generate a spread z-score**:
   - Rolling window `ZSCORE_WINDOW`
   - Entry/exit/stop thresholds from `config.py`

5. **Manage regimes / breakdowns**:
   - Periodic rank re-validation (`BREAKDOWN_CHECK_EVERY`)
   - When validated rank collapses repeatedly, the system liquidates & pauses.

6. **Allocate portfolio weights and model frictions**:
   - Target weights derived from cointegrating vector (scaled to `TARGET_GROSS`)
   - Optional convex hedge optimization via `cvxpy`
   - Costs include:
     - maker/taker fees + commission
     - borrow cost (incl. “hard-to-borrow” names)
     - square-root market impact using ADV and volatility estimates

7. **Measure performance and capacity**:
   - Tear sheet + timeline
   - Capacity analysis across an AUM grid using the impact model

---

## Repo structure

- `config.py`  
  Central configuration (universe, econometrics settings, thresholds, cost/impact, ZeroMQ endpoints)

- `Dataset/`
  - `download_dataset.py` — downloads Yahoo Finance data into `Dataset/{Asset}.csv`
  - `loader.py` — loads and aligns panels (`prices`, `log_prices`, `volume`, `adv`)
  - `assets.py` — ticker mapping (used by `download_dataset.py`)

- `Econometrics/`
  - `johansen.py` — Johansen cointegration test
  - `vecm_model.py` / `spread.py` — VECM fitting and spread construction
  - `selection.py` — relation selection logic (e.g., shortest half-life)

- `Backtester/`
  - `engine.py` — main walk-forward backtest loop (signal → weights → costs → net returns)
  - `microstructure.py` — market impact + fee/borrow cost calculations
  - `tearsheet.py` — metrics + tear sheet formatting
  - `__init__.py` — exports backtest functions

- `Risk/`
  - `hedge_optimizer.py` — convex hedge optimizer (target gross + constraints)
  - `breakdown.py` — rolling cointegration breakdown monitor

- `Messaging/`
  - `broker.py`, `feed_publisher.py`, `math_engine.py` — ZeroMQ pub/sub live replay pipeline

- `scripts/`
  - `run_backtest.py` — end-to-end research run, writes deliverables
  - `run_live.py` — async live replay using ZeroMQ

- `Results/`  
  Deliverables (CSV timelines, capacity analysis, PNG plots, `tearsheet.txt`)

- `Models/`  
  Saved snapshot model: `vecm_arb_model.json`

---

## Configuration (`config.py`) — key knobs

Notable parameters you may want to tune:

### Econometrics
- `JOHANSEN_DET_ORDER` — deterministic term in cointegration relation
- `K_AR_DIFF` — VECM lag differences
- `SIGNIF_IDX` — Johansen critical value column selection
- `TRADE_VECTOR` — which cointegrating vector index to trade
- `USE_LOG_PRICES` — build spreads from log prices (additive spreads)

### Signal generation
- `TRAIN_WINDOW` — estimation window length
- `REFIT_EVERY` — how often beta/half-life are re-estimated
- `ZSCORE_WINDOW` — rolling z-score mean/std window
- `ENTRY_Z`, `EXIT_Z`, `STOP_Z` — trading thresholds
- `SELECT_BY` — relation selection rule (`shortest_half_life` or `dominant`)
- `HALF_LIFE_FILTER` / `MIN_HALF_LIFE` / `MAX_HALF_LIFE` — filter out noisy or slow relations

### Portfolio + risk constraints
- `TARGET_GROSS` — gross exposure fraction of equity
- `MAX_WEIGHT` — per-asset concentration cap

### Microstructure / costs
- `MAKER_FEE_BPS`, `TAKER_FEE_BPS`, `COMMISSION_BPS`
- `DEFAULT_BORROW_RATE`, `HARD_TO_BORROW_RATE`, `HARD_TO_BORROW`
- Square-root impact:
  - `IMPACT_COEF`, `ADV_WINDOW`, `TRADING_DAYS`

### Breakdown protocol
- `BREAKDOWN_CHECK_EVERY`, `BREAKDOWN_WINDOW`
- `MIN_RANK`, `BREAKDOWN_CONFIRM`

### ZeroMQ live replay
- `MARKETDATA_ADDR`, `SIGNAL_ADDR`
- `MARKETDATA_TOPIC`, `SIGNAL_TOPIC`
- `LIVE_REPLAY_SPEED` (bars/sec)

---

## Backtest outputs

After running `scripts/run_backtest.py`, the following are written:

**Files**
- `Results/tearsheet.txt`  
- `Results/backtest_timeline.csv`  
- `Results/capacity_analysis.csv`

**Model snapshot**
- `Models/vecm_arb_model.json`

**Plots (PNG)**
- `Results/capacity_curve.png`
- `Results/equity_drawdown.png`
- `Results/spread_zscore.png`
- `Results/cost_decomposition.png`
- `Results/cointegration_rank.png`
(plus additional visuals produced by `Visualization/plots.py`)

---

## Notes / assumptions

- Spread construction and econometrics use **log prices** by default.
- The backtest is **walk-forward**:
  - it re-estimates the cointegration relation periodically (`REFIT_EVERY`)
  - and uses a rolling z-score window (`ZSCORE_WINDOW`)
- Trading is **pause/liquidate** driven by breakdown monitoring (rank collapse + half-life blow-out logic inside the risk module).
- Live mode is a **replay pipeline**:
  - it streams historical bars at an accelerated pace over ZeroMQ
  - and emits signals via the math engine

---

## License
See `LICENSE`.
python scripts/run_live.py --bars 60 --speed 200 --halt-nvidia
python scripts/run_backtest.py
python Dataset/download_dataset.py --period 5y
pip install -r requirements.txt
