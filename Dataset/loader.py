from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make ``import config`` work regardless of the cwd the script is launched from.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402


def _read_one(asset: str) -> pd.DataFrame:
    path = config.DATASET_DIR / f"{asset}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python Dataset/download_dataset.py` first."
        )
    df = pd.read_csv(path)
    # yfinance writes timezone-aware ISO timestamps in the Date column.
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None).dt.normalize()
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def load_panels(assets: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load and align all panels.

    Returns a dict with keys:
      ``prices``      close-price panel        (DataFrame, assets as columns)
      ``log_prices``  natural-log of prices    (DataFrame)
      ``volume``      share volume             (DataFrame)
      ``adv``         rolling average daily $ volume (DataFrame)
    All panels share one inner-joined DatetimeIndex.
    """
    assets = assets or config.ASSETS
    closes, vols = {}, {}
    for a in assets:
        df = _read_one(a)
        closes[a] = df["Close"]
        vols[a] = df["Volume"]

    prices = pd.DataFrame(closes).dropna(how="any")
    volume = pd.DataFrame(vols).reindex(prices.index)
    # $-volume, then rolling mean -> ADV used by the square-root impact model.
    dollar_vol = prices * volume
    adv = dollar_vol.rolling(config.ADV_WINDOW, min_periods=5).mean().bfill()

    panels = {
        "prices": prices,
        "log_prices": np.log(prices),
        "volume": volume,
        "adv": adv,
    }
    return panels


def returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns panel."""
    return prices.pct_change().dropna(how="any")


if __name__ == "__main__":
    p = load_panels()
    px = p["prices"]
    print(f"Loaded {px.shape[1]} assets x {px.shape[0]} bars")
    print(f"Date range: {px.index.min().date()} -> {px.index.max().date()}")
    print("\nLast close prices:")
    print(px.tail(1).T.round(2))
    print("\nMean ADV ($mm):")
    print((p["adv"].mean() / 1e6).round(1))
