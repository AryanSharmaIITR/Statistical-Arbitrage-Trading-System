"""
Refresh the per-asset daily dataset from Yahoo Finance.

    python Dataset/download_dataset.py [--period 5y]

Writes one CSV per asset next to this file, matching the schema the loader
expects (Date, Open, High, Low, Close, Volume, Dividends, Stock Splits).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yfinance as yf

from assets import assets_tickers

HERE = Path(__file__).resolve().parent


def get_dataset(period: str = "5y") -> None:
    for asset, ticker in assets_tickers.items():
        data = yf.Ticker(ticker).history(period=period)
        out = HERE / f"{asset}.csv"
        data.to_csv(out)
        print(f"  {asset:<12} {ticker:<6} -> {out.name}  ({len(data)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="5y", help="yfinance history period")
    args = ap.parse_args()
    print(f"Downloading {len(assets_tickers)} assets ({args.period}) ...")
    get_dataset(args.period)
    print("done.")
