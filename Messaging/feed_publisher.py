from __future__ import annotations

import asyncio

import pandas as pd

import config
from .broker import AsyncPublisher


async def run_feed(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    address: str = config.MARKETDATA_ADDR,
    speed: float = config.LIVE_REPLAY_SPEED,
    halts: dict[str, set] | None = None,
    seed: int = 0,
):
    pub = AsyncPublisher(address, bind=True)
    halts = halts or {}
    # Tiny settle so subscribers finish connecting before the first publish.
    await asyncio.sleep(0.3)

    dt = 1.0 / max(speed, 1e-6)
    # Deterministic per-asset phase offset -> staggered, non-synchronous arrival.
    offsets = {a: ((seed + i * 7) % 11) / 11.0 * dt
               for i, a in enumerate(prices.columns)}

    try:
        for ts, row in prices.iterrows():
            vol_row = volume.loc[ts]
            for asset in prices.columns:
                halted = ts in halts.get(asset, set())
                msg = {
                    "asset": asset,
                    "ts": str(ts),
                    "price": None if halted else float(row[asset]),
                    "volume": float(vol_row[asset]) if pd.notna(vol_row[asset]) else 0.0,
                    "halted": halted,
                }
                await pub.publish(f"{config.MARKETDATA_TOPIC}.{asset}", msg)
                await asyncio.sleep(offsets[asset])  # non-synchronous jitter
            await asyncio.sleep(dt)
        # Sentinel so the engine knows the replay is finished.
        await pub.publish(f"{config.MARKETDATA_TOPIC}.__end__",
                          {"asset": "__end__", "ts": str(prices.index[-1])})
    finally:
        await asyncio.sleep(0.1)
        pub.close()
