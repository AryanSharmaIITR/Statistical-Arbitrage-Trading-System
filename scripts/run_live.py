from __future__ import annotations
import argparse
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config
from Dataset.loader import load_panels
from Messaging import run_feed, MathEngine

async def _run(live_bars: int, speed: float, halt_nvidia: bool):
    panels = load_panels()
    n = config.TRAIN_WINDOW + live_bars
    prices = panels["prices"].iloc[-n:]
    volume = panels["volume"].iloc[-n:]
    assets = list(prices.columns)

    halts = None
    if halt_nvidia:
        halts = {"NVIDIA": set(prices.index[-5:])}

    engine = MathEngine(assets)
    feed_task = asyncio.create_task(
        run_feed(prices, volume, speed=speed, halts=halts))

    print(f"Streaming {len(prices)} bars over ZeroMQ "
          f"({config.MARKETDATA_ADDR}) at {speed} bars/s ...")
    print("warm-up: first %d bars build the estimation window\n" % config.TRAIN_WINDOW)
    print(f"{'date':<12} {'z':>7} {'rank':>4} {'half_life':>9} "
          f"{'halted':>6}  action")
    print("-" * 60)

    async def consume():
        signals = await engine.run()
        return signals

    signals = await consume()
    await feed_task

    for s in signals[-live_bars:]:
        print(f"{s['ts'][:10]:<12} {s['zscore']:>7.2f} {s['rank']:>4d} "
              f"{s['half_life']:>9.1f} {s['n_halted']:>6d}  {s['action']}")
    print("-" * 60)
    acted = [s for s in signals if s["action"] in
             ("long_spread", "short_spread", "close")]
    print(f"{len(signals)} signals emitted, {len(acted)} actionable "
          f"(entries/exits). Pipeline ran without blocking.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=60, help="live bars to replay")
    ap.add_argument("--speed", type=float, default=config.LIVE_REPLAY_SPEED,
                    help="replay speed (bars/sec)")
    ap.add_argument("--halt-nvidia", action="store_true",
                    help="halt NVIDIA for the last 5 bars (feed flags it)")
    args = ap.parse_args()
    asyncio.run(_run(args.bars, args.speed, args.halt_nvidia))

if __name__ == "__main__":
    main()
