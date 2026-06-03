from __future__ import annotations

import asyncio
from collections import deque

import numpy as np
import pandas as pd

import config
from .broker import AsyncSubscriber, AsyncPublisher
from Econometrics import select_relation, build_spread


class MathEngine:
    def __init__(self, assets: list[str], train_window: int = config.TRAIN_WINDOW,
                 zwindow: int = config.ZSCORE_WINDOW):
        self.assets = assets
        self.train_window = train_window
        self.zwindow = zwindow
        # Latest price per asset (the "order-book state") + rolling history.
        self.latest: dict[str, float] = {a: np.nan for a in assets}
        self.halted: dict[str, bool] = {a: False for a in assets}
        self.history: deque[list[float]] = deque(maxlen=train_window)
        self.ts_history: deque[str] = deque(maxlen=train_window)
        self.beta: np.ndarray | None = None
        self.bars_seen = 0
        self.signals: list[dict] = []

    def _full_cross_section(self) -> bool:
        return all(not np.isnan(self.latest[a]) for a in self.assets)

    def _on_bar(self) -> dict | None:
        """Called once per complete cross-section snapshot; returns a signal."""
        self.bars_seen += 1
        self.history.append([self.latest[a] for a in self.assets])
        self.ts_history.append(self._last_ts)
        if len(self.history) < self.train_window:
            return None

        log_win = pd.DataFrame(
            np.log(np.array(self.history)), columns=self.assets
        )
        # (Re)estimate the cointegrating vector on a schedule.
        if self.beta is None or self.bars_seen % config.REFIT_EVERY == 0:
            rel = select_relation(log_win, method=config.SELECT_BY)
            self.beta = rel.beta
            self._last_rank = rel.rank
            self._last_hl = rel.half_life

        spread = build_spread(log_win, self.beta)
        z = (spread.iloc[-1] - spread.iloc[-self.zwindow:].mean()) / (
            spread.iloc[-self.zwindow:].std() or np.nan)
        sig = {
            "ts": self._last_ts,
            "zscore": float(z),
            "rank": int(getattr(self, "_last_rank", 0)),
            "half_life": float(getattr(self, "_last_hl", np.inf)),
            "action": self._z_to_action(z),
            "n_halted": int(sum(self.halted.values())),
        }
        return sig

    @staticmethod
    def _z_to_action(z: float) -> str:
        if not np.isfinite(z):
            return "flat"
        if z > config.ENTRY_Z:
            return "short_spread"
        if z < -config.ENTRY_Z:
            return "long_spread"
        if abs(z) < config.EXIT_Z:
            return "close"
        return "hold"

    async def run(self, n_bars: int | None = None,
                  md_addr: str = config.MARKETDATA_ADDR,
                  sig_addr: str = config.SIGNAL_ADDR):
        sub = AsyncSubscriber(md_addr, topics=[config.MARKETDATA_TOPIC])
        sigpub = AsyncPublisher(sig_addr, bind=True)
        await asyncio.sleep(0.2)
        self._last_ts = ""
        try:
            while True:
                topic, msg = await sub.recv()
                if msg.get("asset") == "__end__":
                    break
                a = msg["asset"]
                if a not in self.latest:
                    continue
                self.halted[a] = bool(msg.get("halted", False))
                if msg["price"] is not None:
                    self.latest[a] = msg["price"]
                self._last_ts = msg["ts"]

                # Emit a signal once per fresh, complete cross-section.
                if self._full_cross_section() and self._ts_advanced(msg["ts"]):
                    sig = self._on_bar()
                    if sig is not None:
                        self.signals.append(sig)
                        await sigpub.publish(config.SIGNAL_TOPIC, sig)
                        if n_bars and self.bars_seen >= n_bars:
                            break
        finally:
            sub.close()
            sigpub.close()
        return self.signals

    def _ts_advanced(self, ts: str) -> bool:
        if ts != getattr(self, "_committed_ts", None):
            self._committed_ts = ts
            return True
        return False
