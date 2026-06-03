from __future__ import annotations

import json
from typing import Any

import zmq
import zmq.asyncio


def _encode(topic: str, payload: dict[str, Any]) -> bytes:
    return topic.encode() + b" " + json.dumps(payload).encode()


def _decode(frame: bytes) -> tuple[str, dict[str, Any]]:
    topic, _, body = frame.partition(b" ")
    return topic.decode(), json.loads(body.decode())


class Publisher:
    """A PUB socket. One per independent feed microservice."""

    def __init__(self, address: str, bind: bool = True, hwm: int = 10_000):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.set_hwm(hwm)
        if bind:
            self._sock.bind(address)
        else:
            self._sock.connect(address)

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        self._sock.send(_encode(topic, payload))

    def close(self) -> None:
        self._sock.close(linger=0)


class AsyncSubscriber:

    def __init__(self, address: str, topics: list[str], hwm: int = 1000,
                 bind: bool = False):
        self._ctx = zmq.asyncio.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.set_hwm(hwm)
        if bind:
            self._sock.bind(address)
        else:
            self._sock.connect(address)
        for t in topics:
            self._sock.setsockopt(zmq.SUBSCRIBE, t.encode())

    async def recv(self) -> tuple[str, dict[str, Any]]:
        frame = await self._sock.recv()
        return _decode(frame)

    def close(self) -> None:
        self._sock.close(linger=0)


class AsyncPublisher:
    """Async PUB socket (for services that already run inside an event loop)."""

    def __init__(self, address: str, bind: bool = True, hwm: int = 10_000):
        self._ctx = zmq.asyncio.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.set_hwm(hwm)
        if bind:
            self._sock.bind(address)
        else:
            self._sock.connect(address)

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        await self._sock.send(_encode(topic, payload))

    def close(self) -> None:
        self._sock.close(linger=0)
