"""arXiv rate limiting: one request per 3 s, one connection at a time (D-042, D-064)."""

import asyncio
import math
import time


class ArxivRateLimiter:
    """Serializes arXiv requests and spaces them.

    The semaphore is held across the whole request, not just its start. A token-bucket
    limiter paces starts only, which lets a second request begin while a slow one is still
    open -- measured at 2 concurrent connections, breaking arXiv's "single connection"
    rule. Holding it for the duration makes the effective spacing
    max(min_interval, request_duration) with no extra code.
    """

    def __init__(self, min_interval: float) -> None:
        self._semaphore = asyncio.Semaphore(1)
        self._min_interval = min_interval
        self._last_request = -math.inf

    async def __aenter__(self) -> None:
        await self._semaphore.acquire()
        wait = self._min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request = time.monotonic()

    async def __aexit__(self, *exc: object) -> None:
        self._semaphore.release()