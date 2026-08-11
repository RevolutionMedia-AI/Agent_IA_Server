import asyncio
import logging
import time

log = logging.getLogger("stt_server")


class BackoffPolicy:
    """Exponential backoff with jitter for WS reconnects."""
    def __init__(self, base_ms: int = 250, max_ms: int = 8000, factor: float = 2.0, jitter: float = 0.3):
        self.base_ms = base_ms
        self.max_ms = max_ms
        self.factor = factor
        self.jitter = jitter
        self._attempt = 0

    def reset(self) -> None:
        self._attempt = 0

    def next_delay_s(self) -> float:
        raw_ms = self.base_ms * (self.factor ** self._attempt)
        raw_ms = min(raw_ms, self.max_ms)
        import random
        delay = (raw_ms / 1000.0) * (1.0 + random.uniform(-self.jitter, self.jitter))
        # ponytail: clamp final delay so jitter never exceeds max_ms.
        delay = min(max(0.0, delay), self.max_ms / 1000.0)
        self._attempt += 1
        return delay


async def with_backoff(operation, *, policy: BackoffPolicy, max_attempts: int = 5,
                       on_retry=None):
    """Run async operation with exponential backoff. `operation` is async callable
    that returns success or raises. on_retry(attempt, exc, next_delay) is called before each retry."""
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return await operation()
        except Exception as exc:
            last_exc = exc
            delay = policy.next_delay_s()
            if on_retry:
                try:
                    on_retry(attempt, exc, delay)
                except Exception:
                    pass
            if attempt + 1 < max_attempts:
                await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


if __name__ == "__main__":
    import asyncio
    policy = BackoffPolicy(base_ms=50, max_ms=400, factor=2.0)
    delays = [policy.next_delay_s() for _ in range(5)]
    print("delays:", delays)
    assert all(0 < d <= 0.4 for d in delays), f"delay out of bounds: {delays}"

    async def flaky():
        flaky.counter = getattr(flaky, "counter", 0) + 1
        if flaky.counter < 3:
            raise RuntimeError("transient")
        return "ok"

    policy2 = BackoffPolicy(base_ms=10, max_ms=50, factor=2.0)
    result = asyncio.run(with_backoff(flaky, policy=policy2, max_attempts=5))
    print("result:", result)
    assert result == "ok"
    print("reconnect: OK")
