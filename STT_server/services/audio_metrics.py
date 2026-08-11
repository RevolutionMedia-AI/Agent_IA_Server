import time
import statistics

_FALLBACK: dict[str, "CallMetrics"] = {}


class CallMetrics:
    def __init__(self, call_id: str):
        self.call_id = call_id
        self.counters: dict[str, int] = {}
        self.gauges: dict[str, float] = {}
        self.latencies_ms: dict[str, list[float]] = {}
        self.created_at = time.monotonic()

    def incr(self, name: str, by: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + by

    def gauge(self, name: str, value: float) -> None:
        self.gauges[name] = value

    def observe_ms(self, name: str, value_ms: float) -> None:
        lst = self.latencies_ms.setdefault(name, [])
        lst.append(value_ms)
        if len(lst) > 100:
            del lst[:-100]

    def _percentiles(self, lst: list[float]) -> tuple[float, float, int]:
        s = sorted(lst); n = len(s)
        p50 = statistics.median(s) if s else 0.0
        p99 = s[min(int(0.99 * n), n - 1)] if n else 0.0
        return (p50, p99, n)

    def summary(self) -> dict:
        lat = {k: self._percentiles(v) for k, v in self.latencies_ms.items()}
        return {
            "call_id": self.call_id,
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "latency_p50_p99": lat,
            "age_seconds": time.monotonic() - self.created_at,
        }

    def to_log_line(self) -> str:
        lat = ",".join(f"{k}:[{p50:.1f},{p99:.1f},{n}]" for k, (p50, p99, n) in ((k, self._percentiles(v)) for k, v in self.latencies_ms.items()))
        ctr = ",".join(f"{k}:{v}" for k, v in self.counters.items())
        ggs = ",".join(f"{k}:{v}" for k, v in self.gauges.items())
        age = time.monotonic() - self.created_at
        return f"call={self.call_id} age={age:.1f}s counters={{{ctr}}} gauges={{{ggs}}} latencies_ms={{{lat}}}"


def attach_metrics(session, call_id: str) -> CallMetrics:
    metrics = getattr(session, "metrics", None)
    if metrics is None:
        metrics = CallMetrics(call_id)
        try:
            session.metrics = metrics
        except AttributeError:
            _FALLBACK[call_id] = metrics
    return metrics


if __name__ == "__main__":
    m = CallMetrics("test_call")
    for i in range(10):
        m.incr("seq_gaps")
        m.observe_ms("mark_ack_rtt", i * 5.0)
    m.gauge("queue_depth_playback", 7)
    print(m.to_log_line())
    print("audio_metrics: OK")
