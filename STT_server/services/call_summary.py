import asyncio
import logging

try:
    from .audio_metrics import CallMetrics
except ImportError:
    from audio_metrics import CallMetrics  # ponytail: lets __main__ self-test run flat


async def emit_call_summary(log, session):
    metrics = getattr(session, "metrics", None)
    if metrics is None:
        log.info("[CALL_SUMMARY] session=%s (no metrics attached)", session.session_key)
        return
    log.info("[CALL_SUMMARY] %s", metrics.to_log_line())


if __name__ == "__main__":
    log = logging.getLogger("test")

    class S:
        session_key = "fake"
        metrics = CallMetrics("fake")
    S.metrics.incr("seq_gaps", 3)
    asyncio.run(emit_call_summary(log, S()))
    print("call_summary: OK")
