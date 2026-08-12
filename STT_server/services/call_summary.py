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
    # ponytail: AUDIO-001 + AUDIO-004 — augment the per-call summary
    # with the queue overflow counters (collected by common.py's
    # enqueue helpers) and the frame-processor stats (collected by
    # playback_service's AudioFrameProcessor). These are the metrics
    # the audit marks as NEEDS_MEASUREMENT_FIRST — exposing them per
    # call gives operators the queue drops / tail bytes / high-water
    # data they need to size the policy later.
    extras: list[str] = []
    try:
        from .common import queue_overflow_stats
        qstats = queue_overflow_stats()
        if any(qstats.values()):
            extras.append(f"queues={qstats}")
    except Exception:
        pass
    try:
        frame_proc = getattr(session, "_playback_frame_proc", None)
        if frame_proc is not None:
            fp_stats = frame_proc.stats()
            if any(fp_stats.values()):
                extras.append(f"frames={fp_stats}")
    except Exception:
        pass
    line = metrics.to_log_line()
    if extras:
        line = f"{line} {' '.join(extras)}"
    log.info("[CALL_SUMMARY] %s", line)


if __name__ == "__main__":
    log = logging.getLogger("test")

    class S:
        session_key = "fake"
        metrics = CallMetrics("fake")
    S.metrics.incr("seq_gaps", 3)
    asyncio.run(emit_call_summary(log, S()))
    print("call_summary: OK")
