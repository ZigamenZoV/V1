"""Prometheus-метрики (спека §8): FPS декода/инференса, пропуски, треки,
события, unmatched-доля ReID, дельта аудита. Дочерние процессы шлют
stats-словари в main через очередь; main раскладывает их по гейджам.
"""
from __future__ import annotations

from prometheus_client import (CollectorRegistry, Counter, Gauge,
                               generate_latest)

registry = CollectorRegistry()

DECODE_FPS = Gauge("pc_decode_fps", "Decode FPS per camera", ["camera"], registry=registry)
CAPTURE_RESTARTS = Gauge("pc_capture_restarts", "Capture restarts per camera", ["camera"], registry=registry)
INFER_FPS = Gauge("pc_infer_fps", "Detector inference rate, ticks/s", registry=registry)
INFER_MS = Gauge("pc_infer_batch_ms", "Batch inference time, ms", registry=registry)
SKIPPED_FRAMES = Counter("pc_skipped_frames_total", "Frames skipped by latest-frame policy",
                         ["camera"], registry=registry)
ACTIVE_TRACKS = Gauge("pc_active_tracks", "Active tracks per camera", ["camera"], registry=registry)
WORKER_FPS = Gauge("pc_worker_fps", "Tracker processing FPS per camera", ["camera"], registry=registry)
EVENTS = Counter("pc_events_total", "Counting events", ["camera", "type"], registry=registry)
OCCUPANCY = Gauge("pc_occupancy", "Current occupancy", registry=registry)
LEDGER_SIZE = Gauge("pc_ledger_size", "Entries in the ReID ledger", registry=registry)
UNMATCHED_RATIO = Gauge("pc_reid_unmatched_ratio", "Share of exits unmatched in the ledger", registry=registry)
AUDIT_DELTA = Gauge("pc_audit_delta", "Last audit delta (audit_ema - occupancy)", registry=registry)
AUDIT_RAW = Gauge("pc_audit_raw", "Last raw audit count", ["camera"], registry=registry)


def ingest_stats(msg: dict) -> None:
    """Раскладывает stats-словарь от дочернего процесса по метрикам."""
    kind = msg.get("kind")
    if kind == "capture":
        cam = msg["camera"]
        DECODE_FPS.labels(cam).set(msg.get("decode_fps", 0))
        CAPTURE_RESTARTS.labels(cam).set(msg.get("restarts", 0))
    elif kind == "infer":
        INFER_FPS.set(msg.get("infer_fps", 0))
        INFER_MS.set(msg.get("batch_ms", 0))
        for cam, n in (msg.get("skipped") or {}).items():
            if n:
                SKIPPED_FRAMES.labels(cam).inc(n)
    elif kind == "worker":
        cam = msg["camera"]
        ACTIVE_TRACKS.labels(cam).set(msg.get("active_tracks", 0))
        WORKER_FPS.labels(cam).set(msg.get("fps", 0))


def render() -> bytes:
    return generate_latest(registry)
