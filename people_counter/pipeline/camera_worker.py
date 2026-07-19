"""CPU-worker камеры: трекер + зоны + FSM + финализация траекторий
(спека §0, §8 — по процессу на камеру, GIL обходится процессами).

Вход — очередь детекций от GPU-процесса; кадры для кропов — из
shared-memory кольца по seq. Выход — события enter/exit в main.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time

import numpy as np

from ..config import AppConfig, load_config
from ..counting import TrajectoryCounter, ZonePair
from ..log import setup as log_setup
from ..track import make_tracker
from ..types import Detections, Frame
from .messages import DetMsg

log = logging.getLogger("worker")


def entry(cfg_path: str, camera_id: str, ring_handle, det_q, out_q, stop_event) -> None:
    """Точка входа дочернего процесса (spawn)."""
    cfg = load_config(cfg_path)
    log_setup(f"worker-{camera_id}", cfg.log_dir, cfg.log_level)
    from .frames import FrameRing
    ring = FrameRing.attach(ring_handle)
    try:
        run(cfg, camera_id, ring, det_q, out_q, stop_event)
    finally:
        ring.close()


def run(cfg: AppConfig, camera_id: str, ring, det_q, out_q, stop_event) -> None:
    cam = cfg.camera(camera_id)
    size = cfg.detector.input_size
    tracker = make_tracker(cfg.tracker, cam.fps)
    counter = TrajectoryCounter(cam, ZonePair(cam.zones), cam.fps, cfg.reid)
    log.info("[%s] worker started: tracker=%s, fps=%.1f", camera_id,
             cfg.tracker.backend, cam.fps)

    dummy = None  # кадр-заглушка, если слот кольца уже перезаписан
    processed = 0
    next_stats = time.monotonic() + cfg.pipeline.stats_interval_s

    parent = mp.parent_process()          # None в single-process режиме
    while not stop_event.is_set():
        try:
            msg: DetMsg = det_q.get(timeout=0.25)
        except queue.Empty:
            if parent is not None and not parent.is_alive():
                log.error("[%s] parent process died — exiting", camera_id)
                if hasattr(out_q, "cancel_join_thread"):
                    out_q.cancel_join_thread()   # буферы читать некому
                return
            continue

        frame = ring.get(camera_id, msg.seq)
        if frame is None:
            if dummy is None:
                dummy = Frame(camera_id, -1, 0.0, 0.0,
                              np.zeros((size, size, 3), np.uint8))
            tracker_frame = dummy
        else:
            tracker_frame = frame

        det = Detections(msg.boxes, msg.scores)
        tracks, removed = tracker.update(det, tracker_frame)
        counter.update(tracks, frame)
        for ev in counter.finalize(removed):
            out_q.put(("event", ev))
        processed += 1

        now = time.monotonic()
        if now >= next_stats:
            interval = cfg.pipeline.stats_interval_s
            next_stats = now + interval
            out_q.put(("stats", {
                "kind": "worker", "camera": camera_id,
                "fps": round(processed / interval, 2),
                "active_tracks": counter.active_tracks,
            }))
            processed = 0

    # финализируем всё живое при остановке — события не теряются
    for ev in counter.drain():
        out_q.put(("event", ev))
    log.info("[%s] worker stopped", camera_id)
