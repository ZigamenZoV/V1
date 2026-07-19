"""GPU-процесс — единственный владелец GPU (спека §0, §8).

Внутри: capture-потоки всех камер (NVDEC, latest-frame slot), батч-инференс
детектора по gate-камерам, JPEG-снапшоты для дашборда, плановый аудит по
обзорным камерам (вытесняется в паузы — вне горячего пути) и watchdog
зависших RTSP. ReID здесь не живёт — он событийный и работает на CPU в
main-процессе.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import time

import cv2
import numpy as np

from ..capture import make_source
from ..config import AppConfig, load_config
from ..counting.zones import ZonePair
from ..detect import make_detector
from ..log import setup as log_setup
from .messages import DetMsg, put_drop_oldest

log = logging.getLogger("gpu")


def entry(cfg_path: str, ring_handles: dict, det_queues: dict, out_q, stop_event) -> None:
    """Точка входа дочернего процесса (spawn)."""
    cfg = load_config(cfg_path)
    log_setup("gpu", cfg.log_dir, cfg.log_level)
    from .frames import FrameRing
    rings = {cam: FrameRing.attach(h) for cam, h in ring_handles.items()}
    try:
        run(cfg, rings, det_queues, out_q, stop_event)
    finally:
        for r in rings.values():
            r.close()


def run(cfg: AppConfig, rings: dict, det_queues: dict, out_q, stop_event) -> None:
    size = cfg.detector.input_size
    cams = cfg.enabled_cameras()
    gates = [c.id for c in cfg.gate_cameras()]
    overviews = [c.id for c in cfg.overview_cameras()]

    sources = {c.id: make_source(c, size, size) for c in cams}
    for s in sources.values():
        s.start()
    log.info("capture started: %s", ", ".join(sources))

    detector = None
    try:
        detector = make_detector(cfg.detector, cfg.detector_batch())
        log.info("detector: %s, batch=%d, input=%d",
                 cfg.detector.backend, cfg.detector_batch(), size)

        audit_backend = None
        if cfg.audit.enabled and overviews:
            from ..audit import make_audit_backend
            audit_backend = make_audit_backend(cfg.audit)

        zone_overlays = {c.id: ZonePair(c.zones) for c in cfg.gate_cameras() if c.zones}
        orphaned = False
        last_seq: dict[str, int] = {c: -1 for c in gates}
        last_boxes: dict[str, np.ndarray] = {}
        skipped: dict[str, int] = {c: 0 for c in gates}
        infer_times: list[float] = []
        ticks = 0
        p = cfg.pipeline
        now = time.monotonic()
        next_snapshot, next_stats, next_watchdog = now, now + p.stats_interval_s, now + 2.0
        next_audit = {c: now + cfg.audit.interval_s for c in overviews}

        while not stop_event.is_set():
            # --- батч свежих кадров gate-камер (latest-frame policy) ---
            batch = []
            for cam_id in gates:
                f = sources[cam_id].slot.get()
                if f is not None and f.seq != last_seq[cam_id]:
                    if last_seq[cam_id] >= 0:
                        skipped[cam_id] += max(0, f.seq - last_seq[cam_id] - 1)
                    last_seq[cam_id] = f.seq
                    batch.append(f)

            if batch:
                t0 = time.perf_counter()
                try:
                    dets = detector.infer(batch)
                except Exception:
                    log.exception("inference failed — pausing 1s and continuing")
                    time.sleep(1.0)
                    continue
                infer_times.append((time.perf_counter() - t0) * 1000)
                ticks += 1
                for f, det in zip(batch, dets):
                    if f.camera_id in rings:
                        rings[f.camera_id].put(f)
                    last_boxes[f.camera_id] = det.boxes
                    put_drop_oldest(det_queues[f.camera_id],
                                    DetMsg(f.camera_id, f.seq, f.ts_mono, f.ts_wall,
                                           det.boxes, det.scores))
            else:
                time.sleep(0.005)

            now = time.monotonic()

            # --- аудит по расписанию (спека §7): такт с наименьшей загрузкой —
            # пропустить один цикл детекции приемлемо ---
            if audit_backend is not None:
                for cam_id in overviews:
                    if now >= next_audit[cam_id]:
                        next_audit[cam_id] = now + cfg.audit.interval_s
                        f = sources[cam_id].slot.get()
                        if f is not None:
                            try:
                                cnt = audit_backend.count(f.image)
                                out_q.put(("audit", cam_id, float(cnt)))
                            except Exception:
                                log.exception("audit failed for %s", cam_id)

            # --- снапшоты для дашборда/калибровки ---
            if now >= next_snapshot:
                next_snapshot = now + p.snapshot_interval_s
                for cam_id, src in sources.items():
                    f = src.slot.get()
                    if f is None:
                        continue
                    img = f.image
                    if cam_id in zone_overlays:
                        img = zone_overlays[cam_id].draw(img)
                    else:
                        img = img.copy()
                    for b in last_boxes.get(cam_id, ())[:200]:
                        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                                      (0, 220, 255), 1)
                    w = p.snapshot_width
                    h = int(img.shape[0] * w / img.shape[1])
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                    ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        put_drop_oldest(out_q, ("snapshot", cam_id, jpeg.tobytes(), f.ts_wall))

            # --- watchdog RTSP (спека §8) + сиротство: родитель умер → выходим ---
            if now >= next_watchdog:
                next_watchdog = now + 2.0
                for cam_id, src in sources.items():
                    if src.is_stale(p.stale_restart_s):
                        src.restart()
                parent = mp.parent_process()
                if parent is not None and not parent.is_alive():
                    log.error("parent process died — exiting")
                    orphaned = True
                    break

            # --- статистика ---
            if now >= next_stats:
                interval = p.stats_interval_s
                next_stats = now + interval
                for cam_id, src in sources.items():
                    out_q.put(("stats", {"kind": "capture", "camera": cam_id, **src.stats()}))
                out_q.put(("stats", {
                    "kind": "infer",
                    "infer_fps": round(ticks / interval, 2),
                    "batch_ms": round(float(np.mean(infer_times)), 2) if infer_times else 0.0,
                    "skipped": dict(skipped),
                }))
                ticks = 0
                infer_times.clear()
                skipped = {c: 0 for c in gates}
    finally:
        for s in sources.values():
            s.stop()
        if detector is not None:
            detector.close()
        # родитель мёртв → буферы очередей никто не прочитает; без cancel
        # процесс повиснет, доливая их в переполненный пайп при выходе
        if orphaned:
            for q in (*det_queues.values(), out_q):
                if hasattr(q, "cancel_join_thread"):
                    q.cancel_join_thread()
        log.info("gpu process stopped")
