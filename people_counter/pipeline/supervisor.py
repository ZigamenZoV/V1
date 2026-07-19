"""Супервизор: собирает конвейер, потребляет события, держит систему 24/7.

Процессная модель (спека §8): 1 процесс — владелец GPU (детектор),
по CPU-worker'у на gate-камеру, брокер — multiprocessing.Queue. Упавший
дочерний процесс перезапускается с backoff. Main-процесс владеет:
Event Store, occupancy-интегратором, ReID-леджером, метриками и FastAPI.

Режим single (pipeline.mode) — те же компоненты в потоках одного процесса:
отладка, mock-режим, машины без нужды в изоляции GIL.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import time
from pathlib import Path

from .. import metrics
from ..config import AppConfig
from ..occupancy import OccupancyIntegrator
from ..reid import Ledger
from ..store import EventStore
from ..types import Event, new_embedding_id
from . import camera_worker, gpu_process
from .frames import FrameRing, InProcRing, ring_name

log = logging.getLogger("supervisor")


class EventBus:
    """Раздача live-сообщений SSE-подписчикам дашборда."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: list[queue.Queue] = []

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=500)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, item: dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(item)
            except queue.Full:
                pass                      # медленный подписчик теряет сообщения


class Supervisor:
    def __init__(self, cfg: AppConfig, cfg_path: str | Path) -> None:
        self.cfg = cfg
        self.cfg_path = str(cfg_path)
        self.store = EventStore(cfg.store.path)
        initial = self.store.last_occupancy() or 0.0
        self.occupancy = OccupancyIntegrator(cfg.audit, initial)
        if initial:
            log.info("occupancy restored from DB: %.0f", initial)
        self.ledger = Ledger(cfg.reid) if cfg.reid.enabled else None
        self.embedder = self._make_embedder()
        self.bus = EventBus()
        self.snapshots: dict[str, tuple[bytes, float]] = {}
        self.stats_cache: dict[str, dict] = {}
        self.started_at = time.time()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._procs: dict[str, mp.Process] = {}
        self._proc_factories: dict[str, object] = {}
        self._restart_at: dict[str, float] = {}
        self._rings: dict[str, object] = {}
        self.out_q = None

    def _make_embedder(self):
        if not (self.cfg.reid.enabled):
            return None
        try:
            from ..reid.embedder import OsnetEmbedder
            return OsnetEmbedder(self.cfg.reid.model)
        except (FileNotFoundError, ImportError) as e:
            log.warning("ReID embedder unavailable (%s) — events go without embeddings, "
                        "exits will be unmatched", e)
            return None

    # ------------------------------------------------------------------ start
    def start(self) -> None:
        if self.cfg.pipeline.mode == "multiprocess":
            self._start_multiprocess()
        else:
            self._start_single()
        self._start_api()
        log.info("service started: mode=%s, cameras=%d (gate=%d), api=http://%s:%d",
                 self.cfg.pipeline.mode, len(self.cfg.enabled_cameras()),
                 len(self.cfg.gate_cameras()), self.cfg.api.host, self.cfg.api.port)
        try:
            self._consume_loop()
        except KeyboardInterrupt:
            log.info("Ctrl+C — shutting down")
        finally:
            self.stop()

    def _start_multiprocess(self) -> None:
        ctx = mp.get_context("spawn")
        self._mp_stop = ctx.Event()
        self.out_q = ctx.Queue(2048)
        size = self.cfg.detector.input_size
        det_queues: dict[str, object] = {}
        handles: dict[str, object] = {}
        for cam in self.cfg.gate_cameras():
            det_queues[cam.id] = ctx.Queue(self.cfg.pipeline.det_queue)
            ring = FrameRing.create(ctx, ring_name(cam.id), self.cfg.pipeline.ring_slots, size, size)
            self._rings[cam.id] = ring
            handles[cam.id] = ring.h

        def gpu_factory() -> mp.Process:
            return ctx.Process(target=gpu_process.entry, name="pc-gpu",
                               args=(self.cfg_path, handles, det_queues, self.out_q, self._mp_stop))

        self._proc_factories["gpu"] = gpu_factory
        for cam in self.cfg.gate_cameras():
            cam_id = cam.id

            def worker_factory(cam_id=cam_id) -> mp.Process:
                return ctx.Process(target=camera_worker.entry, name=f"pc-worker-{cam_id}",
                                   args=(self.cfg_path, cam_id, handles[cam_id],
                                         det_queues[cam_id], self.out_q, self._mp_stop))

            self._proc_factories[f"worker-{cam_id}"] = worker_factory

        for name, factory in self._proc_factories.items():
            p = factory()
            p.start()
            self._procs[name] = p
            log.info("process %s started (pid=%d)", name, p.pid)

        t = threading.Thread(target=self._monitor_children, name="monitor", daemon=True)
        t.start()
        self._threads.append(t)

    def _start_single(self) -> None:
        self._mp_stop = threading.Event()
        self.out_q = queue.Queue(2048)
        det_queues: dict[str, object] = {}
        rings: dict[str, object] = {}
        for cam in self.cfg.gate_cameras():
            det_queues[cam.id] = queue.Queue(self.cfg.pipeline.det_queue)
            rings[cam.id] = InProcRing(self.cfg.pipeline.ring_slots)
            self._rings[cam.id] = rings[cam.id]

        t = threading.Thread(target=gpu_process.run, name="gpu",
                             args=(self.cfg, rings, det_queues, self.out_q, self._mp_stop),
                             daemon=True)
        t.start()
        self._threads.append(t)
        for cam in self.cfg.gate_cameras():
            t = threading.Thread(target=camera_worker.run, name=f"worker-{cam.id}",
                                 args=(self.cfg, cam.id, rings[cam.id], det_queues[cam.id],
                                       self.out_q, self._mp_stop),
                                 daemon=True)
            t.start()
            self._threads.append(t)

    def _start_api(self) -> None:
        import uvicorn

        from ..api.app import create_app
        app = create_app(self)
        config = uvicorn.Config(app, host=self.cfg.api.host, port=self.cfg.api.port,
                                log_level="warning", access_log=False)
        self._api_server = uvicorn.Server(config)
        t = threading.Thread(target=self._api_server.run, name="api", daemon=True)
        t.start()
        self._threads.append(t)

    # ------------------------------------------------------------- monitoring
    def _monitor_children(self) -> None:
        """Рестарт упавших дочерних процессов (стабильность 24/7)."""
        while not self._stop.is_set():
            time.sleep(2.0)
            if self._stop.is_set():
                return
            for name, p in list(self._procs.items()):
                if p.is_alive():
                    continue
                now = time.monotonic()
                if now < self._restart_at.get(name, 0.0):
                    continue
                self._restart_at[name] = now + 5.0
                log.error("process %s died (exitcode=%s) — restarting", name, p.exitcode)
                try:
                    newp = self._proc_factories[name]()  # type: ignore[operator]
                    newp.start()
                    self._procs[name] = newp
                except Exception:
                    log.exception("failed to restart %s", name)

    # ---------------------------------------------------------------- consume
    def _consume_loop(self) -> None:
        occ_int = self.cfg.store.occupancy_snapshot_s
        next_occ = time.monotonic() + occ_int
        while not self._stop.is_set():
            try:
                item = self.out_q.get(timeout=0.5)
            except queue.Empty:
                item = None
            if item is not None:
                try:
                    self._dispatch(item)
                except Exception:
                    log.exception("failed to process %r message", item[0] if item else None)
            now = time.monotonic()
            if now >= next_occ:
                next_occ = now + occ_int
                self.store.log_occupancy(self.occupancy.value, "snapshot")
                if self.ledger:
                    self.ledger.maybe_dump()
                    metrics.LEDGER_SIZE.set(self.ledger.size)
                    metrics.UNMATCHED_RATIO.set(self.ledger.unmatched_ratio)
                metrics.OCCUPANCY.set(self.occupancy.value)

    def _dispatch(self, item: tuple) -> None:
        kind = item[0]
        if kind == "event":
            self._on_event(item[1])
        elif kind == "audit":
            self._on_audit(item[1], item[2])
        elif kind == "stats":
            d = item[1]
            metrics.ingest_stats(d)
            key = d.get("kind", "?") + ":" + d.get("camera", "-")
            self.stats_cache[key] = {**d, "_ts": time.time()}
            self.bus.publish({"type": "stats", **d})
        elif kind == "snapshot":
            self.snapshots[item[1]] = (item[2], item[3])

    def _on_event(self, ev: Event) -> None:
        emb = None
        if self.embedder is not None and ev.crop is not None:
            try:
                emb = self.embedder.embed(ev.crop)
            except Exception:
                log.exception("embedding computation failed")
        if ev.type == "enter":
            ev.embedding_id = self.ledger.on_enter(ev.camera_id, emb) if self.ledger \
                else new_embedding_id()
        else:
            ev.embedding_id = new_embedding_id()
            if self.ledger:
                m = self.ledger.on_exit(ev.camera_id, emb)
                ev.matched_entry_id = m.entry_id
                ev.unmatched = m.entry_id is None
        value = self.occupancy.apply_event(ev.type)
        self.store.add_event(ev)
        self.store.log_occupancy(value, "events", 1.0 if ev.type == "enter" else -1.0)
        metrics.EVENTS.labels(ev.camera_id, ev.type).inc()
        metrics.OCCUPANCY.set(value)
        if self.ledger:
            metrics.LEDGER_SIZE.set(self.ledger.size)
            metrics.UNMATCHED_RATIO.set(self.ledger.unmatched_ratio)
        log.info("event: %s %s (cam=%s, occupancy=%d)", ev.type, ev.ts, ev.camera_id, round(value))
        self.bus.publish({"type": "event", "ts": ev.ts, "camera_id": ev.camera_id,
                          "event": ev.type, "confidence": ev.confidence,
                          "unmatched": ev.unmatched, "occupancy": int(round(value))})

    def _on_audit(self, camera_id: str, count: float) -> None:
        applied, value, alert = self.occupancy.apply_audit(count)
        self.store.add_audit(camera_id, count, applied, value)
        self.store.log_occupancy(value, "audit", applied)
        snap = self.occupancy.snapshot()
        metrics.AUDIT_RAW.labels(camera_id).set(count)
        metrics.AUDIT_DELTA.set(snap["last_audit_delta"])
        metrics.OCCUPANCY.set(value)
        self.bus.publish({"type": "audit", "camera_id": camera_id, "count": count,
                          "applied": round(applied, 2), "alert": alert,
                          "occupancy": int(round(value))})

    # ------------------------------------------------------------------ state
    def health(self) -> dict:
        now = time.time()
        cams = {}
        degraded: list[str] = []
        for cam in self.cfg.enabled_cameras():
            cap = self.stats_cache.get(f"capture:{cam.id}", {})
            wrk = self.stats_cache.get(f"worker:{cam.id}", {})
            snap = self.snapshots.get(cam.id)
            stats_age = round(now - cap["_ts"], 1) if cap else None
            cams[cam.id] = {
                "role": cam.role,
                "decode_fps": cap.get("decode_fps"),
                "restarts": cap.get("restarts"),
                "worker_fps": wrk.get("fps"),
                "active_tracks": wrk.get("active_tracks"),
                "stats_age_s": stats_age,
                "snapshot_age_s": round(now - snap[1], 1) if snap else None,
            }
            if cam.role == "gate" and (stats_age is None or stats_age > 15 or
                                       (cap.get("decode_fps") or 0) <= 0):
                degraded.append(cam.id)
        infer = self.stats_cache.get("infer:-", {})
        out = {
            "status": "degraded" if degraded else "ok",
            "degraded_cameras": degraded,
            "uptime_s": round(now - self.started_at, 1),
            "mode": self.cfg.pipeline.mode,
            "detector": {"backend": self.cfg.detector.backend,
                         "infer_fps": infer.get("infer_fps"),
                         "batch_ms": infer.get("batch_ms")},
            "cameras": cams,
            **self.occupancy.snapshot(),
        }
        if self.ledger:
            out["ledger"] = {"size": self.ledger.size,
                             "unmatched_ratio": round(self.ledger.unmatched_ratio, 3)}
        return out

    # ------------------------------------------------------------------- stop
    def stop(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._mp_stop.set()
        if hasattr(self, "_api_server"):
            self._api_server.should_exit = True
        for name, p in self._procs.items():
            p.join(timeout=10)
            if p.is_alive():
                log.warning("process %s did not stop — terminating", name)
                p.terminate()
        # добираем финальные события, которые дети сбросили при остановке
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                self._dispatch(self.out_q.get_nowait())
            except queue.Empty:
                break
            except Exception:
                break
        for r in self._rings.values():
            r.close()
        self.store.log_occupancy(self.occupancy.value, "snapshot")
        if self.ledger:
            self.ledger.dump_now()
        self.store.close()
        log.info("service stopped, occupancy=%d", round(self.occupancy.value))
