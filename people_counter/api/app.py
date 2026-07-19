"""Сервисный слой: FastAPI (спека §8).

REST: /occupancy, /events, /health, /metrics, /cameras (+снапшоты);
live-дашборд — SSE /stream. Дашборд самодостаточен (inline JS/CSS):
edge-машина может жить без интернета.
"""
from __future__ import annotations

import asyncio
import json
import queue
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from .. import metrics

if TYPE_CHECKING:
    from ..pipeline.supervisor import Supervisor

_STATIC = Path(__file__).parent / "static"


def create_app(sup: "Supervisor") -> FastAPI:
    app = FastAPI(title="People Counter v2", version="2.0.0", docs_url="/docs")

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return (_STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/occupancy")
    def occupancy() -> dict:
        snap = sup.occupancy.snapshot()
        snap["today"] = sup.store.counts_today()
        return snap

    @app.get("/occupancy/history")
    def occupancy_history(hours: float = 24.0) -> list[dict]:
        return sup.store.occupancy_history(hours)

    @app.get("/events")
    def events(limit: int = 100, since: str | None = None,
               camera: str | None = None) -> list[dict]:
        return sup.store.events(limit=min(limit, 1000), since=since, camera_id=camera)

    @app.get("/health")
    def health() -> dict:
        return sup.health()

    @app.get("/metrics")
    def prom_metrics() -> Response:
        return Response(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/cameras")
    def cameras() -> list[dict]:
        h = sup.health()["cameras"]
        return [{"id": c.id, "role": c.role, "fps": c.fps, "source": c.source,
                 **(h.get(c.id) or {})} for c in sup.cfg.enabled_cameras()]

    @app.get("/cameras/{camera_id}/snapshot.jpg")
    def snapshot(camera_id: str) -> Response:
        item = sup.snapshots.get(camera_id)
        if item is None:
            raise HTTPException(404, f"no snapshot for camera {camera_id}")
        return Response(item[0], media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        sub = sup.bus.subscribe()

        async def gen():
            try:
                yield "retry: 3000\n\n"
                while True:
                    try:
                        item = await asyncio.to_thread(sub.get, True, 15.0)
                        yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                    except queue.Empty:
                        yield ": ping\n\n"
            finally:
                sup.bus.unsubscribe(sub)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    return app
