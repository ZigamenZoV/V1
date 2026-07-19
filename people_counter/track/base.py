"""Фабрика трекера (спека §4): boxmot BoT-SORT / ByteTrack, смена конфигом.
`simple` — встроенный лёгкий трекер без torch-зависимостей (mock-режим,
отладка, fallback при CPU-узком месте)."""
from __future__ import annotations

from ..config import TrackerConfig
from ..types import Tracker


def make_tracker(cfg: TrackerConfig, fps: float) -> Tracker:
    if cfg.backend == "simple":
        from .simple import SimpleTracker
        return SimpleTracker(cfg, fps)
    if cfg.backend in ("botsort", "bytetrack"):
        from .botsort import BoxmotTracker
        return BoxmotTracker(cfg, fps)
    raise ValueError(f"unknown tracker backend: {cfg.backend}")
