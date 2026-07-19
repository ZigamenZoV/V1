"""Источники кадров и latest-frame slot.

Политика (спека §3): очереди кадров запрещены. Capture-поток пишет кадр
в слот размера 1, потребитель всегда берёт свежайший — при просадке FPS
система деградирует по частоте, а не по задержке.
"""
from __future__ import annotations

import threading
from typing import Protocol

from ..config import CameraConfig
from ..types import Frame


class LatestSlot:
    """Потокобезопасный слот «последний кадр» (размер 1)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Frame | None = None

    def put(self, frame: Frame) -> None:
        with self._lock:
            self._frame = frame

    def get(self) -> Frame | None:
        """Свежайший кадр (может вернуть уже виденный seq — фильтрует потребитель)."""
        with self._lock:
            return self._frame


class Source(Protocol):
    """Источник кадров камеры. start() запускает capture-поток,
    кадры появляются в slot."""

    camera_id: str
    slot: LatestSlot

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def restart(self) -> None: ...        # принудительный рестарт (watchdog)
    def stats(self) -> dict: ...          # decode_fps, frames, restarts, stale_s
    def is_stale(self, max_age_s: float) -> bool: ...


def make_source(cam: CameraConfig, width: int, height: int) -> Source:
    """Фабрика источника по конфигу камеры."""
    if cam.source == "synthetic":
        from .synthetic import SyntheticSource
        return SyntheticSource(cam, width, height)
    from .ffmpeg_source import FFmpegSource
    return FFmpegSource(cam, width, height)
