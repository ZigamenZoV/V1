"""Общие типы данных конвейера."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(slots=True)
class Frame:
    """Кадр в разрешении инференса (BGR). gt_* заполняется только
    синтетическим источником — для mock-детектора и e2e-тестов."""
    camera_id: str
    seq: int
    ts_mono: float
    ts_wall: float
    image: np.ndarray                      # HxWx3 uint8, BGR
    gt_boxes: np.ndarray | None = None     # Nx4 xyxy (px кадра инференса)
    gt_ids: np.ndarray | None = None       # N


@dataclass(slots=True)
class Detections:
    """Головы на одном кадре. Координаты — пиксели кадра инференса."""
    boxes: np.ndarray    # Nx4 float32 xyxy
    scores: np.ndarray   # N float32

    @staticmethod
    def empty() -> "Detections":
        return Detections(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))

    def __len__(self) -> int:
        return int(self.boxes.shape[0])


@dataclass(slots=True)
class Track:
    """Активный трек на текущем кадре."""
    track_id: int
    box: np.ndarray      # 4 float32 xyxy
    score: float

    @property
    def center(self) -> tuple[float, float]:
        b = self.box
        return (float(b[0] + b[2]) / 2.0, float(b[1] + b[3]) / 2.0)


@dataclass(slots=True)
class Event:
    """Событие enter/exit (схема спеки §5.3)."""
    ts: str                      # ISO wall-clock
    camera_id: str
    track_id: int
    type: str                    # "enter" | "exit"
    trajectory_len: int
    confidence: float
    embedding_id: str | None = None
    matched_entry_id: str | None = None   # для exit: id погашенной enter-записи леджера
    unmatched: bool = False               # exit без матча в леджере
    crop: np.ndarray | None = None        # BGR-кроп тела (не сериализуется в БД)

    @staticmethod
    def make(camera_id: str, track_id: int, type_: str,
             trajectory_len: int, confidence: float,
             crop: np.ndarray | None = None) -> "Event":
        now = time.time()
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))
        return Event(
            ts=f"{ts}.{int(now * 1000) % 1000:03d}",
            camera_id=camera_id, track_id=track_id, type=type_,
            trajectory_len=trajectory_len, confidence=confidence, crop=crop,
        )


def new_embedding_id() -> str:
    return uuid.uuid4().hex[:16]


class Detector(Protocol):
    """Абстракция инференс-рантайма (§2.2): рантайм меняется конфигом."""

    input_size: int

    def infer(self, frames: list[Frame]) -> list[Detections]: ...

    def close(self) -> None: ...


class Tracker(Protocol):
    """Абстракция трекера: update() на каждый такт детекции.

    Возвращает (активные подтверждённые треки, id треков, потерянных
    окончательно на этом такте — для финализации траекторий)."""

    def update(self, det: Detections, frame: Frame) -> tuple[list[Track], list[int]]: ...
