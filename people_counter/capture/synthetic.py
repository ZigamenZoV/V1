"""Синтетическая сцена: «люди» ходят через входную группу.

Даёт детерминированный (по seed) поток кадров с ground-truth боксами голов —
на нём проверяется весь конвейер (mock-детектор → трекер → FSM → события)
без камер и GPU. Геометрия сцены согласована с зонами в config/mock:
зона A — левая половина двери, зона B — правая (внутренняя).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import CameraConfig
from ..types import Frame
from .base import LatestSlot


@dataclass
class _Agent:
    aid: int
    path: list[tuple[float, float]]        # waypoint'ы центра головы
    speed: float
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    wp: int = 0
    done: bool = False

    def step(self) -> None:
        if self.wp >= len(self.path):
            self.done = True
            return
        target = np.asarray(self.path[self.wp], dtype=float)
        d = target - self.pos
        dist = float(np.hypot(*d))
        if dist <= self.speed:
            self.pos = target
            self.wp += 1
        else:
            self.pos = self.pos + d / dist * self.speed


class SyntheticSource:
    """Источник с интерфейсом Source. Сценарии агентов:
    enter  — OUT→A→B, затем вглубь и деспаун (ожидаем событие enter);
    exit   — B→A→OUT (ожидаем exit);
    passby — вертикальный проход через A (события быть не должно);
    peek   — заглянул в A, коснулся B на мгновение, ушёл (события быть не должно).
    """

    def __init__(self, cam: CameraConfig, width: int, height: int,
                 realtime: bool = True) -> None:
        self.camera_id = cam.id
        self.cam = cam
        self.w, self.h = width, height
        self.realtime = realtime
        self.slot = LatestSlot()
        self._rng = np.random.default_rng(cam.synthetic.seed)
        self._agents: list[_Agent] = []
        self._next_aid = 1
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fps_win: list[float] = []
        self._last_frame_mono = 0.0
        # счётчики сгенерированной «правды» — для сверки в тестах
        self.gt_enters = 0
        self.gt_exits = 0

    # --- генерация сценариев -------------------------------------------------
    def spawn(self, kind: str) -> None:
        """Ручной спаун агента (используется e2e-тестами для детерминизма)."""
        self._spawn(kind)

    def _spawn(self, kind: str) -> None:
        w, h = self.w, self.h
        s = self.cam.synthetic
        y = float(self._rng.uniform(h * 0.30, h * 0.70))
        jitter = lambda: float(self._rng.uniform(-h * 0.05, h * 0.05))  # noqa: E731
        if kind == "enter":
            path = [(w * 0.18, y), (w * 0.40, y + jitter()), (w * 0.75, y + jitter()), (w * 1.10, y)]
            start = (-30.0, y)
            self.gt_enters += 1
        elif kind == "exit":
            path = [(w * 0.75, y), (w * 0.40, y + jitter()), (w * 0.18, y + jitter()), (-30.0, y)]
            start = (float(w) + 30.0, y)
            self.gt_exits += 1
        elif kind == "passby":
            x = float(self._rng.uniform(w * 0.12, w * 0.30))      # только зона A
            path = [(x, h * 0.2), (x, h * 0.8), (x, float(h) + 30.0)]
            start = (x, -30.0)
        else:  # peek: заходит в A, на мгновение пересекает границу B, возвращается
            path = [(w * 0.30, y), (w * 0.52, y), (w * 0.30, y), (-30.0, y)]
            start = (-30.0, y)
        a = _Agent(self._next_aid, path, s.speed_px)
        a.pos = np.asarray(start, dtype=float)
        self._next_aid += 1
        self._agents.append(a)

    def _maybe_spawn(self) -> None:
        s = self.cam.synthetic
        per_frame = 1.0 / max(self.cam.fps, 1e-6) / 60.0
        for kind, rate in (("enter", s.enters_per_min), ("exit", s.exits_per_min),
                           ("passby", s.passby_per_min), ("peek", s.peek_per_min)):
            if self._rng.random() < rate * per_frame:
                self._spawn(kind)

    # --- кадры ---------------------------------------------------------------
    def render(self) -> Frame:
        """Синхронный шаг сцены (используется и напрямую в e2e-тестах)."""
        self._maybe_spawn()
        img = np.full((self.h, self.w, 3), 46, np.uint8)
        # дверной проём: граница зон A|B для наглядности снапшотов
        cv2.line(img, (int(self.w * 0.5), 0), (int(self.w * 0.5), self.h), (90, 90, 90), 2)
        boxes, ids = [], []
        r = self.cam.synthetic.head_px / 2.0
        for a in self._agents:
            a.step()
            x, y = a.pos
            if not a.done and -r < x < self.w + r and -r < y < self.h + r:
                cv2.circle(img, (int(x), int(y)), int(r), (200, 180, 160), -1)
                boxes.append([x - r, y - r, x + r, y + r])
                ids.append(a.aid)
        self._agents = [a for a in self._agents if not a.done]
        self._seq += 1
        now = time.monotonic()
        self._last_frame_mono = now
        self._fps_win.append(now)
        self._fps_win = [t for t in self._fps_win if now - t < 5.0]
        return Frame(
            self.camera_id, self._seq, now, time.time(), img,
            gt_boxes=np.asarray(boxes, np.float32).reshape(-1, 4),
            gt_ids=np.asarray(ids, np.int64),
        )

    # --- интерфейс Source ----------------------------------------------------
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"syn-{self.camera_id}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        period = 1.0 / max(self.cam.fps, 1e-6)
        next_t = time.monotonic()
        while not self._stop.is_set():
            self.slot.put(self.render())
            if self.realtime:
                next_t += period
                delay = next_t - time.monotonic()
                if delay > 0:
                    self._stop.wait(delay)
                else:
                    next_t = time.monotonic()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def restart(self) -> None:
        pass

    def is_stale(self, max_age_s: float) -> bool:
        return False

    def stats(self) -> dict:
        return {
            "decode_fps": round(len(self._fps_win) / 5.0, 2),
            "frames": self._seq,
            "restarts": 0,
            "stale_s": 0.0,
        }
