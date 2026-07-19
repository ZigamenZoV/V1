"""Кольцо кадров между GPU-процессом и worker'ом камеры.

Worker'у нужен кадр с номером seq, на котором считались детекции (кропы
для ReID). Полные кадры через pickle-очередь гонять дорого — используется
SharedMemory-кольцо: GPU-процесс пишет, worker читает по seq. Если worker
отстал больше чем на slots кадров — кадр уже перезаписан, вернётся None
(трекинг продолжается по боксам, страдает только кроп).
"""
from __future__ import annotations

import multiprocessing as mp
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

from ..types import Frame


@dataclass
class RingHandle:
    """Передаётся дочернему процессу как аргумент (picklable при spawn)."""
    shm_name: str
    slots: int
    h: int
    w: int
    meta: object      # mp.Array('d', slots*3): (seq, ts_mono, ts_wall) на слот
    latest: object    # mp.Value('q'): последний записанный seq


class FrameRing:
    """Обёртка над SharedMemory. Создатель — супервизор (переживает рестарты
    дочерних процессов); GPU-процесс и worker подключаются по handle."""

    def __init__(self, handle: RingHandle, shm: shared_memory.SharedMemory,
                 owner: bool) -> None:
        self.h = handle
        self._shm = shm
        self._owner = owner
        self._view = np.ndarray((handle.slots, handle.h, handle.w, 3),
                                dtype=np.uint8, buffer=shm.buf)

    # --- фабрики ---------------------------------------------------------
    @staticmethod
    def create(ctx, name: str, slots: int, h: int, w: int) -> "FrameRing":
        shm = shared_memory.SharedMemory(name=name, create=True, size=slots * h * w * 3)
        handle = RingHandle(name, slots, h, w,
                            ctx.Array("d", slots * 3), ctx.Value("q", -1))
        return FrameRing(handle, shm, owner=True)

    @staticmethod
    def attach(handle: RingHandle) -> "FrameRing":
        shm = shared_memory.SharedMemory(name=handle.shm_name, create=False)
        return FrameRing(handle, shm, owner=False)

    # --- запись/чтение ---------------------------------------------------
    def put(self, frame: Frame) -> None:
        idx = frame.seq % self.h.slots
        self._view[idx] = frame.image
        with self.h.meta.get_lock():
            self.h.meta[idx * 3:idx * 3 + 3] = (float(frame.seq), frame.ts_mono, frame.ts_wall)
        with self.h.latest.get_lock():
            self.h.latest.value = frame.seq

    def get(self, camera_id: str, seq: int) -> Frame | None:
        latest = self.h.latest.value
        if seq < 0 or latest < 0 or latest - seq >= self.h.slots - 1:
            return None                       # перезаписан или ещё не записан
        idx = seq % self.h.slots
        with self.h.meta.get_lock():
            s, ts_mono, ts_wall = self.h.meta[idx * 3:idx * 3 + 3]
        if int(s) != seq:
            return None
        img = self._view[idx].copy()
        with self.h.meta.get_lock():          # гонка с писателем: перечитать метку
            if int(self.h.meta[idx * 3]) != seq:
                return None
        return Frame(camera_id, seq, ts_mono, ts_wall, img)

    def close(self) -> None:
        self._shm.close()
        if self._owner:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass


class InProcRing:
    """Кольцо для single-process режима: те же put/get, без shared memory."""

    def __init__(self, slots: int) -> None:
        self._slots = slots
        self._lock = threading.Lock()
        self._frames: OrderedDict[int, Frame] = OrderedDict()

    def put(self, frame: Frame) -> None:
        with self._lock:
            self._frames[frame.seq] = frame
            while len(self._frames) > self._slots:
                self._frames.popitem(last=False)

    def get(self, camera_id: str, seq: int) -> Frame | None:
        with self._lock:
            return self._frames.get(seq)

    def close(self) -> None:
        pass


def ring_name(camera_id: str) -> str:
    """Уникальное имя shm на запуск (два инстанса сервиса не пересекутся)."""
    import os
    safe = "".join(ch if ch.isalnum() else "_" for ch in camera_id)
    return f"pc_{os.getpid()}_{safe}_{int(time.time()) % 100000}"
