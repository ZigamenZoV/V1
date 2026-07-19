"""Сообщения между процессами и помощники очередей.

Брокер — multiprocessing.Queue (спека §8). Детекции при отстающем
потребителе вытесняются по принципу drop-oldest: деградация по частоте,
не по задержке — как и для кадров (§3).
"""
from __future__ import annotations

import queue
from dataclasses import dataclass

import numpy as np

# out_queue (в main) несёт тегированные кортежи:
#   ("event", Event)
#   ("audit", camera_id, count: float)
#   ("stats", dict)                        # см. metrics.ingest_stats
#   ("snapshot", camera_id, jpeg: bytes, ts_wall: float)


@dataclass(slots=True)
class DetMsg:
    """Детекции одного кадра: GPU-процесс → worker камеры."""
    camera_id: str
    seq: int
    ts_mono: float
    ts_wall: float
    boxes: np.ndarray     # Nx4 float32 xyxy
    scores: np.ndarray    # N float32


def put_drop_oldest(q, item) -> bool:
    """Кладёт в очередь; при переполнении выталкивает самый старый элемент.
    Возвращает False, если элемент пришлось вытеснять."""
    try:
        q.put_nowait(item)
        return True
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass
        return False
