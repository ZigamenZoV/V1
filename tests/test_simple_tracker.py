"""Встроенный трекер: устойчивость ID при пропусках детекций и разделение
соседних объектов."""
import numpy as np

from people_counter.config import TrackerConfig
from people_counter.track.simple import SimpleTracker
from people_counter.types import Detections, Frame


def frame() -> Frame:
    return Frame("t", 0, 0.0, 0.0, np.zeros((560, 560, 3), np.uint8))


def det(*boxes: tuple[float, float, float, float], score: float = 0.9) -> Detections:
    if not boxes:
        return Detections.empty()
    return Detections(np.array(boxes, np.float32),
                      np.full(len(boxes), score, np.float32))


def box_at(x: float, y: float, r: float = 9.0) -> tuple[float, float, float, float]:
    return (x - r, y - r, x + r, y + r)


def test_id_stable_through_dropout():
    trk = SimpleTracker(TrackerConfig(backend="simple", track_buffer_s=2.0), fps=10)
    ids = set()
    x = 100.0
    for i in range(30):
        x += 4.0
        if 10 <= i < 13:                       # 3 кадра детекция пропала
            d = det()
        else:
            d = det(box_at(x, 200))
        active, _ = trk.update(d, frame())
        ids.update(t.track_id for t in active)
    assert len(ids) == 1                       # ID пережил пропуск


def test_two_parallel_objects_distinct_ids():
    trk = SimpleTracker(TrackerConfig(backend="simple"), fps=10)
    seen: dict[int, list[float]] = {}
    for i in range(25):
        x = 100 + i * 4.0
        active, _ = trk.update(det(box_at(x, 150), box_at(x, 400)), frame())
        for t in active:
            seen.setdefault(t.track_id, []).append(t.center[1])
    assert len(seen) == 2
    for ys in seen.values():                   # каждый ID остался на своей траектории
        assert max(ys) - min(ys) < 30


def test_removed_reported_after_buffer():
    trk = SimpleTracker(TrackerConfig(backend="simple", track_buffer_s=1.0), fps=10)
    for i in range(5):
        trk.update(det(box_at(100 + i * 4, 200)), frame())
    removed_all = []
    for _ in range(15):                        # buffer = 10 кадров
        _, removed = trk.update(det(), frame())
        removed_all += removed
    assert len(removed_all) == 1


def test_low_conf_rescues_track_but_does_not_spawn():
    cfg = TrackerConfig(backend="simple", conf_high=0.5, conf_low=0.1)
    trk = SimpleTracker(cfg, fps=10)
    for i in range(5):
        trk.update(det(box_at(100 + i * 4, 200), score=0.9), frame())
    # низкоуверенная детекция продолжает существующий трек…
    active, _ = trk.update(det(box_at(124, 200), score=0.2), frame())
    assert len(active) == 1
    # …но сама по себе новый трек не создаёт
    trk2 = SimpleTracker(cfg, fps=10)
    for i in range(5):
        active2, _ = trk2.update(det(box_at(300, 300), score=0.2), frame())
    assert active2 == []
