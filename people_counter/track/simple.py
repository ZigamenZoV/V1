"""Встроенный трекер: byte-подобная двухступенчатая жадная ассоциация.

Матчинг по IoU + центр-дистанции (спека §4: чистый IoU на мелких боксах
голов работает хуже — добавлена нормированная дистанция центров).
Прогноз — постоянная скорость (EMA). Без torch и внешних зависимостей.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import TrackerConfig
from ..types import Detections, Frame, Track


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / max(area, 1e-9))


@dataclass
class _T:
    tid: int
    box: np.ndarray
    score: float
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    hits: int = 1
    misses: int = 0
    confirmed: bool = False

    def predict(self) -> np.ndarray:
        shift = np.tile(self.vel, 2)
        return self.box + shift

    def correct(self, box: np.ndarray, score: float) -> None:
        new_c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], np.float32)
        old_c = np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2], np.float32)
        self.vel = 0.6 * self.vel + 0.4 * (new_c - old_c)
        self.box = box.astype(np.float32)
        self.score = score
        self.hits += 1
        self.misses = 0


class SimpleTracker:
    def __init__(self, cfg: TrackerConfig, fps: float) -> None:
        self.cfg = cfg
        self.buffer_frames = max(1, int(cfg.track_buffer_s * fps))
        self._tracks: list[_T] = []
        self._next_id = 1

    def _match(self, tracks: list[_T], det_boxes: np.ndarray, det_scores: np.ndarray,
               strict: bool) -> tuple[dict[int, int], set[int]]:
        """Жадная ассоциация. Возвращает (idx трека → idx детекции, использованные det)."""
        cfg = self.cfg
        pairs: list[tuple[float, int, int]] = []
        for ti, t in enumerate(tracks):
            pred = t.predict()
            psize = max((pred[2] - pred[0] + pred[3] - pred[1]) / 2.0, 4.0)
            pc = np.array([(pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2])
            for di in range(len(det_boxes)):
                b = det_boxes[di]
                iou = _iou(pred, b)
                dc = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                dist_units = float(np.hypot(*(pc - dc))) / psize   # в размерах головы
                gate_dist = 1.2 if strict else 2.0
                if iou < cfg.match_iou and dist_units > gate_dist:
                    continue
                cost = (1.0 - iou) + cfg.center_dist_w * min(dist_units, 3.0)
                pairs.append((cost, ti, di))
        pairs.sort(key=lambda p: p[0])
        m: dict[int, int] = {}
        used_d: set[int] = set()
        for _, ti, di in pairs:
            if ti in m or di in used_d:
                continue
            m[ti] = di
            used_d.add(di)
        return m, used_d

    def update(self, det: Detections, frame: Frame) -> tuple[list[Track], list[int]]:
        cfg = self.cfg
        hi = det.scores >= cfg.conf_high
        lo = (det.scores >= cfg.conf_low) & ~hi
        hi_boxes, hi_scores = det.boxes[hi], det.scores[hi]
        lo_boxes, lo_scores = det.boxes[lo], det.scores[lo]

        # ступень 1: все треки × high-conf детекции
        m1, used1 = self._match(self._tracks, hi_boxes, hi_scores, strict=False)
        matched: set[int] = set()
        for ti, di in m1.items():
            self._tracks[ti].correct(hi_boxes[di], float(hi_scores[di]))
            matched.add(ti)

        # ступень 2: оставшиеся треки × low-conf (byte): спасает окклюзии в дверях
        rest = [i for i in range(len(self._tracks)) if i not in matched]
        rest_tracks = [self._tracks[i] for i in rest]
        m2, _ = self._match(rest_tracks, lo_boxes, lo_scores, strict=True)
        for ri, di in m2.items():
            self._tracks[rest[ri]].correct(lo_boxes[di], float(lo_scores[di]))
            matched.add(rest[ri])

        # новые треки из несматченных high-conf
        for di in range(len(hi_boxes)):
            if di not in used1:
                self._tracks.append(_T(self._next_id, hi_boxes[di].astype(np.float32),
                                       float(hi_scores[di])))
                self._next_id += 1

        # промахи, подтверждение, чистка
        removed: list[int] = []
        alive: list[_T] = []
        for i, t in enumerate(self._tracks):
            if i in matched:
                if not t.confirmed and t.hits >= cfg.min_hits:
                    t.confirmed = True
            else:
                t.misses += 1
                t.box = t.predict()          # коастинг по скорости
            # неподтверждённый трек (вероятный шум) умирает быстро
            limit = self.buffer_frames if t.confirmed else min(3, self.buffer_frames)
            if t.misses > limit:
                if t.confirmed:
                    removed.append(t.tid)    # окончательно потерян → финализация
            else:
                alive.append(t)
        self._tracks = alive

        active = [Track(t.tid, t.box.copy(), t.score)
                  for t in self._tracks if t.confirmed and t.misses == 0]
        return active, removed
