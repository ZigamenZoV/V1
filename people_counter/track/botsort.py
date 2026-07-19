"""Обёртка boxmot: BoT-SORT (базовый выбор, лучший IDF1) / ByteTrack (fallback
при CPU-узком месте, §11). Работает на CPU.

Камеры статичные → GMC (компенсация движения камеры) отключается — это
основной CPU-налог BoT-SORT и здесь он бесполезен (§4). ReID-ветка
(osnet_x0_25) выключена по умолчанию; включать конфигом при ID-switch
в дверях.
"""
from __future__ import annotations

import inspect
import logging
from pathlib import Path

import numpy as np

from ..config import TrackerConfig
from ..types import Detections, Frame, Track

log = logging.getLogger(__name__)


class _NullCMC:
    """Заглушка компенсации движения камеры: тождественное преобразование."""

    def apply(self, img, dets=None):
        return np.eye(2, 3, dtype=np.float32)


def _filter_kwargs(cls, kwargs: dict) -> dict:
    """boxmot меняет сигнатуры между версиями — передаём только известные."""
    sig = inspect.signature(cls.__init__)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


class BoxmotTracker:
    def __init__(self, cfg: TrackerConfig, fps: float) -> None:
        try:
            import boxmot
        except ImportError as e:
            raise ImportError(
                "boxmot is not installed: `pip install -e .[track]` (pulls torch), "
                "or set tracker.backend: simple in the config") from e

        self.cfg = cfg
        self.buffer_frames = max(1, int(cfg.track_buffer_s * fps))
        common = dict(
            reid_weights=Path(cfg.reid_weights),
            device="cpu",
            half=False,
            with_reid=cfg.with_reid,
            track_buffer=self.buffer_frames,
            track_high_thresh=cfg.conf_high,
            track_low_thresh=cfg.conf_low,
            new_track_thresh=cfg.conf_high,
            match_thresh=0.85,
            frame_rate=int(round(fps)),
            cmc_method="ecc",           # будет заменён на _NullCMC ниже
        )
        if cfg.backend == "botsort":
            cls = boxmot.BotSort if hasattr(boxmot, "BotSort") else boxmot.BoTSORT
        else:
            cls = boxmot.ByteTrack
        self._trk = cls(**_filter_kwargs(cls, common))
        # статичные камеры: глушим GMC независимо от версии boxmot
        gmc_attrs = [a for a in ("cmc", "gmc") if hasattr(self._trk, a)]
        for attr in gmc_attrs:
            setattr(self._trk, attr, _NullCMC())
        # отслеживание потери треков: boxmot не сообщает об удалении
        self._last_seen: dict[int, int] = {}
        self._frame_idx = 0
        log.info("BoxmotTracker: %s, buffer=%d frames, reid=%s, gmc=%s",
                 cls.__name__, self.buffer_frames, cfg.with_reid,
                 "off" if gmc_attrs else "n/a")

    def update(self, det: Detections, frame: Frame) -> tuple[list[Track], list[int]]:
        self._frame_idx += 1
        n = len(det)
        arr = np.zeros((n, 6), np.float32)
        if n:
            arr[:, :4] = det.boxes
            arr[:, 4] = det.scores
            arr[:, 5] = 0.0                      # класс head
        out = self._trk.update(arr, frame.image)  # M x (x1,y1,x2,y2,id,conf,cls,ind)

        active: list[Track] = []
        if out is not None and len(out):
            for row in np.asarray(out):
                tid = int(row[4])
                active.append(Track(tid, row[:4].astype(np.float32), float(row[5])))
                self._last_seen[tid] = self._frame_idx

        removed = [tid for tid, seen in self._last_seen.items()
                   if self._frame_idx - seen > self.buffer_frames]
        for tid in removed:
            del self._last_seen[tid]
        return active, removed
