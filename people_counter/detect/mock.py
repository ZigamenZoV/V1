"""Mock-детектор: возвращает ground-truth боксы синтетического источника
с шумом (джиттер, dropout, ложные срабатывания). Нужен для e2e-проверки
трекинга/FSM/событий без модели и GPU."""
from __future__ import annotations

import numpy as np

from ..config import DetectorConfig
from ..types import Detections, Frame


class MockDetector:
    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self.input_size = cfg.input_size
        self._rng = np.random.default_rng(1234)

    def infer(self, frames: list[Frame]) -> list[Detections]:
        m = self.cfg.mock
        out: list[Detections] = []
        for f in frames:
            if f.gt_boxes is None or len(f.gt_boxes) == 0:
                boxes = np.zeros((0, 4), np.float32)
            else:
                keep = self._rng.random(len(f.gt_boxes)) >= m.dropout
                boxes = f.gt_boxes[keep].copy()
                boxes += self._rng.normal(0, m.jitter_px, boxes.shape).astype(np.float32)
            scores = np.clip(self._rng.normal(0.82, 0.08, len(boxes)), 0.3, 0.99).astype(np.float32)
            if self._rng.random() < m.false_pos_per_frame:
                s = self.input_size
                x, y = self._rng.uniform(0, s - 20, 2)
                fp = np.array([[x, y, x + 16, y + 16]], np.float32)
                boxes = np.vstack([boxes, fp]) if len(boxes) else fp
                scores = np.append(scores, np.float32(self._rng.uniform(0.3, 0.5)))
            out.append(Detections(boxes.astype(np.float32), scores))
        return out

    def close(self) -> None:
        pass
