"""Пара полигонов входной группы (спека §5.1).

Триггер — центр бокса головы (эквивалент sv.PolygonZone c
triggering_anchors=[CENTER], без зависимости от supervision: тот же
cv2.pointPolygonTest под капотом). Полигоны — в координатах кадра инференса.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import ZonesConfig

ZONE_A = "A"   # уличная/тамбурная сторона
ZONE_B = "B"   # внутренняя сторона
OUT = None


class ZonePair:
    def __init__(self, cfg: ZonesConfig) -> None:
        self._a = np.asarray(cfg.a, np.float32).reshape(-1, 1, 2)
        self._b = np.asarray(cfg.b, np.float32).reshape(-1, 1, 2)

    def locate(self, center: tuple[float, float]) -> str | None:
        """'A' | 'B' | None(OUT) для точки-центра головы. При пересечении
        полигонов приоритет у B (внутренняя сторона)."""
        pt = (float(center[0]), float(center[1]))
        if cv2.pointPolygonTest(self._b, pt, False) >= 0:
            return ZONE_B
        if cv2.pointPolygonTest(self._a, pt, False) >= 0:
            return ZONE_A
        return OUT

    def draw(self, img: np.ndarray) -> np.ndarray:
        """Оверлей зон для снапшотов/калибровки."""
        vis = img.copy()
        overlay = img.copy()
        cv2.fillPoly(overlay, [self._a.astype(np.int32)], (60, 140, 220))
        cv2.fillPoly(overlay, [self._b.astype(np.int32)], (80, 200, 120))
        cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
        cv2.polylines(vis, [self._a.astype(np.int32)], True, (60, 140, 220), 2)
        cv2.polylines(vis, [self._b.astype(np.int32)], True, (80, 200, 120), 2)
        return vis
