"""Счётчики толпы для аудита occupancy (спека §7).

Прямой пересчёт по обзорным камерам раз в 30–120 с, вне горячего пути.
Быстрый старт — LWCC (CSRNet/DM-Count/SFANet, готовые веса). Целевой
вариант — P2PNet (point-based, по MAE лучше DM-Count): веса тренируются/
берутся отдельно, подключается тем же интерфейсом.
"""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from ..config import AuditConfig

log = logging.getLogger(__name__)


class AuditBackend(Protocol):
    name: str

    def count(self, frame_bgr: np.ndarray) -> float: ...


class NullBackend:
    """Аудит выключен/недоступен."""
    name = "null"

    def count(self, frame_bgr: np.ndarray) -> float:
        raise RuntimeError("null audit backend does not count")


class LwccBackend:
    """LWCC: pip install lwcc (тянет torch). Веса скачиваются при первом вызове."""
    name = "lwcc"

    def __init__(self, cfg: AuditConfig) -> None:
        from lwcc import LWCC  # noqa: F401 — проверка импорта на старте
        self._lwcc = LWCC
        self.cfg = cfg
        log.info("LwccBackend: %s / %s", cfg.model_name, cfg.model_weights)

    def count(self, frame_bgr: np.ndarray) -> float:
        # LWCC принимает пути к файлам; отдаём кадр через временный файл —
        # аудит не realtime, накладные расходы несущественны.
        import tempfile
        from pathlib import Path

        import cv2

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "audit.jpg"
            cv2.imwrite(str(p), frame_bgr)
            n = self._lwcc.get_count(str(p), model_name=self.cfg.model_name,
                                     model_weights=self.cfg.model_weights)
        return float(n)


class P2PNetBackend:
    """P2PNet: предсказывает точки-головы напрямую. Подключение весов —
    отдельная задача (репозиторий TencentYoutuResearch/CrowdCounting-P2PNet);
    для дальних мелких голов допустим SAHI-тайлинг поверх count()."""
    name = "p2pnet"

    def __init__(self, cfg: AuditConfig) -> None:
        raise NotImplementedError(
            "P2PNet backend is not wired up: add weights and inference code to this "
            "class (interface: count(frame)->float), or use audit.backend: lwcc")

    def count(self, frame_bgr: np.ndarray) -> float:  # pragma: no cover
        raise NotImplementedError


def make_audit_backend(cfg: AuditConfig) -> AuditBackend | None:
    """None → аудит отключён (сервис работает без него)."""
    if not cfg.enabled or cfg.backend == "null":
        return None
    try:
        if cfg.backend == "lwcc":
            return LwccBackend(cfg)
        if cfg.backend == "p2pnet":
            return P2PNetBackend(cfg)
    except (ImportError, NotImplementedError) as e:
        log.warning("audit disabled: %s", e)
        return None
    raise ValueError(f"unknown audit backend: {cfg.backend}")
