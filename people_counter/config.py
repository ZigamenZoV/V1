"""Конфигурация: app.yaml + config/cameras/*.yaml (по файлу на камеру).

Полигоны зон задаются в координатах кадра инференса (спека §8).
Относительные пути (модели, БД, логи) разрешаются от корня проекта —
запускать сервис из корня либо указывать абсолютные пути.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class SyntheticConfig(BaseModel):
    """Параметры синтетической сцены (mock-режим и e2e-тесты)."""
    seed: int = 7
    enters_per_min: float = 8.0
    exits_per_min: float = 6.0
    passby_per_min: float = 4.0    # проход мимо двери (только зона A)
    peek_per_min: float = 2.0      # «сунул голову» — событие не должно родиться
    speed_px: float = 3.0          # px за кадр
    head_px: int = 18              # сторона бокса головы


class ZonesConfig(BaseModel):
    """Пара полигонов входной группы: A — уличная/тамбурная, B — внутренняя."""
    a: list[tuple[float, float]]
    b: list[tuple[float, float]]

    @field_validator("a", "b")
    @classmethod
    def _min_points(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(v) < 3:
            raise ValueError("zone polygon needs at least 3 points")
        return v


class FsmConfig(BaseModel):
    """Устойчивость счёта (спека §5.1–5.2)."""
    k_frames: int = 3              # K последовательных кадров для фиксации состояния
    cooldown_s: float = 2.0        # пауза после завершённой последовательности трека
    max_gap_frames: int = 8        # допустимый разрыв OUT между стабильными A и B
    min_track_frames: int = 8      # короче — шум, отбрасываем
    min_path_px: float = 40.0      # минимальная длина пути центра


class CameraConfig(BaseModel):
    id: str
    enabled: bool = True
    role: Literal["gate", "overview"] = "gate"
    source: Literal["rtsp", "file", "synthetic"] = "rtsp"
    url: str = ""                  # rtsp://... (через go2rtc/MediaMTX) или путь к файлу
    codec: Literal["h264", "hevc"] = "h264"
    use_nvdec: bool = True
    fps: float = 10.0              # целевая частота декода (8–12 достаточно, §4)
    roi: tuple[int, int, int, int] | None = None   # кроп ROI входной группы из full-HD (x,y,w,h), §10.3
    zones: ZonesConfig | None = None
    fsm: FsmConfig = Field(default_factory=FsmConfig)
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)

    @model_validator(mode="after")
    def _check(self) -> "CameraConfig":
        if self.role == "gate" and self.zones is None and self.source != "synthetic":
            raise ValueError(f"camera {self.id}: gate camera requires zones.a/zones.b")
        if self.source in ("rtsp", "file") and not self.url:
            raise ValueError(f"camera {self.id}: source url is not set")
        return self


class MockDetectorConfig(BaseModel):
    jitter_px: float = 1.0
    dropout: float = 0.05          # доля пропущенных детекций
    false_pos_per_frame: float = 0.02


class DetectorConfig(BaseModel):
    backend: Literal["mock", "onnxruntime", "tensorrt"] = "onnxruntime"
    model: str = "models/rfdetr_head.onnx"
    engine: str = "models/rfdetr_head_fp16.engine"
    input_size: int = 560          # кратно 56 (сетка DINOv2 ViT/14, §1.3)
    batch: int = 0                 # 0 → число gate-камер
    conf_gate: float = 0.30        # порог на гейте ниже: recall важнее (§10.2)
    conf_audit: float = 0.50
    max_det: int = 300
    providers: list[str] = Field(default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    mock: MockDetectorConfig = Field(default_factory=MockDetectorConfig)

    @field_validator("input_size")
    @classmethod
    def _mult56(cls, v: int) -> int:
        if v % 56 != 0:
            raise ValueError("input_size must be a multiple of 56 (560, 616, 672)")
        return v


class TrackerConfig(BaseModel):
    backend: Literal["botsort", "bytetrack", "simple"] = "botsort"
    track_buffer_s: float = 2.0    # ≈ 2×FPS кадров переживания окклюзии (§4)
    match_iou: float = 0.25        # IoU-порог ослаблен: боксы голов мелкие (§4)
    center_dist_w: float = 0.5     # вес центр-дистанции в cost (§4)
    conf_high: float = 0.45        # byte-ассоциация high
    conf_low: float = 0.10         # byte-ассоциация low
    min_hits: int = 2
    with_reid: bool = False        # ReID-ветка BoT-SORT: включать при ID-switch в дверях (§11)
    reid_weights: str = "models/osnet_x0_25_msmt17.pt"


class ReidConfig(BaseModel):
    """ReID-леджер (§6) — корректирующий слой, не первичный счётчик."""
    enabled: bool = True
    model: str = "models/osnet_x0_25.onnx"
    threshold: float = 0.60        # cosine, калибровать на своих данных (0.55–0.65)
    time_half_life_s: float = 1800.0
    time_bonus: float = 0.10       # бонус к score недавно вошедшим
    ttl_s: float = 86400.0         # сутки/рабочий день
    crop_expand_w: float = 2.5     # бокс тела = голова, расширенная вширь…
    crop_expand_h: float = 5.5     # …и вниз
    dump_path: str = "data/ledger.npz"
    dump_interval_s: float = 300.0


class AuditConfig(BaseModel):
    """Аудит occupancy по обзорным камерам (§7)."""
    enabled: bool = False
    backend: Literal["lwcc", "p2pnet", "null"] = "lwcc"
    model_name: str = "DM-Count"   # для lwcc: CSRNet | DM-Count | SFANet
    model_weights: str = "SHA"
    interval_s: float = 60.0       # 30–120 c
    alpha: float = 0.25            # occupancy += α·(audit − occupancy)
    ema: float = 0.5               # EMA по нескольким замерам
    outlier_abs: float = 15.0      # выброс: расходится сильнее → ждём подтверждения
    alert_delta: float = 10.0      # порог алерта рассинхрона


class StoreConfig(BaseModel):
    path: str = "data/people_counter.db"
    occupancy_snapshot_s: float = 60.0


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class PipelineConfig(BaseModel):
    mode: Literal["multiprocess", "single"] = "multiprocess"
    ring_slots: int = 6            # слоты shared-memory кольца кадров
    det_queue: int = 64
    snapshot_interval_s: float = 1.0
    snapshot_width: int = 480
    stats_interval_s: float = 2.0
    stale_restart_s: float = 10.0  # watchdog: нет кадров N c → рестарт capture (§8)


class AppConfig(BaseModel):
    site: str = "site-1"
    log_dir: str = "logs"
    log_level: str = "INFO"
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    reid: ReidConfig = Field(default_factory=ReidConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    cameras_dir: str | None = None
    cameras: list[CameraConfig] = Field(default_factory=list)

    # --- удобные выборки ---
    def enabled_cameras(self) -> list[CameraConfig]:
        return [c for c in self.cameras if c.enabled]

    def gate_cameras(self) -> list[CameraConfig]:
        return [c for c in self.enabled_cameras() if c.role == "gate"]

    def overview_cameras(self) -> list[CameraConfig]:
        return [c for c in self.enabled_cameras() if c.role == "overview"]

    def camera(self, camera_id: str) -> CameraConfig:
        for c in self.cameras:
            if c.id == camera_id:
                return c
        raise KeyError(camera_id)

    def detector_batch(self) -> int:
        return self.detector.batch or max(1, len(self.gate_cameras()))


def load_config(path: str | Path) -> AppConfig:
    """Читает app.yaml и подмешивает камеры из cameras_dir (по файлу на камеру)."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    cameras: list[dict] = list(raw.get("cameras") or [])
    cameras_dir = raw.get("cameras_dir")
    if cameras_dir:
        cdir = Path(cameras_dir)
        if not cdir.is_absolute():
            cdir = path.parent / cdir
        if cdir.is_dir():
            for f in sorted(cdir.glob("*.yaml")):
                cam = yaml.safe_load(f.read_text(encoding="utf-8"))
                if cam:
                    cameras.append(cam)
    raw["cameras"] = cameras

    cfg = AppConfig.model_validate(raw)
    ids = [c.id for c in cfg.cameras]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate camera ids: {ids}")
    return cfg
