"""Фабрика детектора. Интерфейс `Detector.infer(frames) -> [Detections]`
абстрагирован (спека §2.2): рантайм меняется конфигом, без правок конвейера."""
from __future__ import annotations

from ..config import DetectorConfig
from ..types import Detector


def make_detector(cfg: DetectorConfig, batch: int) -> Detector:
    if cfg.backend == "mock":
        from .mock import MockDetector
        return MockDetector(cfg)
    if cfg.backend == "onnxruntime":
        from .onnx_rt import OnnxDetector
        return OnnxDetector(cfg, batch)
    if cfg.backend == "tensorrt":
        from .trt import TrtDetector
        return TrtDetector(cfg, batch)
    raise ValueError(f"unknown detector backend: {cfg.backend}")
