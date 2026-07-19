"""Резервный рантайм: ONNX Runtime (спека §2.2).

CUDA Execution Provider, fallback на CPU. Ожидаемо на 20–40% медленнее
TensorRT-engine, но zero-friction деплой на Windows.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import DetectorConfig
from ..types import Detections, Frame
from .preprocess import postprocess, preprocess_batch, split_outputs

log = logging.getLogger(__name__)


class OnnxDetector:
    def __init__(self, cfg: DetectorConfig, batch: int) -> None:
        import onnxruntime as ort

        if not Path(cfg.model).is_file():
            raise FileNotFoundError(
                f"ONNX model not found: {cfg.model} "
                f"(export it: python scripts/export_onnx.py)")
        self.cfg = cfg
        self.batch = batch
        self.input_size = cfg.input_size
        avail = ort.get_available_providers()
        providers = [p for p in cfg.providers if p in avail] or ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(cfg.model, sess_options=so, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        log.info("OnnxDetector: %s, providers=%s", cfg.model, self.sess.get_providers())

    def infer(self, frames: list[Frame]) -> list[Detections]:
        x = preprocess_batch([f.image for f in frames], self.input_size, max(self.batch, len(frames)))
        outputs = self.sess.run(None, {self.input_name: x})
        dets, logits = split_outputs(list(outputs))
        per_frame = postprocess(dets, logits, self.input_size, self.cfg.conf_gate, self.cfg.max_det)
        return [Detections(b, s) for (b, s) in per_frame[: len(frames)]]

    def close(self) -> None:
        del self.sess
