"""OSNet x0.25 (ONNX, CPU) — эмбеддинги тел по событию (спека §6).

Нагрузка — единицы инференсов в минуту, GPU не участвует. Экспорт весов:
python scripts/export_osnet.py (torchreid → ONNX).
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class OsnetEmbedder:
    INPUT_W, INPUT_H = 128, 256

    def __init__(self, model_path: str) -> None:
        import onnxruntime as ort

        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"OSNet ONNX not found: {model_path} (export it: python scripts/export_osnet.py)")
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2          # событийная нагрузка — не отъедаем ядра у трекинга
        self.sess = ort.InferenceSession(model_path, sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        log.info("OsnetEmbedder: %s", model_path)

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        """BGR-кроп тела → L2-нормированный эмбеддинг (float32)."""
        img = cv2.resize(crop_bgr, (self.INPUT_W, self.INPUT_H), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _MEAN) / _STD
        x = rgb.transpose(2, 0, 1)[None]
        (emb,) = self.sess.run(None, {self.input_name: x})
        v = emb.reshape(-1).astype(np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v
