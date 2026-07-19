"""Пре/пост-процессинг RF-DETR.

Препроцесс rfdetr: квадратный resize (без letterbox), RGB, ImageNet-нормализация.
Модель NMS-free (§1.4): постпроцессинг — sigmoid + порог уверенности, NMS-плагин
не нужен. Боксы модели — cxcywh, нормированные к [0..1].
"""
from __future__ import annotations

import cv2
import numpy as np

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess_batch(images: list[np.ndarray], size: int, pad_to: int) -> np.ndarray:
    """BGR-кадры → NCHW float32 батч фиксированного размера pad_to
    (статический шейп engine, §2.1); хвост добивается нулями."""
    if len(images) > pad_to:
        raise ValueError(f"batch of {len(images)} frames exceeds static size {pad_to}")
    out = np.zeros((pad_to, 3, size, size), np.float32)
    for i, img in enumerate(images):
        if img.shape[0] != size or img.shape[1] != size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _MEAN) / _STD
        out[i] = rgb.transpose(2, 0, 1)
    return np.ascontiguousarray(out)


def postprocess(dets: np.ndarray, logits: np.ndarray, size: int,
                conf: float, max_det: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(dets [B,Q,4] cxcywh norm, logits [B,Q,C]) → [(boxes xyxy px, scores)] по кадрам."""
    prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
    results: list[tuple[np.ndarray, np.ndarray]] = []
    for b in range(dets.shape[0]):
        scores = prob[b].max(axis=-1)                 # один класс head; max по классам
        keep = scores >= conf
        if keep.sum() > max_det:
            idx = np.argsort(-scores)[:max_det]
            mask = np.zeros_like(keep)
            mask[idx] = True
            keep &= mask
        cxcywh = dets[b][keep].astype(np.float32) * size
        s = scores[keep].astype(np.float32)
        xyxy = np.empty_like(cxcywh)
        xyxy[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2
        xyxy[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2
        xyxy[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2
        xyxy[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2
        np.clip(xyxy, 0, size, out=xyxy)
        results.append((xyxy, s))
    return results


def split_outputs(outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Определяет (dets, logits) по форме: у боксов последняя ось == 4."""
    if len(outputs) < 2:
        raise ValueError(f"expected >=2 model outputs, got {len(outputs)}")
    a, b = outputs[0], outputs[1]
    if a.shape[-1] == 4:
        return a, b
    if b.shape[-1] == 4:
        return b, a
    raise ValueError(f"no box output (last dim 4) found: {[o.shape for o in outputs]}")
