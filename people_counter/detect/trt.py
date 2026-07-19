"""Основной рантайм: TensorRT FP16, статический шейп (спека §2.1).

Engine собирается НА ЦЕЛЕВОЙ машине (планы непереносимы между GPU/версиями):
    python scripts/build_engine.py --onnx models/rfdetr_head.onnx --batch 2
FP16 на TU117 даёт ×~2 к FP32 за счёт отдельных FP16-юнитов Turing,
Tensor Cores не требуются.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..config import DetectorConfig
from ..types import Detections, Frame
from .preprocess import postprocess, preprocess_batch, split_outputs

log = logging.getLogger(__name__)


class TrtDetector:
    def __init__(self, cfg: DetectorConfig, batch: int) -> None:
        import tensorrt as trt
        from cuda import cudart

        if not Path(cfg.engine).is_file():
            raise FileNotFoundError(
                f"TensorRT engine not found: {cfg.engine} "
                f"(build it: python scripts/build_engine.py)")
        self.cfg = cfg
        self.batch = batch
        self.input_size = cfg.input_size
        self._cudart = cudart

        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(cfg.engine, "rb") as f, trt.Runtime(trt_logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()

        # Буферы: имя тензора → (host ndarray, device ptr)
        self._io: dict[str, tuple[np.ndarray, int]] = {}
        self._input_name: str | None = None
        self._output_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host = np.zeros(shape, dtype=dtype)
            err, dptr = cudart.cudaMalloc(host.nbytes)
            _check(err)
            self._io[name] = (host, dptr)
            self.ctx.set_tensor_address(name, dptr)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_name = name
                if shape[0] != batch:
                    log.warning("engine batch=%d differs from config batch=%d — using engine",
                                shape[0], batch)
                self.batch = shape[0]
            else:
                self._output_names.append(name)
        err, self._stream = cudart.cudaStreamCreate()
        _check(err)
        log.info("TrtDetector: %s, batch=%d, input=%s", cfg.engine, self.batch, self._input_name)

    def infer(self, frames: list[Frame]) -> list[Detections]:
        cudart = self._cudart
        assert self._input_name
        host_in, dev_in = self._io[self._input_name]
        x = preprocess_batch([f.image for f in frames], self.input_size, self.batch)
        np.copyto(host_in, x.astype(host_in.dtype, copy=False))
        _check(cudart.cudaMemcpyAsync(
            dev_in, host_in.ctypes.data, host_in.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self._stream)[0])
        self.ctx.execute_async_v3(self._stream)
        outs = []
        for name in self._output_names:
            host, dev = self._io[name]
            _check(cudart.cudaMemcpyAsync(
                host.ctypes.data, dev, host.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self._stream)[0])
            outs.append(host)
        _check(cudart.cudaStreamSynchronize(self._stream)[0])

        dets, logits = split_outputs([o.astype(np.float32) for o in outs])
        per_frame = postprocess(dets, logits, self.input_size, self.cfg.conf_gate, self.cfg.max_det)
        return [Detections(b, s) for (b, s) in per_frame[: len(frames)]]

    def close(self) -> None:
        cudart = self._cudart
        for _, (_, dptr) in self._io.items():
            cudart.cudaFree(dptr)
        cudart.cudaStreamDestroy(self._stream)


def _check(err) -> None:
    """cuda-python возвращает cudaError_t (иногда первым элементом кортежа)."""
    if isinstance(err, tuple):
        err = err[0]
    if int(err) != 0:
        raise RuntimeError(f"CUDA error: {err}")
