"""Сборка TensorRT engine: ONNX → FP16, статический шейп (спека §2.1).

Запускать ТОЛЬКО на целевой машине: планы TensorRT непереносимы между
GPU и версиями. Батч фиксируется = числу gate-камер (§2.1). INT8 (DP4A) —
только если FP16 не вытянул целевой FPS, порядок: FP16 → разрешение → INT8 (§2.3).

  python scripts/build_engine.py --onnx models/rfdetr_head.onnx --batch 2
  python scripts/build_engine.py ... --int8 --calib-dir data/calib_frames
"""
from __future__ import annotations

import argparse
from pathlib import Path


def build(onnx_path: str, engine_path: str, batch: int, size: int,
          fp16: bool, int8: bool, calib_dir: str | None, workspace_gb: float) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    data = Path(onnx_path).read_bytes()
    if not parser.parse(data):
        for i in range(parser.num_errors):
            print("  parse error:", parser.get_error(i))
        raise SystemExit("failed to parse ONNX")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 int(workspace_gb * (1 << 30)))
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        if not calib_dir:
            raise SystemExit("--int8 requires --calib-dir "
                             "(500-1000 frames from the target cameras, spec §2.3)")
        config.int8_calibrator = _make_calibrator(calib_dir, batch, size)

    # статический шейп: фиксируем вход (динамика на TU117 не нужна и мешает, §2.1)
    inp = network.get_input(0)
    inp.shape = (batch, 3, size, size)

    print(f"Building: {onnx_path} -> {engine_path} "
          f"(batch={batch}, {size}x{size}, fp16={fp16}, int8={int8})...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise SystemExit("engine build failed (see TensorRT log above); "
                         "fallback: detector.backend: onnxruntime (spec §2.2)")
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    Path(engine_path).write_bytes(engine_bytes)
    print(f"OK: {engine_path} ({len(engine_bytes) / 1e6:.1f} MB)")
    print("Speed check: python -m people_counter bench-detector --config config/app.yaml")


def _make_calibrator(calib_dir: str, batch: int, size: int):
    """INT8-энтропийный калибратор по кадрам с целевых камер."""
    import numpy as np
    import tensorrt as trt
    from cuda import cudart

    import cv2

    files = sorted(Path(calib_dir).glob("*.jpg")) + sorted(Path(calib_dir).glob("*.png"))
    if len(files) < 100:
        raise SystemExit(f"not enough calibration frames: {len(files)} (< 100)")

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from people_counter.detect.preprocess import preprocess_batch

    class Calib(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            super().__init__()
            self.i = 0
            nbytes = batch * 3 * size * size * 4
            err, self.dptr = cudart.cudaMalloc(nbytes)
            self.cache = Path(calib_dir) / "calib.cache"

        def get_batch_size(self) -> int:
            return batch

        def get_batch(self, names):
            if self.i + batch > len(files):
                return None
            imgs = [cv2.imread(str(f)) for f in files[self.i:self.i + batch]]
            self.i += batch
            x = preprocess_batch(imgs, size, batch)
            cudart.cudaMemcpy(self.dptr, x.ctypes.data, x.nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            return [int(self.dptr)]

        def read_calibration_cache(self):
            return self.cache.read_bytes() if self.cache.is_file() else None

        def write_calibration_cache(self, cache) -> None:
            self.cache.write_bytes(cache)

    return Calib()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="models/rfdetr_head.onnx")
    ap.add_argument("--engine", default="models/rfdetr_head_fp16.engine")
    ap.add_argument("--batch", type=int, required=True, help="= number of gate cameras")
    ap.add_argument("--size", type=int, default=560)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib-dir", default=None)
    ap.add_argument("--workspace-gb", type=float, default=1.5)
    args = ap.parse_args()
    build(args.onnx, args.engine, args.batch, args.size,
          not args.no_fp16, args.int8, args.calib_dir, args.workspace_gb)


if __name__ == "__main__":
    main()
