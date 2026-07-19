"""CLI: python -m people_counter <command>

  run             start the service (production)
  simulate        run on a synthetic scene (mock mode, no cameras/GPU needed)
  check           diagnose environment and config before going live
  bench-detector  measure detector FPS on random frames (spec §2.4: measure on-site)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path


def _cmd_run(args: argparse.Namespace) -> int:
    from .config import load_config
    from .log import setup
    from .pipeline.supervisor import Supervisor

    cfg = load_config(args.config)
    if args.mode:
        cfg.pipeline.mode = args.mode
    setup("main", cfg.log_dir, cfg.log_level)
    sup = Supervisor(cfg, args.config)
    sup.start()          # блокируется до Ctrl+C
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    """Диагностика: конфиг, ffmpeg/NVDEC, GPU, рантаймы, модели, БД."""
    import shutil
    import subprocess

    ok = True

    def item(name: str, good: bool, note: str = "") -> None:
        nonlocal ok
        mark = "[ OK ]" if good else "[FAIL]"
        print(f"  {mark} {name}" + (f" — {note}" if note else ""))
        if not good:
            ok = False

    print("Config:")
    try:
        from .config import load_config
        cfg = load_config(args.config)
        gates = cfg.gate_cameras()
        item("config load", True,
             f"{len(cfg.enabled_cameras())} cameras (gate={len(gates)}, "
             f"overview={len(cfg.overview_cameras())})")
    except Exception as e:
        item("config load", False, str(e))
        return 1

    print("Environment:")
    ff = shutil.which("ffmpeg")
    needs_ffmpeg = any(c.source in ("rtsp", "file") for c in cfg.enabled_cameras())
    item("ffmpeg in PATH", bool(ff) or not needs_ffmpeg,
         ff or ("not found" + ("" if needs_ffmpeg else " (not needed: synthetic only)")))
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        item("NVIDIA GPU", out.returncode == 0, out.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        item("NVIDIA GPU", cfg.detector.backend == "mock", "nvidia-smi unavailable")

    print("Detector:")
    if cfg.detector.backend == "onnxruntime":
        try:
            import onnxruntime as ort
            provs = ort.get_available_providers()
            item("onnxruntime", True, ", ".join(provs))
            item("CUDAExecutionProvider", "CUDAExecutionProvider" in provs,
                 "no CUDA EP — CPU fallback (slow); pip install onnxruntime-gpu")
        except ImportError as e:
            item("onnxruntime", False, str(e))
        item(f"model {cfg.detector.model}", Path(cfg.detector.model).is_file(),
             "export: python scripts/export_onnx.py")
    elif cfg.detector.backend == "tensorrt":
        try:
            import tensorrt
            item("tensorrt", True, tensorrt.__version__)
        except ImportError as e:
            item("tensorrt", False, f"{e} — fallback: detector.backend: onnxruntime")
        item(f"engine {cfg.detector.engine}", Path(cfg.detector.engine).is_file(),
             "build: python scripts/build_engine.py (on this machine!)")
    else:
        item("mock detector", True, "production requires onnxruntime|tensorrt")

    print("Tracker:")
    if cfg.tracker.backend in ("botsort", "bytetrack"):
        try:
            import boxmot
            item("boxmot", True, getattr(boxmot, "__version__", "?"))
        except ImportError:
            item("boxmot", False, "pip install -e .[track], or tracker.backend: simple")
    else:
        item("simple tracker", True)

    print("ReID / audit:")
    if cfg.reid.enabled:
        item(f"OSNet {cfg.reid.model}", Path(cfg.reid.model).is_file(),
             "export: python scripts/export_osnet.py (otherwise events go without embeddings)")
    if cfg.audit.enabled:
        try:
            import lwcc  # noqa: F401
            item("lwcc", True)
        except ImportError:
            item("lwcc", False, "pip install lwcc (pulls torch)")

    print("Cameras:")
    for cam in cfg.enabled_cameras():
        if cam.source == "synthetic":
            item(f"{cam.id} (synthetic)", True)
            continue
        if not ff:
            item(f"{cam.id} ({cam.url})", False, "ffmpeg missing")
            continue
        probe = shutil.which("ffprobe")
        if not probe:
            item(f"{cam.id}", True, "no ffprobe — skipping stream check")
            continue
        try:
            r = subprocess.run(
                [probe, "-v", "error", "-rtsp_transport", "tcp", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_name,width,height", "-of", "csv=p=0",
                 cam.url], capture_output=True, text=True, timeout=15)
            item(f"{cam.id} ({cam.url})", r.returncode == 0,
                 r.stdout.strip() or r.stderr.strip()[:120])
        except subprocess.TimeoutExpired:
            item(f"{cam.id} ({cam.url})", False, "timeout 15s")

    print("Storage:")
    try:
        from .store import EventStore
        st = EventStore(cfg.store.path)
        st.close()
        item(f"SQLite {cfg.store.path}", True)
    except Exception as e:
        item(f"SQLite {cfg.store.path}", False, str(e))

    print("\nResult:", "ready to run" if ok else "issues found (see FAIL)")
    return 0 if ok else 1


def _cmd_bench(args: argparse.Namespace) -> int:
    import numpy as np

    from .config import load_config
    from .detect import make_detector
    from .types import Frame

    cfg = load_config(args.config)
    batch = cfg.detector_batch()
    det = make_detector(cfg.detector, batch)
    size = cfg.detector.input_size
    rng = np.random.default_rng(0)
    frames = [Frame(f"bench-{i}", 0, 0.0, 0.0,
                    rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
              for i in range(batch)]
    print(f"Benchmark: backend={cfg.detector.backend}, input={size}, batch={batch}, "
          f"iters={args.iters}")
    det.infer(frames)  # прогрев
    t0 = time.perf_counter()
    for _ in range(args.iters):
        det.infer(frames)
    dt = time.perf_counter() - t0
    per_batch = dt / args.iters * 1000
    print(f"  {per_batch:.1f} ms/batch -> {args.iters / dt:.1f} ticks/s "
          f"-> {args.iters * batch / dt:.1f} frames/s total")
    print(f"  Target per gate camera is 8-12 FPS (spec §2.4): "
          f"{'sufficient' if args.iters / dt >= 8 else 'NOT enough — see spec §10/§11 (resolution, INT8, YOLO fallback)'}")
    det.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="people-counter", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="start the service")
    p_run.add_argument("--config", default="config/app.yaml")
    p_run.add_argument("--mode", choices=["multiprocess", "single"], default=None,
                       help="override pipeline.mode")
    p_run.set_defaults(fn=_cmd_run)

    p_sim = sub.add_parser("simulate", help="mock mode on a synthetic scene")
    p_sim.add_argument("--config", default="config/mock.yaml")
    p_sim.add_argument("--mode", choices=["multiprocess", "single"], default=None)
    p_sim.set_defaults(fn=_cmd_run)

    p_check = sub.add_parser("check", help="environment diagnostics")
    p_check.add_argument("--config", default="config/app.yaml")
    p_check.set_defaults(fn=_cmd_check)

    p_bench = sub.add_parser("bench-detector", help="measure detector FPS")
    p_bench.add_argument("--config", default="config/app.yaml")
    p_bench.add_argument("--iters", type=int, default=100)
    p_bench.set_defaults(fn=_cmd_bench)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
