"""Экспорт RF-DETR Nano → ONNX (спека §1.4).

Модель NMS-free: постпроцессинг сводится к порогу уверенности, TensorRT-граф
без NMS-плагина. Запускать там, где стоит [train]-окружение (можно на
арендованном GPU, ONNX переносим — в отличие от TensorRT-engine).

  python scripts/export_onnx.py --weights runs/heads/checkpoint_best_ema.pth \
      --resolution 560 --out models/rfdetr_head.onnx
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None,
                    help="head finetune checkpoint (.pth); without it COCO weights "
                         "are used (person class as a stopgap, spec §12 step 2)")
    ap.add_argument("--resolution", type=int, default=560, help="multiple of 56")
    ap.add_argument("--out", default="models/rfdetr_head.onnx")
    args = ap.parse_args()

    from rfdetr import RFDETRNano

    kwargs: dict = {"resolution": args.resolution}
    if args.weights:
        kwargs["pretrain_weights"] = args.weights
    model = RFDETRNano(**kwargs)

    out_dir = Path("_export_tmp")
    model.export(output_dir=str(out_dir))          # rfdetr пишет inference_model.onnx

    exported = next(out_dir.glob("*.onnx"), None)
    if exported is None:
        raise SystemExit(f"rfdetr did not produce an .onnx in {out_dir}")
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), dst)
    shutil.rmtree(out_dir, ignore_errors=True)
    print(f"OK: {dst} (input {args.resolution}x{args.resolution})")
    print("Next: python scripts/build_engine.py — strictly on the target machine (spec §2.1)")


if __name__ == "__main__":
    main()
