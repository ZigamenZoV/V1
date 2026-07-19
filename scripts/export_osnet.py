"""Экспорт OSNet x0.25 (torchreid) → ONNX для ReID-леджера (спека §6).

  pip install -e .[train]
  python scripts/export_osnet.py --out models/osnet_x0_25.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="osnet_x0_25")
    ap.add_argument("--weights", default=None,
                    help="custom .pth weights; defaults to torchreid pretrained")
    ap.add_argument("--out", default="models/osnet_x0_25.onnx")
    args = ap.parse_args()

    import torch
    import torchreid

    model = torchreid.models.build_model(args.model, num_classes=1000, pretrained=True)
    if args.weights:
        torchreid.utils.load_pretrained_weights(model, args.weights)
    model.eval()

    dummy = torch.randn(1, 3, 256, 128)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["input"], output_names=["embedding"],
        opset_version=17, dynamic_axes=None,      # статический batch=1: инференс событийный
    )
    print(f"OK: {args.out}")


if __name__ == "__main__":
    main()
