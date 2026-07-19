"""Файнтюн RF-DETR Nano на головах (спека §1.3).

Двухэтапно (§12 шаг 3): сначала корпус (CrowdHuman-heads + SCUT-HEAD +
RPEE-Heads), затем СВОИ кадры с целевых камер малым LR — перенос под
ракурс/оптику/свет даёт больше любого тюнинга архитектуры (§10.1).

На 4 GB локально: batch 2–4 + grad accumulation + AMP; реалистичнее —
арендованный GPU/Colab (датасет голов небольшой, Nano обучается быстро).

  python scripts/train_heads.py --dataset datasets/heads_train --epochs 30
  python scripts/train_heads.py --dataset datasets/own_frames \
      --weights runs/heads/checkpoint_best_ema.pth --lr 5e-5 --epochs 10
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="COCO dataset dir (annotations.json + images/), class `head`")
    ap.add_argument("--out", default="runs/heads")
    ap.add_argument("--weights", default=None, help="start from your checkpoint (stage 2)")
    ap.add_argument("--resolution", type=int, default=560)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    from rfdetr import RFDETRNano

    kwargs: dict = {"resolution": args.resolution}
    if args.weights:
        kwargs["pretrain_weights"] = args.weights
    model = RFDETRNano(**kwargs)

    # EMA-веса в rfdetr включены по умолчанию; early stopping — по val mAP@50
    # на СВОИХ кадрах, не на CrowdHuman (§1.3). Валидация после обучения:
    # mAP@50 на held-out + визуальный контроль худших кадров (контровый свет,
    # головные уборы, капюшоны).
    model.train(
        dataset_dir=args.dataset,
        output_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        early_stopping=True,
    )
    print(f"Done. Best (EMA) checkpoint in {args.out}; next: scripts/export_onnx.py")


if __name__ == "__main__":
    main()
