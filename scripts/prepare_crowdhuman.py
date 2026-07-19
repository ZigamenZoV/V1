"""CrowdHuman → COCO c единственным классом `head` (спека §1.2).

Берётся только head-аннотация (поле hbox), ignore-области пропускаются.
Скачайте CrowdHuman (annotation_train.odgt + Images) с crowdhuman.org, затем:

  python scripts/prepare_crowdhuman.py --odgt annotation_train.odgt \
      --images-dir Images --out datasets/heads_train
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def convert(odgt: Path, images_dir: Path, out_dir: Path, copy_images: bool) -> None:
    images, annotations = [], []
    ann_id = 1
    skipped = 0
    with odgt.open(encoding="utf-8") as f:
        for img_id, line in enumerate(f, start=1):
            item = json.loads(line)
            fname = item["ID"] + ".jpg"
            src = images_dir / fname
            if not src.is_file():
                skipped += 1
                continue
            images.append({"id": img_id, "file_name": fname, "width": 0, "height": 0})
            for gt in item.get("gtboxes", []):
                if gt.get("tag") != "person":
                    continue
                extra = gt.get("head_attr") or {}
                if extra.get("ignore") == 1:
                    continue
                hbox = gt.get("hbox")
                if not hbox or hbox[2] <= 0 or hbox[3] <= 0:
                    continue
                annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": 1,
                    "bbox": [float(v) for v in hbox],
                    "area": float(hbox[2] * hbox[3]), "iscrowd": 0,
                })
                ann_id += 1
            if copy_images:
                dst = out_dir / "images" / fname
                if not dst.is_file():
                    shutil.copy2(src, dst)

    coco = {
        "info": {"description": "CrowdHuman heads (hbox only)"},
        "categories": [{"id": 1, "name": "head"}],
        "images": images,
        "annotations": annotations,
    }
    out_json = out_dir / "annotations.json"
    out_json.write_text(json.dumps(coco), encoding="utf-8")
    print(f"OK: {len(images)} images, {len(annotations)} heads -> {out_json}")
    if skipped:
        print(f"  skipped {skipped} records with no image file")
    print("rfdetr reads image sizes itself; fill width/height with a separate "
          "PIL pass if your tooling needs them.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--odgt", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy-images", action="store_true",
                    help="copy images into out/images")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    convert(Path(args.odgt), Path(args.images_dir), out, args.copy_images)


if __name__ == "__main__":
    main()
