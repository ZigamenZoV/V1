"""Разметка зон A/B по снапшоту камеры (спека §5.1, §8).

Берёт кадр с работающего сервиса (/cameras/<id>/snapshot.jpg) или из файла,
даёт накликать два полигона и печатает YAML-блок zones для конфига камеры.
Снапшот сервиса уже в координатах кадра инференса — то, что нужно.

  python scripts/calibrate_zones.py --camera gate-1 --api http://127.0.0.1:8000
  python scripts/calibrate_zones.py --image frame.png

Управление: ЛКМ — точка; Enter — завершить полигон (сначала A, потом B);
Backspace — убрать точку; Esc — выход без результата.
"""
from __future__ import annotations

import argparse
import urllib.request

import cv2
import numpy as np

COLORS = {"A": (60, 140, 220), "B": (80, 200, 120)}


def grab(args: argparse.Namespace) -> np.ndarray:
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"cannot read: {args.image}")
        return img
    url = f"{args.api}/cameras/{args.camera}/snapshot.jpg"
    with urllib.request.urlopen(url, timeout=10) as r:
        data = np.frombuffer(r.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def draw_state(base: np.ndarray, polys: dict[str, list], cur: list, cur_name: str) -> np.ndarray:
    img = base.copy()
    for name, pts in polys.items():
        if len(pts) >= 3:
            cv2.polylines(img, [np.array(pts, np.int32)], True, COLORS[name], 2)
            cv2.putText(img, name, tuple(np.mean(pts, axis=0).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS[name], 2)
    for i, p in enumerate(cur):
        cv2.circle(img, tuple(p), 3, COLORS[cur_name], -1)
        if i:
            cv2.line(img, tuple(cur[i - 1]), tuple(p), COLORS[cur_name], 1)
    # Hershey-шрифты OpenCV — только ASCII
    label = f"Polygon {cur_name} ({'street' if cur_name == 'A' else 'inner'} side) | " \
            f"LMB point, Enter finish, Backspace undo, Esc quit"
    cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="gate-1")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--image", default=None, help="frame file instead of the API snapshot")
    ap.add_argument("--scale", type=float, default=None,
                    help="coordinate multiplier (if the frame is not at inference resolution)")
    args = ap.parse_args()

    base = grab(args)
    polys: dict[str, list] = {}
    cur: list = []
    order = ["A", "B"]
    step = 0

    def on_mouse(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and step < len(order):
            cur.append([x, y])

    win = f"zones: {args.camera}"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        name = order[step] if step < len(order) else order[-1]
        cv2.imshow(win, draw_state(base, polys, cur, name))
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            raise SystemExit("cancelled")
        if key == 8 and cur:
            cur.pop()
        if key == 13:
            if len(cur) < 3:
                continue
            polys[order[step]] = list(cur)
            cur.clear()
            step += 1
            if step == len(order):
                break
    cv2.destroyAllWindows()

    k = args.scale or 1.0
    print("\n# paste into config/cameras/<camera>.yaml:")
    print("zones:")
    for name in order:
        print(f"  {'a' if name == 'A' else 'b'}:")
        for x, y in polys[name]:
            print(f"    - [{int(x * k)}, {int(y * k)}]")


if __name__ == "__main__":
    main()
