"""
Semi-automated labeling: YOLO draft boxes for front-side field detection.

Draft labels live in test_data/id_cards/draft_labels/ until human review.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import export_id_to_excel as eid

# Front-side classes from Egyptian-ID-Detectr-3/data.yaml (ignore back-only fields).
FRONT_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "address",
        "dob",
        "firstName",
        "front_logo",
        "lastName",
        "nid",
        "photo",
        "serial",
    }
)

REQUIRED_FRONT_FIELDS: frozenset[str] = frozenset(
    {"firstName", "lastName", "address", "nid", "dob"}
)


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    class_name: str
    cx: float
    cy: float
    w: float
    h: float
    conf: float

    def to_label_line(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


def _device() -> str:
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def xyxy_to_xywhn(xyxy: np.ndarray, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    return cx, cy, bw, bh


def detect_front_boxes(
    image_path: Path,
    *,
    field_yolo: object | None = None,
    weights: Path | None = None,
    conf: float = 0.25,
) -> list[YoloBox]:
    """Raw Egyptian-ID-Detectr-3 detections (front classes only)."""
    image_path = image_path.expanduser().resolve()
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = bgr.shape[:2]
    fw = weights or Path("runs/train_id_detectr_hyper/weights/best.pt")
    model = field_yolo if field_yolo is not None else eid.get_yolo(fw)
    id_to_name = eid.load_class_names()
    name_to_id = {v: k for k, v in id_to_name.items()}

    r = model.predict(source=bgr, conf=conf, device=_device(), imgsz=640, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []

    xyxy = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()

    boxes: list[YoloBox] = []
    for i in range(len(cls)):
        c = int(cls[i])
        name = id_to_name.get(c, str(c))
        if name not in FRONT_FIELD_NAMES:
            continue
        cx, cy, bw, bh = xyxy_to_xywhn(xyxy[i], w, h)
        boxes.append(
            YoloBox(
                class_id=c,
                class_name=name,
                cx=float(cx),
                cy=float(cy),
                w=float(bw),
                h=float(bh),
                conf=float(confs[i]),
            )
        )
    # Keep highest-confidence box per class
    best: dict[str, YoloBox] = {}
    for b in boxes:
        prev = best.get(b.class_name)
        if prev is None or b.conf > prev.conf:
            best[b.class_name] = b
    return [best[k] for k in sorted(best, key=lambda n: name_to_id.get(n, 0))]


def write_draft_label_file(boxes: list[YoloBox], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [b.to_label_line() for b in boxes]
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_draft_label_file(label_path: Path, class_names: dict[int, str] | None = None) -> list[YoloBox]:
    class_names = class_names or eid.load_class_names()
    boxes: list[YoloBox] = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cid = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        boxes.append(
            YoloBox(
                class_id=cid,
                class_name=class_names.get(cid, str(cid)),
                cx=cx,
                cy=cy,
                w=w,
                h=h,
                conf=1.0,
            )
        )
    return boxes


def missing_required_detections(boxes: list[YoloBox]) -> list[str]:
    found = {b.class_name for b in boxes}
    return sorted(REQUIRED_FRONT_FIELDS - found)
