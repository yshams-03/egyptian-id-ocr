"""
Read national ID digits using the already-trained Arabic-digit YOLO model
(classes 0–9), same family of runs as val_batch0_pred.jpg from validation.

No new training — only runs inference on your image.

Default weights: runs/train_arabic_numbers_v2/weights/best.pt

Examples:
  py extract_nid_digits.py "C:\\path\\to\\id_or_nid_crop.jpg"
  py extract_nid_digits.py full_id.jpg --nid-field-weights runs\\train_id_detectr_hyper\\weights\\best.pt
  py extract_nid_digits.py id.jpg --dedupe-iou 0.5 --conf 0.2 --save runs\\nid_digits_overlay.jpg
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
DEFAULT_DIGIT_WEIGHTS = BASE / "runs" / "train_arabic_numbers_v2" / "weights" / "best.pt"


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def dedupe_detections(
    xyxy: np.ndarray, cls: np.ndarray, conf: np.ndarray, iou_thresh: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy NMS by confidence: drop digit boxes that heavily overlap a higher-conf box."""
    if len(xyxy) == 0 or iou_thresh <= 0:
        return xyxy, cls, conf
    order = np.argsort(-conf)
    keep: list[int] = []
    for i in order:
        i = int(i)
        ok = True
        for j in keep:
            if iou_xyxy(xyxy[i], xyxy[j]) >= iou_thresh:
                ok = False
                break
        if ok:
            keep.append(i)
    keep_arr = np.array(keep, dtype=int)
    return xyxy[keep_arr], cls[keep_arr], conf[keep_arr]


def crop_nid_field(
    bgr: np.ndarray,
    field_weights: Path,
    conf: float,
    pad: int,
    device: str,
) -> np.ndarray:
    """Crop best `nid` box from Egyptian-ID field detector (existing model, no training)."""
    from ultralytics import YOLO

    model = YOLO(str(field_weights))
    r = model.predict(source=bgr, conf=conf, device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return bgr
    names = model.names
    nid_cls = None
    if isinstance(names, dict):
        for k, v in names.items():
            if str(v) == "nid":
                nid_cls = int(k)
                break
    if nid_cls is None:
        return bgr
    best_i: int | None = None
    best_c = -1.0
    for i in range(len(r.boxes)):
        if int(r.boxes.cls[i]) != nid_cls:
            continue
        c = float(r.boxes.conf[i])
        if c > best_c:
            best_c, best_i = c, i
    if best_i is None:
        return bgr
    xyxy = r.boxes.xyxy[best_i].cpu().numpy()
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return bgr
    return bgr[y1:y2, x1:x2].copy()


def reading_order_indices(xyxy: np.ndarray, mode: str) -> np.ndarray:
    """Order digit boxes for concatenation.

    - ``ltr``: left-to-right (best for a **single horizontal** NID strip / nid crop).
    - ``row_col``: top-to-bottom, then left-to-right (better for **full card** with several digit lines).
    - ``auto``: if vertical spread of box centers is small vs digit height, use ``ltr``, else ``row_col``.
    """
    xc = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    yc = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
    heights = np.maximum(1.0, xyxy[:, 3] - xyxy[:, 1])
    median_h = float(np.median(heights))
    spread_y = float(np.max(yc) - np.min(yc)) if len(yc) else 0.0
    one_line = spread_y < 0.55 * median_h

    if mode == "auto":
        mode = "ltr" if one_line else "row_col"
    if mode == "ltr":
        # numpy lexsort: last key is primary → xc is primary (left to right)
        return np.lexsort((yc, xc))
    if mode == "row_col":
        return np.lexsort((xc, yc))
    raise ValueError(f"Unknown reading order: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate digit detections (0–9) into a national-ID style string."
    )
    parser.add_argument("image", type=Path, help="Image path (quote if spaces). Prefer a tight crop of the ID number line.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_DIGIT_WEIGHTS,
        help="Arabic-numbers YOLO .pt (10 classes: digits).",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save annotated preview (same as val_batch*_pred style).",
    )
    parser.add_argument(
        "--nid-field-weights",
        type=Path,
        default=None,
        help="Field detector .pt (e.g. train_id_detectr_hyper). When set, crops `nid` first, then digit model runs on that crop only.",
    )
    parser.add_argument(
        "--field-conf",
        type=float,
        default=0.25,
        help="Confidence for nid-field detector when --nid-field-weights is used.",
    )
    parser.add_argument("--nid-pad", type=int, default=8, help="Pixels to pad nid crop.")
    parser.add_argument(
        "--dedupe-iou",
        type=float,
        default=0.45,
        help="Merge overlapping digit boxes (same digit detected twice). Use 0 to disable.",
    )
    parser.add_argument(
        "--reading-order",
        choices=("auto", "ltr", "row_col"),
        default="auto",
        help="How to sort digit boxes: auto (default), left-to-right (ltr), or row then column (row_col).",
    )
    parser.add_argument(
        "--compare",
        default=None,
        metavar="DIGITS",
        help="Optional known 14-digit NID to print character match rate after inference (debug).",
    )
    args = parser.parse_args()

    img = args.image.expanduser().resolve()
    if not img.is_file():
        raise SystemExit(f"Image not found: {img}")

    wpath = args.weights.expanduser().resolve()
    if not wpath.is_file():
        raise SystemExit(f"Weights not found: {wpath}\nTrain with run_egyptian_id_ocr.py or pass --weights.")

    import cv2
    import torch
    from ultralytics import YOLO

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"

    bgr = cv2.imread(str(img))
    if bgr is None:
        raise SystemExit(f"Could not read image: {img}")

    source: str | np.ndarray = str(img)
    if args.nid_field_weights is not None:
        fw = args.nid_field_weights.expanduser().resolve()
        if not fw.is_file():
            raise SystemExit(f"--nid-field-weights not found: {fw}")
        crop = crop_nid_field(bgr, fw, args.field_conf, args.nid_pad, device)
        source = crop
        print("Using nid-field crop for digit model (size {}x{}).".format(crop.shape[1], crop.shape[0]))

    model = YOLO(str(wpath))
    results = model.predict(
        source=source,
        conf=args.conf,
        device=device,
        imgsz=args.imgsz,
        verbose=False,
    )
    r = results[0]

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save), r.plot())
        print(f"Saved overlay: {args.save.resolve()}")

    if r.boxes is None or len(r.boxes) == 0:
        print("national_id_digits:")
        print("(no digit detections — use a crop of the number line, lower --conf, or check weights)")
        return

    xyxy = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()
    if args.dedupe_iou > 0:
        before = len(xyxy)
        xyxy, cls, conf = dedupe_detections(xyxy, cls, conf, args.dedupe_iou)
        if len(xyxy) < before:
            print(f"Deduped digit boxes: {before} -> {len(xyxy)} (iou>={args.dedupe_iou})")
    order = reading_order_indices(xyxy, args.reading_order)

    names = model.names
    chars: list[str] = []
    for idx in order:
        c = int(cls[idx])
        label = names.get(c, names[c]) if isinstance(names, dict) else str(c)
        chars.append(str(label))
        print(f"  digit {label!r} conf={float(conf[idx]):.3f}")

    nid = "".join(chars)
    print()
    print("national_id_digits:", nid)
    if args.compare:
        ref = re.sub(r"\D", "", args.compare)
        if len(ref) == 14 and len(nid) == 14:
            matches = sum(1 for a, b in zip(nid, ref) if a == b)
            print(f"Compare to given NID: {matches}/14 characters match (position-wise).")
        else:
            print("Compare: need both predicted and --compare strings to be 14 digits after stripping.")
    if len(nid) == 14:
        print("(14 digits — length matches Egyptian NID format; verify digits in --save overlay.)")
    elif args.nid_field_weights is None and len(chars) > 10:
        print(
            "Tip: many digits on a full card often include date/serial. "
            "Re-run with:\n"
            "  --nid-field-weights runs\\train_id_detectr_hyper\\weights\\best.pt"
        )


if __name__ == "__main__":
    main()
