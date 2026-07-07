#!/usr/bin/env python
"""A/B name OCR preprocessing presets on held-out + roboflow_train."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import export_id_to_excel as eid
import extract_name_address as ena
from tests.ground_truth import discover_test_cases
from tests.id_metrics import cer, exact_match
from tests.labeling.sources import HELD_OUT_SOURCES, SOURCE_ROBOFLOW_TRAIN


def _expected_name(gt: dict) -> str:
    return (
        (gt.get("full_name") or "").strip()
        or (gt.get("name") or "").strip()
        or f"{gt.get('first_name', '')} {gt.get('last_name', '')}".strip()
    )


def _clahe_bgr(bgr, clip: float = 2.5):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)


def _sharpen_bgr(bgr):
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.0)
    return cv2.addWeighted(bgr, 1.5, blur, -0.5, 0)


def _prep(bgr, mode: str, min_side: int, max_side: int):
    big = eid.upscale_crop(bgr, min_side=min_side)
    big = eid.resize_for_speed(big, max_side=max_side)
    if mode == "raw":
        return big
    if mode == "clahe":
        return _clahe_bgr(big)
    if mode == "sharpen":
        return _sharpen_bgr(big)
    if mode == "clahe_sharpen":
        return _sharpen_bgr(_clahe_bgr(big))
    if mode == "otsu":
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
    return big


def _easyocr_read(reader, bgr) -> str:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    parts = reader.readtext(rgb, paragraph=True)
    texts = []
    for item in parts:
        text = item[1] if len(item) > 1 else ""
        conf = float(item[2]) if len(item) > 2 else 1.0
        if conf >= 0.05 and text.strip():
            texts.append(text.strip())
    return " ".join(texts).strip()


def ocr_name(bgr, reader, *, mode: str, min_side: int, max_side: int) -> str:
    if bgr.size == 0:
        return ""
    if mode == "multi":
        big = eid.upscale_crop(bgr, min_side=min_side)
        big = eid.resize_for_speed(big, max_side=max_side)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        imgs = [big, _clahe_bgr(big), _sharpen_bgr(big), _sharpen_bgr(_clahe_bgr(big))]
        for prep in ena._field_preprocess_variants(gray):
            imgs.append(cv2.cvtColor(prep, cv2.COLOR_GRAY2BGR))
        cands = []
        for im in imgs:
            t = _easyocr_read(reader, im)
            if t:
                cands.append(t)
        return max(cands, key=lambda s: ena._score_reading(s, True)) if cands else ""
    return _easyocr_read(reader, _prep(bgr, mode, min_side, max_side))


@dataclass(frozen=True)
class Preset:
    label: str
    mode: str = "raw"
    min_side: int = 120
    max_side: int = 520


PRESETS: list[Preset] = [
    Preset("baseline", "raw", 120, 520),
    Preset("up180", "raw", 180, 520),
    Preset("up240", "raw", 240, 520),
    Preset("up240_wide", "raw", 240, 880),
    Preset("clahe120", "clahe", 120, 520),
    Preset("clahe180", "clahe", 180, 520),
    Preset("clahe240", "clahe", 240, 720),
    Preset("clahe_sharp240", "clahe_sharpen", 240, 720),
    Preset("otsu240", "otsu", 240, 720),
    Preset("multi240", "multi", 240, 720),
]


def eval_preset(preset: Preset, cases, reader, field_yolo, device: str) -> dict:
    held_pass = held_n = train_pass = train_n = 0
    held_cers: list[float] = []
    train_cers: list[float] = []

    for case in cases:
        gt = case["ground_truth"]
        source = str(gt.get("source") or "")
        exp = _expected_name(gt)
        if not exp:
            continue
        img = cv2.imread(str(case["front"]))
        if img is None:
            continue
        img = eid.resize_for_speed(img, max_side=880)
        r = field_yolo.predict(source=img, conf=0.25, device=device, imgsz=480, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        best = eid.best_boxes_by_label(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int),
            r.boxes.conf.cpu().numpy(),
            eid.load_class_names(),
        )
        first = last = ""
        if "firstName" in best:
            cr = eid.crop_xyxy(img, best["firstName"][0], 6)
            first = ocr_name(cr, reader, mode=preset.mode, min_side=preset.min_side, max_side=preset.max_side)
        if "lastName" in best:
            cr = eid.crop_xyxy(img, best["lastName"][0], 6)
            last = ocr_name(cr, reader, mode=preset.mode, min_side=preset.min_side, max_side=preset.max_side)
        act = f"{first} {last}".strip()
        c = cer(exp, act)
        ok = exact_match(exp, act, field="name") or c <= 0.15
        if source in HELD_OUT_SOURCES:
            held_n += 1
            held_cers.append(c)
            held_pass += int(ok)
        elif source == SOURCE_ROBOFLOW_TRAIN:
            train_n += 1
            train_cers.append(c)
            train_pass += int(ok)

    return {
        "held_pass": held_pass,
        "held_n": held_n,
        "held_pct": 100 * held_pass / held_n if held_n else 0,
        "held_cer": sum(held_cers) / len(held_cers) if held_cers else 0,
        "train_pass": train_pass,
        "train_n": train_n,
        "train_pct": 100 * train_pass / train_n if train_n else 0,
        "train_cer": sum(train_cers) / len(train_cers) if train_cers else 0,
    }


def main() -> int:
    device = "0"
    try:
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"

    import easyocr
    from ultralytics import YOLO

    reader = easyocr.Reader(["ar", "en"], gpu=device != "cpu", verbose=False)
    field_yolo = YOLO(str(ROOT / "runs" / "train_id_detectr_hyper" / "weights" / "best.pt"))
    cases = [c for c in discover_test_cases(ROOT / "test_data" / "id_cards") if _expected_name(c["ground_truth"])]

    print("| Preset | Held-out pass | Held CER | Train pass | Train CER |")
    print("|--------|---------------|----------|------------|-----------|")
    results = []
    for p in PRESETS:
        m = eval_preset(p, cases, reader, field_yolo, device)
        results.append((p.label, m))
        print(
            f"| {p.label} | {m['held_pass']}/{m['held_n']} ({m['held_pct']:.1f}%) | "
            f"{m['held_cer']:.3f} | {m['train_pass']}/{m['train_n']} ({m['train_pct']:.1f}%) | "
            f"{m['train_cer']:.3f} |"
        )

    best = max(results, key=lambda x: (x[1]["held_pass"], -x[1]["held_cer"], x[1]["train_pass"]))
    print(f"\nBest held-out: {best[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
