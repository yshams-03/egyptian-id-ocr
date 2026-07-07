"""
Save DOB/serial crop artifacts for worst held-out samples.

Examples:
  py -m tests.debug_crops
  py -m tests.debug_crops --limit 5
  py -m tests.debug_crops --samples real_20-2-...,real_IMG_20221219_...
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

import export_id_to_excel as eid
from tests.harness import run_dataset
from tests.id_metrics import (
    FieldScore,
    SampleResult,
    held_out_results,
    normalize_dob,
    normalize_nid,
    normalize_serial,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_models():
    import easyocr
    import torch

    fw = ROOT / "runs" / "train_id_detectr_hyper" / "weights" / "best.pt"
    dw = ROOT / "runs" / "train_arabic_numbers_v2" / "weights" / "best.pt"
    return (
        easyocr.Reader(["ar", "en"], gpu=torch.cuda.is_available(), verbose=False),
        eid.get_yolo(fw),
        eid.get_yolo(dw) if dw.is_file() else None,
        "0" if torch.cuda.is_available() else "cpu",
    )


def _result_severity(r: SampleResult) -> tuple[float, int]:
    failed = [f for f in r.fields if not f.passed and not f.skipped]
    cer_sum = sum(f.cer for f in failed)
    return (cer_sum, len(failed))


def _choose_samples(
    data_dir: Path,
    *,
    stems: list[str] | None,
    limit: int,
    reader: object,
    field_yolo: object,
    digit_yolo: object | None,
) -> list[SampleResult]:
    results = run_dataset(
        data_dir,
        easyocr_reader=reader,
        field_yolo=field_yolo,
        digit_yolo=digit_yolo,
    )
    held = held_out_results(results)
    if stems:
        wanted = set(stems)
        return [r for r in held if Path(r.image_path).stem in wanted]
    failed = [r for r in held if not r.passed]
    failed.sort(key=_result_severity, reverse=True)
    return failed[:limit]


def _save_image(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _safe_tesseract_ocr(crop_big, *, lang: str) -> str:
    try:
        return eid.ocr_crop(crop_big, lang, 6)
    except Exception:
        return ""


def _easyocr_text(reader: object, crop_big) -> str:
    try:
        rgb = cv2.cvtColor(crop_big, cv2.COLOR_BGR2RGB)
        parts = reader.readtext(rgb, detail=0)
        return " ".join(str(p).strip() for p in parts if str(p).strip())
    except Exception:
        return ""


def _overlay_boxes(img, best: dict[str, tuple[Any, float]]) -> Any:
    out = img.copy()
    colors = {"dob": (0, 200, 255), "serial": (255, 200, 0)}
    for label in ("dob", "serial"):
        if label not in best:
            continue
        box, conf = best[label]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), colors[label], 2)
        cv2.putText(
            out,
            f"{label} {conf:.2f}",
            (x1, max(18, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colors[label],
            2,
            cv2.LINE_AA,
        )
    return out


def _field_map(result: SampleResult) -> dict[str, FieldScore]:
    return {f.field: f for f in result.fields}


def _classify(field: str, expected: str, actual: str, crop, box_found: bool) -> str:
    if not box_found or crop is None or crop.size == 0:
        return "crop_problem"
    h, w = crop.shape[:2]
    if min(h, w) < 18:
        return "crop_problem"
    if field == "dob":
        if normalize_dob(expected) == normalize_dob(actual):
            return "formatting_only"
        if len(normalize_nid(actual)) >= 6:
            return "true_ocr_problem"
        return "preprocessing_problem"
    if field == "serial":
        if normalize_serial(expected) == normalize_serial(actual):
            return "formatting_only"
        if actual.strip():
            return "true_ocr_problem"
        return "preprocessing_problem"
    return "unclear"


def _write_summary(
    sample_dir: Path,
    result: SampleResult,
    gt: dict[str, Any],
    raw_outputs: dict[str, str],
    classes: dict[str, str],
) -> None:
    payload = {
        "sample_stem": Path(result.image_path).stem,
        "image_path": result.image_path,
        "source": result.source,
        "expected_dob": gt.get("dob", ""),
        "actual_dob_ocr": result.actual_row.get("dob", ""),
        "expected_serial": gt.get("serial", ""),
        "actual_serial_ocr": result.actual_row.get("serial", ""),
        "expected_national_id": gt.get("national_id", ""),
        "actual_national_id": result.actual_row.get("national_id", ""),
        "decoded_birth_date": result.actual_row.get("decoded_birth_date", ""),
        "raw_ocr": raw_outputs,
        "normalized": {
            "expected_dob": normalize_dob(gt.get("dob", "")),
            "actual_dob": normalize_dob(result.actual_row.get("dob", "")),
            "expected_serial": normalize_serial(gt.get("serial", "")),
            "actual_serial": normalize_serial(result.actual_row.get("serial", "")),
        },
        "classifications": classes,
    }
    (sample_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _build_report(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Crop Debug Report", ""]
    for row in rows:
        lines.append(f"## `{row['stem']}`")
        lines.append(f"- source: `{row['source']}`")
        lines.append(f"- expected DOB: `{row['expected_dob']}`")
        lines.append(f"- actual DOB OCR: `{row['actual_dob']}`")
        lines.append(f"- expected serial: `{row['expected_serial']}`")
        lines.append(f"- actual serial OCR: `{row['actual_serial']}`")
        lines.append(f"- DOB diagnosis: `{row['dob_class']}`")
        lines.append(f"- serial diagnosis: `{row['serial_class']}`")
        lines.append(f"- note: {row['note']}")
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Save DOB/serial crops for held-out failures")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "test_data" / "id_cards")
    parser.add_argument("--samples", default="", help="Comma-separated sample stems")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    reader, field_yolo, digit_yolo, device = _load_models()
    stems = [s.strip() for s in args.samples.split(",") if s.strip()]
    chosen = _choose_samples(
        args.data_dir,
        stems=stems or None,
        limit=args.limit,
        reader=reader,
        field_yolo=field_yolo,
        digit_yolo=digit_yolo,
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "runs" / "debug_crops" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict[str, Any]] = []

    id_to_name = eid.load_class_names()
    for result in chosen:
        img_path = Path(result.image_path)
        gt = json.loads(img_path.with_suffix(".json").read_text(encoding="utf-8"))
        sample_dir = out_dir / img_path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)

        bgr = cv2.imread(str(img_path))
        pred = field_yolo.predict(source=bgr, conf=0.25, device=device, imgsz=640, verbose=False)[0]
        xyxy = pred.boxes.xyxy.cpu().numpy() if pred.boxes is not None and len(pred.boxes) else []
        cls = pred.boxes.cls.cpu().numpy().astype(int) if pred.boxes is not None and len(pred.boxes) else []
        conf = pred.boxes.conf.cpu().numpy() if pred.boxes is not None and len(pred.boxes) else []
        best, _ = eid.best_boxes_with_invalid_fallback(xyxy, cls, conf, id_to_name) if len(cls) else ({}, False)

        raw_outputs: dict[str, str] = {}
        classes: dict[str, str] = {}
        overlay = _overlay_boxes(bgr, best)
        _save_image(sample_dir / "front.jpg", bgr)
        _save_image(sample_dir / "overlay.jpg", overlay)

        for field in ("dob", "serial"):
            crop = None
            crop_big = None
            crop_prep = None
            actual = result.actual_row.get(field, "")
            expected = gt.get(field, "")
            if field in best:
                crop = eid.crop_xyxy(bgr, best[field][0], pad=6)
                if crop.size:
                    crop_big = eid.upscale_crop(crop, min_side=200)
                    crop_prep = eid.preprocess_for_ocr(crop_big)
                    _save_image(sample_dir / f"{field}_crop_raw.jpg", crop)
                    _save_image(sample_dir / f"{field}_crop_preprocessed.jpg", crop_prep)
                    tess = _safe_tesseract_ocr(crop_big, lang="eng")
                    eo = _easyocr_text(reader, crop_big)
                    raw_outputs[field] = tess or eo or actual
                    if tess:
                        raw_outputs[f"{field}_tesseract"] = tess
                    if eo:
                        raw_outputs[f"{field}_easyocr"] = eo
                else:
                    raw_outputs[field] = ""
            else:
                raw_outputs[field] = ""
            classes[field] = _classify(field, expected, actual, crop_big, field in best)

        note = (
            "DOB likely needs crop/preprocessing work first."
            if classes["dob"] in {"crop_problem", "preprocessing_problem"}
            else "DOB crop exists; OCR is still weak on this strip."
        )
        report_rows.append(
            {
                "stem": img_path.stem,
                "source": result.source,
                "expected_dob": gt.get("dob", ""),
                "actual_dob": result.actual_row.get("dob", ""),
                "expected_serial": gt.get("serial", ""),
                "actual_serial": result.actual_row.get("serial", ""),
                "dob_class": classes["dob"],
                "serial_class": classes["serial"],
                "note": note,
            }
        )
        _write_summary(sample_dir, result, gt, raw_outputs, classes)

    _build_report(out_dir, report_rows)
    print(f"Wrote crop debug artifacts to: {out_dir}")
    print("Open report.md and sample folders to inspect DOB/serial framing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
