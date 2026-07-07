"""
Part 1: Pre-fill draft OCR JSON + YOLO boxes using the current pipeline.

Drafts are marked review_status=needs_review — never treated as verified ground truth.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.ground_truth import GROUND_TRUTH_KEYS, IMAGE_EXTS, empty_ground_truth, resolve_ground_truth_path
from tests.labeling.yolo_boxes import (
    detect_front_boxes,
    missing_required_detections,
    read_draft_label_file,
    write_draft_label_file,
)
from tests.stage_runner import run_front_ocr_row, run_nid_decode_fields

DRAFTS_DIRNAME = "drafts"
DRAFT_LABELS_DIRNAME = "draft_labels"
IMPORT_META_DIRNAME = "import_meta"


def _load_import_source(data_dir: Path, stem: str) -> str:
    meta_path = data_dir / IMPORT_META_DIRNAME / f"{stem}.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        if meta.get("source"):
            return str(meta["source"])
    return ""


def is_verified_ground_truth(gt_path: Path | None) -> bool:
    if not gt_path or not gt_path.is_file():
        return False
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    if data.get("review_status") == "needs_review":
        return False
    tags = data.get("tags") or []
    if "synthetic" in tags:
        return True
    return bool(data.get("national_id") or data.get("full_name"))


def discover_front_images(data_dir: Path) -> list[Path]:
    data_dir = data_dir.expanduser().resolve()
    out: list[Path] = []
    for img in sorted(data_dir.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        if img.stem.endswith("_back"):
            continue
        if DRAFTS_DIRNAME in img.parts or DRAFT_LABELS_DIRNAME in img.parts:
            continue
        out.append(img)
    return out


def ocr_row_to_ground_truth(
    row: dict[str, str],
    *,
    image_name: str,
    source: str = "",
) -> dict[str, Any]:
    gt = empty_ground_truth()
    gt["first_name"] = row.get("first_name", "") or row.get("firstName", "")
    gt["last_name"] = row.get("last_name", "") or row.get("lastName", "")
    gt["full_name"] = row.get("full_name", "") or f"{gt['first_name']} {gt['last_name']}".strip()
    gt["address"] = row.get("address", "")
    gt["national_id"] = row.get("national_id", "") or row.get("nid", "")
    gt["dob"] = row.get("dob", "")
    gt["serial"] = row.get("serial", "")
    nid = gt["national_id"]
    if nid:
        gt.update(
            {
                k: v
                for k, v in run_nid_decode_fields(nid).items()
                if k.startswith("decoded_")
            }
        )
        if not gt["dob"]:
            gt["dob"] = gt.get("decoded_birth_date", "")
    gt["tags"] = ["draft_prefill"]
    gt["notes"] = f"DRAFT pre-fill from pipeline for {image_name}. Human review required."
    gt["review_status"] = "needs_review"
    gt["draft_source"] = "extract_front_v1"
    if source:
        gt["source"] = source
    payload = {k: gt.get(k, "") for k in GROUND_TRUTH_KEYS} | {
        "tags": gt["tags"],
        "notes": gt["notes"],
        "review_status": gt["review_status"],
        "draft_source": gt["draft_source"],
    }
    if source:
        payload["source"] = source
    return payload


def review_priority(missing_det: list[str], missing_ocr: list[str]) -> str:
    """high = likely wrong / incomplete; low = likely OK."""
    if missing_det or len(missing_ocr) >= 2:
        return "high"
    if missing_ocr:
        return "medium"
    return "low"


def prefill_one(
    image_path: Path,
    data_dir: Path,
    *,
    field_yolo: object | None = None,
    easyocr_reader: object | None = None,
    digit_yolo: object | None = None,
    use_existing_draft_labels: bool = True,
    local_engine_select_name: bool = True,
) -> dict[str, Any]:
    data_dir = data_dir.expanduser().resolve()
    stem = image_path.stem
    drafts_dir = data_dir / DRAFTS_DIRNAME
    labels_dir = data_dir / DRAFT_LABELS_DIRNAME

    row = run_front_ocr_row(
        image_path,
        fast_mode=True,
        engine="easyocr",
        serial_charset_restrict=True,
        local_engine_select_name=local_engine_select_name,
        easyocr_reader=easyocr_reader,
        field_yolo=field_yolo,
        digit_yolo=digit_yolo,
    )
    boxes = detect_front_boxes(image_path, field_yolo=field_yolo)
    labels_dir = data_dir / DRAFT_LABELS_DIRNAME
    existing_lbl = labels_dir / f"{stem}.txt"
    if use_existing_draft_labels and existing_lbl.is_file() and existing_lbl.stat().st_size > 0:
        boxes = read_draft_label_file(existing_lbl)
        gt_extra_note = "YOLO boxes from Roboflow import (review if needed)."
    else:
        gt_extra_note = "YOLO boxes from model detection (review required)."
    missing_det = missing_required_detections(boxes)
    source = _load_import_source(data_dir, stem)

    gt = ocr_row_to_ground_truth(row, image_name=image_path.name, source=source)
    gt["notes"] = f"DRAFT pre-fill from pipeline for {image_path.name}. {gt_extra_note}"
    gt["draft_missing_detections"] = missing_det
    gt["draft_label_source"] = "roboflow" if use_existing_draft_labels and existing_lbl.is_file() else "model"
    gt["draft_missing_ocr"] = [
        f
        for f in ("full_name", "address", "national_id", "dob")
        if not str(gt.get(f if f != "full_name" else "full_name", "")).strip()
    ]
    gt["draft_review_priority"] = review_priority(missing_det, gt["draft_missing_ocr"])

    draft_json = drafts_dir / f"{stem}.json"
    draft_json.parent.mkdir(parents=True, exist_ok=True)
    draft_json.write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")

    draft_label = labels_dir / f"{stem}.txt"
    write_draft_label_file(boxes, draft_label)

    return {
        "image": image_path.name,
        "draft_json": str(draft_json),
        "draft_label": str(draft_label),
        "boxes": len(boxes),
        "missing_detections": missing_det,
        "missing_ocr": gt["draft_missing_ocr"],
        "review_priority": gt["draft_review_priority"],
        "source": source or "missing",
    }


def _has_pending_draft(data_dir: Path, image_path: Path) -> bool:
    stem = image_path.stem
    draft = data_dir / DRAFTS_DIRNAME / f"{stem}.json"
    if draft.is_file():
        return True
    gt_path = resolve_ground_truth_path(image_path, data_dir)
    if gt_path and gt_path.is_file():
        data = json.loads(gt_path.read_text(encoding="utf-8"))
        return data.get("review_status") == "needs_review"
    return False


def run_prefill(
    data_dir: Path,
    *,
    force: bool = False,
    skip_drafts: bool = True,
    only_subdir: str | None = None,
    only_stems: set[str] | None = None,
    field_yolo: object | None = None,
    easyocr_reader: object | None = None,
    digit_yolo: object | None = None,
    local_engine_select_name: bool = True,
) -> list[dict[str, Any]]:
    data_dir = data_dir.expanduser().resolve()
    results: list[dict[str, Any]] = []
    for img in discover_front_images(data_dir):
        if only_subdir and only_subdir not in img.parts:
            continue
        if only_stems is not None and img.stem not in only_stems:
            continue
        gt_path = resolve_ground_truth_path(img, data_dir)
        if is_verified_ground_truth(gt_path) and not force:
            continue
        if skip_drafts and not force and _has_pending_draft(data_dir, img):
            continue
        results.append(
            prefill_one(
                img,
                data_dir,
                field_yolo=field_yolo,
                easyocr_reader=easyocr_reader,
                digit_yolo=digit_yolo,
                local_engine_select_name=local_engine_select_name,
            )
        )
    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    if not results:
        print("No draft labels written (all images already have verified JSON).")
        print("  Drop new front photos in test_data/id_cards/ or use --force to re-draft.")
        return
    print(f"\n{'Image':<28} {'Boxes':>5}  Missing detections")
    print("-" * 70)
    for r in results:
        miss = ", ".join(r["missing_detections"]) or "—"
        flag = " *** NEEDS MANUAL BOXING" if r["missing_detections"] else ""
        print(f"{r['image']:<28} {r['boxes']:>5}  {miss}{flag}")
    need_box = sum(1 for r in results if r["missing_detections"])
    print(f"\nDraft JSON → test_data/id_cards/drafts/")
    print(f"Draft YOLO → test_data/id_cards/draft_labels/")
    print(f"Total drafted: {len(results)} | Missing required field boxes: {need_box}")
    print("\nNext: py -m tests.labeling.review_app")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-fill draft labels (pipeline-assisted, needs review)")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "test_data" / "id_cards")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-draft even when verified JSON exists (writes to drafts/, not final JSON)",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Only process images under this subfolder name (e.g. real)",
    )
    parser.add_argument(
        "--re-draft",
        action="store_true",
        help="Re-run OCR on images that already have draft JSON (default: skip existing drafts).",
    )
    parser.add_argument(
        "--no-local-engine-select-name",
        action="store_true",
        help="Disable EasyOCR vs Tesseract(ara) scoring on firstName/lastName (default: enabled).",
    )
    args = parser.parse_args()

    reader = field_yolo = digit_yolo = None
    try:
        import export_id_to_excel as eid
        import torch

        fw = ROOT / "runs" / "train_id_detectr_hyper" / "weights" / "best.pt"
        if fw.is_file():
            field_yolo = eid.get_yolo(fw)
        dw = ROOT / "runs" / "train_arabic_numbers_v2" / "weights" / "best.pt"
        if dw.is_file():
            digit_yolo = eid.get_yolo(dw)
        import easyocr

        reader = easyocr.Reader(["ar", "en"], gpu=torch.cuda.is_available(), verbose=False)
    except Exception as ex:
        print(f"Warning: partial model load ({ex})")

    results = run_prefill(
        args.data_dir,
        force=args.force,
        skip_drafts=not args.re_draft,
        only_subdir=args.only,
        field_yolo=field_yolo,
        easyocr_reader=reader,
        digit_yolo=digit_yolo,
        local_engine_select_name=not args.no_local_engine_select_name,
    )
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
