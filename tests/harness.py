"""
Scan test_data/id_cards, run extract_id_all end-to-end, score against ground truth.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from tests.ground_truth import (
    GROUND_TRUTH_KEYS,
    discover_test_cases,
    generate_template_json,
    load_ground_truth,
    prefill_from_national_id,
    resolve_back_image,
)
from tests.id_metrics import SampleResult, StageScores, score_fields
from tests.nid_validate import build_dob_nid_cross_check, validate_extracted_nid
from tests.labeling.sources import SOURCE_SYNTHETIC
from tests.stage_runner import (
    detection_labels_detected,
    run_back_row,
    run_front_ocr_row,
    run_nid_decode_fields,
    run_pipeline_row,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def discover_image_pairs(data_dir: Path) -> list[tuple[Path, Path | None]]:
    """Backward-compatible: (image, ground_truth_json|None)."""
    return [(case["front"], case["ground_truth_path"]) for case in discover_test_cases(data_dir)]


def generate_template_csv(data_dir: Path, out_csv: Path | None = None) -> Path:
    data_dir = data_dir.expanduser().resolve()
    out_csv = out_csv or (data_dir / "ground_truth.template.csv")
    cols = [
        "image_filename",
        "national_id",
        "first_name",
        "last_name",
        "full_name",
        "address",
        "dob",
        "serial",
        "job",
        "religion",
        "expiry_date",
        "back_nid",
        "back_image",
        "tags",
        "notes",
    ]
    rows = []
    for case in discover_test_cases(data_dir):
        img = case["front"]
        gt = case["ground_truth"]
        row = {c: "" for c in cols}
        row["image_filename"] = img.name
        row["back_image"] = (gt.get("back_image") or "") or (
            case["back"].name if case.get("back") else ""
        )
        if case["ground_truth_path"]:
            for k in cols:
                if k in gt and k != "image_filename":
                    v = gt[k]
                    row[k] = ",".join(v) if k == "tags" and isinstance(v, list) else str(v or "")
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return out_csv


def generate_missing_templates(data_dir: Path) -> list[Path]:
    """Create <stem>.json templates for images without ground truth."""
    written: list[Path] = []
    for case in discover_test_cases(data_dir):
        if case["ground_truth_path"]:
            continue
        out = generate_template_json(case["front"])
        written.append(out)
    return written


def csv_to_json_ground_truth(csv_path: Path, data_dir: Path) -> int:
    data_dir = data_dir.expanduser().resolve()
    written = 0
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("image_filename") or "").strip()
            if not name:
                continue
            gt_dir = data_dir / "ground_truth"
            gt_dir.mkdir(parents=True, exist_ok=True)
            out = gt_dir / f"{Path(name).stem}.json"
            tags_raw = (row.get("tags") or "").strip()
            nid = (row.get("national_id") or "").strip()
            payload: dict[str, Any] = {k: (row.get(k) or "").strip() for k in GROUND_TRUTH_KEYS if k not in ("tags", "notes")}
            payload["tags"] = [t.strip() for t in tags_raw.split(",") if t.strip()]
            payload["notes"] = (row.get("notes") or "").strip()
            if nid:
                pref = prefill_from_national_id(nid)
                payload.update(pref)
            if not any(payload.get(k) for k in ("full_name", "address", "national_id", "dob")):
                continue
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            written += 1
    return written


def _score_stages(
    expected: dict[str, Any],
    actual: dict[str, str],
    *,
    front: Path,
    back: Path | None,
    field_yolo: object | None,
    easyocr_reader: object | None,
    fast_mode: bool,
    engine: str,
    serial_charset_restrict: bool,
) -> StageScores:
    required = {"firstName", "lastName", "address", "nid"}
    detected = detection_labels_detected(front, field_yolo=field_yolo)
    missing = sorted(required - detected)
    field_ok = not missing or actual.get("nid_decode_error") != "no detections"

    front_row = run_front_ocr_row(
        front,
        fast_mode=fast_mode,
        engine=engine,
        serial_charset_restrict=serial_charset_restrict,
        easyocr_reader=easyocr_reader,
        field_yolo=field_yolo,
    )
    ocr_scores = score_fields(
        {k: expected.get(k, "") for k in ("full_name", "address", "national_id", "dob", "serial")},
        front_row,
    )
    front_ok = all(s.passed for s in ocr_scores if not s.skipped)

    nid_ok = True
    if (expected.get("decoded_birth_date") or expected.get("decoded_governorate") or "").strip():
        act_dec = run_nid_decode_fields(actual.get("national_id", ""))
        for k in ("decoded_birth_date", "decoded_governorate", "decoded_gender"):
            exp_v = (expected.get(k) or "").strip()
            if exp_v and act_dec.get(k, "") != exp_v and actual.get(k, "") != exp_v:
                nid_ok = False

    back_ok = True
    if back and any((expected.get(k) or "").strip() for k in ("job", "religion", "back_nid", "expiry_date")):
        if easyocr_reader is None:
            back_ok = False
        else:
            back_row = run_back_row(
                back,
                front_nid=actual.get("national_id", ""),
                easyocr_reader=easyocr_reader,
                fast=fast_mode,
            )
            back_scores = score_fields(expected, back_row)
            back_ok = all(s.passed for s in back_scores if s.field in ("job", "religion", "back_nid", "expiry_date") and not s.skipped)

    return StageScores(
        field_detection=field_ok,
        front_ocr=front_ok,
        nid_decode=nid_ok,
        back_extraction=back_ok,
        missing_detection_labels=missing,
    )


def evaluate_sample(
    image_path: Path,
    ground_truth: dict[str, Any],
    *,
    back_image: Path | None = None,
    fast_mode: bool = True,
    engine: str = "easyocr",
    serial_charset_restrict: bool = True,
    easyocr_reader: object | None = None,
    field_yolo: object | None = None,
    digit_yolo: object | None = None,
    auto_card_crop: bool = False,
) -> SampleResult:
    tags = list(ground_truth.get("tags") or [])
    source = str(ground_truth.get("source", "") or "")
    if not source and "synthetic" in tags:
        source = SOURCE_SYNTHETIC
    t0 = time.perf_counter()
    back = back_image or resolve_back_image(image_path, ground_truth)
    try:
        actual = run_pipeline_row(
            image_path,
            back=back,
            fast_mode=fast_mode,
            engine=engine,
            auto_card_crop=auto_card_crop,
            serial_charset_restrict=serial_charset_restrict,
            easyocr_reader=easyocr_reader,
            field_yolo=field_yolo,
            digit_yolo=digit_yolo,
        )
    except Exception as e:
        return SampleResult(
            image_path=str(image_path),
            ground_truth_path="",
            extraction_error=str(e),
            tags=tags,
            duration_s=time.perf_counter() - t0,
        )

    fields = score_fields(ground_truth, actual)
    nid_val = validate_extracted_nid(actual.get("national_id", ""))
    dob_nid = build_dob_nid_cross_check(actual)
    stages = _score_stages(
        ground_truth,
        actual,
        front=image_path,
        back=back,
        field_yolo=field_yolo,
        easyocr_reader=easyocr_reader,
        fast_mode=fast_mode,
        engine=engine,
        serial_charset_restrict=serial_charset_restrict,
    )

    return SampleResult(
        image_path=str(image_path),
        ground_truth_path="",
        fields=fields,
        nid_validation_errors=nid_val.errors if not nid_val.ok else [],
        dob_nid=dob_nid,
        tags=tags,
        source=source,
        duration_s=time.perf_counter() - t0,
        actual_row=dict(actual),
        stages=stages,
    )


def run_dataset(
    data_dir: Path,
    *,
    fast_mode: bool = True,
    engine: str = "easyocr",
    require_ground_truth: bool = True,
    auto_card_crop: bool = False,
    serial_charset_restrict: bool = True,
    easyocr_reader: object | None = None,
    field_yolo: object | None = None,
    digit_yolo: object | None = None,
) -> list[SampleResult]:
    results: list[SampleResult] = []
    for case in discover_test_cases(data_dir):
        if require_ground_truth and not case["ground_truth_path"]:
            continue
        gt = case["ground_truth"]
        r = evaluate_sample(
            case["front"],
            gt,
            back_image=case.get("back"),
            fast_mode=fast_mode,
            engine=engine,
            auto_card_crop=auto_card_crop,
            serial_charset_restrict=serial_charset_restrict,
            easyocr_reader=easyocr_reader,
            field_yolo=field_yolo,
            digit_yolo=digit_yolo,
        )
        r.ground_truth_path = str(case["ground_truth_path"] or "")
        results.append(r)
    return results
