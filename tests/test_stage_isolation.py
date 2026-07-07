"""Per-stage isolation tests (slow — real YOLO/OCR)."""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from tests.harness import evaluate_sample
from tests.id_metrics import score_fields
from tests.stage_runner import run_front_ocr_row, run_nid_decode_fields


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
@pytest.mark.needs_weights
def test_stage_nid_decode(roboflow_ground_truth):
    dec = run_nid_decode_fields(roboflow_ground_truth["national_id"])
    assert dec["decoded_birth_date"] == roboflow_ground_truth["dob"]
    assert dec["decoded_governorate"] == "Ash Sharqia"
    assert dec["decoded_gender"] == "Male"


@pytest.mark.slow
@pytest.mark.needs_weights
def test_stage_front_ocr_name_address(
    roboflow_test_image, roboflow_ground_truth, extraction_models
):
    models = extraction_models
    row = run_front_ocr_row(
        roboflow_test_image,
        fast_mode=True,
        engine="easyocr",
        easyocr_reader=models["reader"],
        field_yolo=models["field_yolo"],
        digit_yolo=models["digit_yolo"],
    )
    scores = score_fields(roboflow_ground_truth, row)
    by = {s.field: s for s in scores}
    assert by["national_id"].exact
    assert by["name"].cer < 0.1


@pytest.mark.slow
@pytest.mark.needs_weights
def test_pipeline_end_to_end(roboflow_test_image, roboflow_ground_truth, extraction_models):
    models = extraction_models
    r = evaluate_sample(
        roboflow_test_image,
        roboflow_ground_truth,
        fast_mode=True,
        engine="easyocr",
        easyocr_reader=models["reader"],
        field_yolo=models["field_yolo"],
        digit_yolo=models["digit_yolo"],
    )
    assert r.actual_row
    assert not r.nid_validation_errors
    by = {f.field: f for f in r.fields}
    assert by["national_id"].exact
    assert r.stages.field_detection


@pytest.mark.slow
@pytest.mark.needs_weights
def test_serial_charset_restrict_does_not_change_other_front_fields(extraction_models):
    sample = (
        ROOT
        / "test_data"
        / "id_cards"
        / "real"
        / "real_1-2-_jpg.rf.f3ecdaf3289346fc43a2bbb90e2a0da9.jpg"
    )
    gt_path = sample.with_suffix(".json")
    if not sample.is_file() or not gt_path.is_file():
        pytest.skip("Regression sample not available in local real-card test set.")

    models = extraction_models
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
    baseline = run_front_ocr_row(
        sample,
        fast_mode=True,
        engine="easyocr",
        serial_charset_restrict=False,
        easyocr_reader=models["reader"],
        field_yolo=models["field_yolo"],
        digit_yolo=models["digit_yolo"],
    )
    restricted = run_front_ocr_row(
        sample,
        fast_mode=True,
        engine="easyocr",
        serial_charset_restrict=True,
        easyocr_reader=models["reader"],
        field_yolo=models["field_yolo"],
        digit_yolo=models["digit_yolo"],
    )

    for field in ("first_name", "last_name", "address", "dob"):
        assert restricted[field] == baseline[field], field

    assert restricted["serial"] == ground_truth["serial"]


@pytest.mark.slow
@pytest.mark.needs_weights
def test_unbatch_text_fields_preserves_dob_serial_batch_output(extraction_models):
    """Name/address use individual EasyOCR; dob/serial batch strip geometry must be unchanged."""
    import cv2

    import export_id_to_excel as eid
    import extract_name_address as ena

    sample = (
        ROOT
        / "test_data"
        / "id_cards"
        / "real"
        / "real_1-2-_jpg.rf.f3ecdaf3289346fc43a2bbb90e2a0da9.jpg"
    )
    if not sample.is_file():
        pytest.skip("Regression sample not available in local real-card test set.")

    models = extraction_models
    reader = models["reader"]
    field_yolo = models["field_yolo"]
    min_side = 120
    pad = 6

    img = cv2.imread(str(sample))
    assert img is not None
    img = eid.resize_for_speed(img, max_side=880)
    r = field_yolo.predict(source=img, conf=0.25, device="0", imgsz=480, verbose=False)[0]
    assert r.boxes is not None and len(r.boxes) > 0
    best = eid.best_boxes_by_label(
        r.boxes.xyxy.cpu().numpy(),
        r.boxes.cls.cpu().numpy().astype(int),
        r.boxes.conf.cpu().numpy(),
        eid.load_class_names(),
    )

    def _upscaled_crop(lab: str):
        cr = eid.crop_xyxy(img, best[lab][0], pad)
        up = eid.upscale_crop(cr, min_side=min_side)
        return eid.resize_for_speed(up, max_side=480)

    # Pre-address-unbatch: name spacers + address/dob/serial in strip.
    name_spacers = [_upscaled_crop(lab) for lab in ("firstName", "lastName") if lab in best]
    with_address_batch = ena.ocr_fields_batch_easyocr(
        [
            (lab, eid.crop_xyxy(img, best[lab][0], pad))
            for lab in ("address", "dob", "serial")
            if lab in best
        ],
        reader,
        min_side=min_side,
        leading_spacers=name_spacers or None,
    )

    # Current: name + address spacers + dob/serial only.
    all_spacers = [
        _upscaled_crop(lab) for lab in ("firstName", "lastName", "address") if lab in best
    ]
    dob_serial_batch = ena.ocr_fields_batch_easyocr(
        [
            (lab, eid.crop_xyxy(img, best[lab][0], pad))
            for lab in ("dob", "serial")
            if lab in best
        ],
        reader,
        min_side=min_side,
        leading_spacers=all_spacers or None,
    )

    for field in ("dob", "serial"):
        assert dob_serial_batch.get(field, "") == with_address_batch.get(field, ""), field

    row = run_front_ocr_row(
        sample,
        fast_mode=True,
        engine="easyocr",
        easyocr_reader=reader,
        field_yolo=field_yolo,
        digit_yolo=models["digit_yolo"],
    )
    assert row["serial"] == eid.merge_serial_ocr("", with_address_batch.get("serial") or None)
