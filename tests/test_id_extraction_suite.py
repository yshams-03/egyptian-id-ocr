"""Integration tests — run real extraction against fixtures (slow)."""
from __future__ import annotations

import pytest

from tests.harness import evaluate_sample, run_dataset
from tests.id_metrics import aggregate_field_accuracy


@pytest.mark.slow
@pytest.mark.needs_weights
def test_roboflow_regression_sample(
    roboflow_test_image, roboflow_ground_truth, extraction_models
):
    """Regression on Egyptian-ID-Detectr-3 test image 15-2."""
    models = extraction_models
    result = evaluate_sample(
        roboflow_test_image,
        roboflow_ground_truth,
        fast_mode=True,
        engine="easyocr",
        easyocr_reader=models["reader"],
        field_yolo=models["field_yolo"],
        digit_yolo=models["digit_yolo"],
    )
    assert not result.extraction_error, result.extraction_error
    assert not result.nid_validation_errors, result.nid_validation_errors
    assert result.dob_nid and result.dob_nid.ok

    by_field = {f.field: f for f in result.fields}
    assert by_field["national_id"].exact, (
        f"NID: got {by_field['national_id'].actual}"
    )
    assert by_field["dob"].exact, f"DOB: got {by_field['dob'].actual}"
    assert by_field["name"].cer < 0.05, (
        f"name CER={by_field['name'].cer} actual={by_field['name'].actual!r}"
    )
    assert by_field["address"].cer < 0.55, (
        f"address CER={by_field['address'].cer} actual={by_field['address'].actual!r}"
    )


@pytest.mark.slow
@pytest.mark.needs_weights
def test_local_dataset_if_present(test_data_dir, extraction_models):
    """Run all labeled pairs in test_data/id_cards when ground truth exists."""
    if not test_data_dir.is_dir():
        pytest.skip("test_data/id_cards not created yet")

    models = extraction_models
    results = run_dataset(
        test_data_dir,
        fast_mode=True,
        engine="easyocr",
        require_ground_truth=True,
        easyocr_reader=models["reader"],
        field_yolo=models["field_yolo"],
        digit_yolo=models["digit_yolo"],
    )
    if not results:
        pytest.skip("No image+JSON pairs in test_data/id_cards")

    acc = aggregate_field_accuracy(results)
    assert acc.get("national_id", 0) >= 0.8, f"national_id accuracy {acc}"
    assert sum(1 for r in results if r.passed) / len(results) >= 0.5
