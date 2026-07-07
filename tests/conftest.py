"""Pytest configuration and shared fixtures."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "test_data" / "id_cards"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FIELD_WEIGHTS = ROOT / "runs" / "train_id_detectr_hyper" / "weights" / "best.pt"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: full OCR + YOLO extraction (seconds per image)")
    config.addinivalue_line("markers", "needs_weights: requires trained YOLO weights")


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    return Path(os.environ.get("ID_TEST_DATA_DIR", DEFAULT_DATA_DIR))


@pytest.fixture(scope="session")
def weights_available() -> bool:
    return FIELD_WEIGHTS.is_file()


@pytest.fixture(scope="session")
def extraction_models(weights_available):
    """Cache YOLO + EasyOCR for the test session (slow once)."""
    if not weights_available:
        pytest.skip("Field YOLO weights not found")
    import export_id_to_excel as eid

    field_yolo = eid.get_yolo(FIELD_WEIGHTS)
    digit_w = ROOT / "runs" / "train_arabic_numbers_v2" / "weights" / "best.pt"
    digit_yolo = eid.get_yolo(digit_w) if digit_w.is_file() else None
    reader = None
    try:
        import easyocr
        import torch

        reader = easyocr.Reader(
            ["ar", "en"], gpu=torch.cuda.is_available(), verbose=False
        )
    except Exception:
        pass
    return {"field_yolo": field_yolo, "digit_yolo": digit_yolo, "reader": reader}


@pytest.fixture
def roboflow_test_image(project_root) -> Path:
    """Public dataset test image shipped with the repo."""
    p = (
        project_root
        / "egyptian_id_detectr"
        / "content"
        / "Egyptian-ID-Detectr-3"
        / "test"
        / "images"
        / "15-2-_jpg.rf.9e7d02144c3ccbc294a9c6ee9c6bbeb7.jpg"
    )
    if not p.is_file():
        pytest.skip(f"Roboflow test image missing: {p}")
    return p


@pytest.fixture
def roboflow_ground_truth() -> dict:
    import json

    p = FIXTURES_DIR / "ground_truth" / "15-2-_jpg.rf.9e7d02144c3ccbc294a9c6ee9c6bbeb7.json"
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR
