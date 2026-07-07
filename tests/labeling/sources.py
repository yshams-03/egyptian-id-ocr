"""
Source tags for ground truth — prevents train/eval leakage tracking.
"""
from __future__ import annotations

# Required on every ground-truth JSON (use explicit value; never omit).
SOURCE_SYNTHETIC = "synthetic_generated"
SOURCE_ROBOFLOW_TRAIN = "roboflow_train"
SOURCE_ROBOFLOW_VALID = "roboflow_valid"
SOURCE_ROBOFLOW_TEST = "roboflow_test"
SOURCE_MANUAL = "manual"

ROBOFLOW_SOURCES: frozenset[str] = frozenset(
    {SOURCE_ROBOFLOW_TRAIN, SOURCE_ROBOFLOW_VALID, SOURCE_ROBOFLOW_TEST}
)

# Held-out for OCR accuracy — must never enter YOLO train split via promotion.
HELD_OUT_SOURCES: frozenset[str] = frozenset({SOURCE_ROBOFLOW_VALID, SOURCE_ROBOFLOW_TEST})

VALID_SOURCES: frozenset[str] = frozenset(
    {
        SOURCE_SYNTHETIC,
        SOURCE_ROBOFLOW_TRAIN,
        SOURCE_ROBOFLOW_VALID,
        SOURCE_ROBOFLOW_TEST,
        SOURCE_MANUAL,
    }
)


def split_name_to_source(split: str) -> str:
    s = split.strip().lower()
    if s == "valid":
        return SOURCE_ROBOFLOW_VALID
    if s == "test":
        return SOURCE_ROBOFLOW_TEST
    if s == "train":
        return SOURCE_ROBOFLOW_TRAIN
    raise ValueError(f"Unknown Roboflow split: {split}")


def normalize_source(value: str) -> str:
    v = (value or "").strip()
    if v in VALID_SOURCES:
        return v
    raise ValueError(
        f"Invalid or missing source {value!r}. "
        f"Required one of: {', '.join(sorted(VALID_SOURCES))}"
    )
