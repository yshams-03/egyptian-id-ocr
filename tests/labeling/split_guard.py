"""
Train/valid assignment guardrails — held-out Roboflow sources never go to train.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tests.labeling.sources import HELD_OUT_SOURCES, SOURCE_ROBOFLOW_TRAIN

DEFAULT_VALID_RATIO = 0.2


def _hash_valid(stem: str, valid_ratio: float) -> bool:
    h = int(hashlib.sha256(stem.encode()).hexdigest(), 16)
    return (h % 1000) / 1000.0 < valid_ratio


def assign_yolo_split(
    source: str,
    stem: str,
    *,
    valid_ratio: float = DEFAULT_VALID_RATIO,
) -> str:
    """
    Dataset split for promoted YOLO labels.
    roboflow_valid / roboflow_test → always valid (never train).
    roboflow_train → hash 80/20 train/valid.
    """
    if source in HELD_OUT_SOURCES:
        return "valid"
    if source == SOURCE_ROBOFLOW_TRAIN:
        return "valid" if _hash_valid(stem, valid_ratio) else "train"
    # synthetic / manual — hash split
    return "valid" if _hash_valid(stem, valid_ratio) else "train"


def check_promotion_allowed(source: str, stem: str, *, valid_ratio: float = DEFAULT_VALID_RATIO) -> None:
    """Refuse if a held-out source would land in train (should be impossible with assign_yolo_split)."""
    split = assign_yolo_split(source, stem, valid_ratio=valid_ratio)
    if source in HELD_OUT_SOURCES and split == "train":
        raise RuntimeError(
            f"REFUSED: {stem} source={source} cannot be promoted to train split "
            f"(held-out eval image)."
        )


def summarize_promotion_plan(items: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    """
    items: [{stem, source}, ...]
    Returns {split: Counter(source)} e.g. train/valid counts per source.
    """
    out: dict[str, Counter[str]] = {"train": Counter(), "valid": Counter()}
    for it in items:
        src = it.get("source", "unknown")
        stem = it["stem"]
        split = assign_yolo_split(src, stem)
        out[split][src] += 1
    return out


def print_promotion_summary(items: list[dict[str, Any]], *, title: str = "Promotion plan") -> None:
    plan = summarize_promotion_plan(items)
    print(f"\n{title}")
    print(f"  train: {dict(plan['train'])}")
    print(f"  valid: {dict(plan['valid'])}")
    held_in_train = sum(plan["train"].get(s, 0) for s in HELD_OUT_SOURCES)
    if held_in_train:
        print(f"  *** LEAKAGE: {held_in_train} held-out image(s) would enter train — ABORT ***")
    else:
        print("  held-out sources (roboflow_valid/test): 0 in train ✓")


def scan_reviewed_dataset_labels(dataset_root: Path) -> list[dict[str, str]]:
    """Find reviewed_* labels in train/valid and infer source from matching JSON if present."""
    dataset_root = dataset_root.expanduser().resolve()
    id_cards = Path("test_data/id_cards")
    found: list[dict[str, str]] = []
    for split in ("train", "valid"):
        lbl_dir = dataset_root / split / "labels"
        if not lbl_dir.is_dir():
            continue
        for lbl in lbl_dir.glob("reviewed_*.txt"):
            stem = lbl.stem  # reviewed_real_...
            source = "unknown"
            # try match ground truth json
            for gj in id_cards.rglob(f"{stem.replace('reviewed_', '', 1)}.json"):
                try:
                    data = json.loads(gj.read_text(encoding="utf-8"))
                    source = data.get("source", source)
                except Exception:
                    pass
            found.append({"stem": stem, "source": source, "split": split})
    return found


def assert_no_held_out_in_train(dataset_root: Path) -> None:
    rows = scan_reviewed_dataset_labels(dataset_root)
    leaks = [r for r in rows if r["split"] == "train" and r["source"] in HELD_OUT_SOURCES]
    if leaks:
        names = ", ".join(r["stem"] for r in leaks[:5])
        raise RuntimeError(
            f"Held-out eval images in YOLO train split: {names}. "
            f"Remove from {dataset_root}/train/ before retraining."
        )
