"""
Promote human-reviewed labels into Egyptian-ID-Detectr-3 train/valid split.

Images are copied with a reviewed_ prefix — original Roboflow files are never overwritten.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import export_id_to_excel as eid

from tests.ground_truth import GROUND_TRUTH_KEYS
from tests.labeling.prefill import DRAFT_LABELS_DIRNAME, DRAFTS_DIRNAME
from tests.labeling.split_guard import (
    DEFAULT_VALID_RATIO,
    assign_yolo_split,
    check_promotion_allowed,
)
from tests.labeling.sources import normalize_source
from tests.labeling.yolo_boxes import read_draft_label_file, write_draft_label_file, YoloBox

DATASET_ROOT = Path("egyptian_id_detectr/content/Egyptian-ID-Detectr-3")


def _find_image(data_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        direct = data_dir / f"{stem}{ext}"
        if direct.is_file():
            return direct
        for p in data_dir.rglob(f"{stem}{ext}"):
            if p.is_file():
                return p
    return None


def list_pending_drafts(
    data_dir: Path,
    *,
    sort_mode: str = "fast_confirm",
) -> list[dict[str, Any]]:
    """
    sort_mode:
      fast_confirm — low-priority (boxed + OCR'd) first; default for throughput
      needs_work   — high-priority (missing boxes/OCR) first
      stem         — alphabetical by stem
    """
    data_dir = data_dir.expanduser().resolve()
    drafts = data_dir / DRAFTS_DIRNAME
    if not drafts.is_dir():
        return []
    pending: list[dict[str, Any]] = []
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    for dj in sorted(drafts.glob("*.json")):
        stem = dj.stem
        img = _find_image(data_dir, stem)
        label = data_dir / DRAFT_LABELS_DIRNAME / f"{stem}.txt"
        gt = json.loads(dj.read_text(encoding="utf-8"))
        missing_ocr = gt.get("draft_missing_ocr", [])
        missing_det = gt.get("draft_missing_detections", [])
        priority = gt.get("draft_review_priority") or (
            "high" if missing_det or len(missing_ocr) >= 2 else ("medium" if missing_ocr else "low")
        )
        pending.append(
            {
                "stem": stem,
                "image_path": str(img) if img else "",
                "draft_json": str(dj),
                "draft_label": str(label) if label.is_file() else "",
                "review_status": gt.get("review_status", "needs_review"),
                "missing_detections": missing_det,
                "missing_ocr": missing_ocr,
                "review_priority": priority,
                "label_source": gt.get("draft_label_source", ""),
                "source": gt.get("source", ""),
            }
        )
    if sort_mode == "stem":
        pending.sort(key=lambda r: r["stem"])
    elif sort_mode == "needs_work":
        pending.sort(
            key=lambda r: (
                priority_rank.get(str(r.get("review_priority", "low")), 9),
                -len(r.get("missing_ocr") or []),
                -len(r.get("missing_detections") or []),
                r["stem"],
            )
        )
    else:
        # fast_confirm (default): low -> medium -> high
        pending.sort(
            key=lambda r: (
                -priority_rank.get(str(r.get("review_priority", "low")), 9),
                len(r.get("missing_ocr") or []),
                len(r.get("missing_detections") or []),
                r["stem"],
            )
        )
    return pending


def save_reviewed(
    data_dir: Path,
    stem: str,
    ground_truth: dict[str, Any],
    boxes: list[dict[str, Any]],
    *,
    valid_ratio: float = DEFAULT_VALID_RATIO,
    promote_to_dataset: bool = True,
) -> dict[str, str]:
    """
    Write verified JSON beside image, remove review_status, optionally promote YOLO labels.
    Returns paths written.
    """
    data_dir = data_dir.expanduser().resolve()
    drafts_dir = data_dir / DRAFTS_DIRNAME
    labels_dir = data_dir / DRAFT_LABELS_DIRNAME

    image_path = _find_image(data_dir, stem)
    if image_path is None:
        raise FileNotFoundError(f"No image for stem {stem}")

    source = normalize_source(str(ground_truth.get("source", "")))

    clean: dict[str, Any] = {k: ground_truth.get(k, "") for k in GROUND_TRUTH_KEYS}
    clean["source"] = source
    tags = [t for t in ground_truth.get("tags", []) if t != "draft_prefill"]
    if "reviewed" not in tags:
        tags.append("reviewed")
    clean["tags"] = tags
    clean["notes"] = (ground_truth.get("notes") or "").replace("DRAFT", "Reviewed").strip()

    final_json = image_path.with_suffix(".json")
    final_json.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")

    class_names = eid.load_class_names()
    yolo_boxes: list[YoloBox] = []
    for b in boxes:
        cid = int(b["class_id"])
        yolo_boxes.append(
            YoloBox(
                class_id=cid,
                class_name=class_names.get(cid, str(cid)),
                cx=float(b["cx"]),
                cy=float(b["cy"]),
                w=float(b["w"]),
                h=float(b["h"]),
                conf=1.0,
            )
        )

    reviewed_label = labels_dir / f"{stem}.txt"
    write_draft_label_file(yolo_boxes, reviewed_label)

    out: dict[str, str] = {"ground_truth": str(final_json), "draft_label": str(reviewed_label)}

    if promote_to_dataset:
        check_promotion_allowed(source, stem, valid_ratio=valid_ratio)
        split = assign_yolo_split(source, stem, valid_ratio=valid_ratio)
        ds_images = DATASET_ROOT / split / "images"
        ds_labels = DATASET_ROOT / split / "labels"
        ds_images.mkdir(parents=True, exist_ok=True)
        ds_labels.mkdir(parents=True, exist_ok=True)
        dest_stem = f"reviewed_{stem}"
        dest_img = ds_images / f"{dest_stem}{image_path.suffix.lower()}"
        dest_lbl = ds_labels / f"{dest_stem}.txt"
        shutil.copy2(image_path, dest_img)
        write_draft_label_file(yolo_boxes, dest_lbl)
        out["dataset_image"] = str(dest_img)
        out["dataset_label"] = str(dest_lbl)
        out["dataset_split"] = split

    draft_json = drafts_dir / f"{stem}.json"
    if draft_json.is_file():
        draft_json.unlink()

    return out


def load_draft_for_review(data_dir: Path, stem: str) -> dict[str, Any]:
    data_dir = data_dir.expanduser().resolve()
    draft_json = data_dir / DRAFTS_DIRNAME / f"{stem}.json"
    if not draft_json.is_file():
        raise FileNotFoundError(f"No draft for {stem}")
    gt = json.loads(draft_json.read_text(encoding="utf-8"))
    label_path = data_dir / DRAFT_LABELS_DIRNAME / f"{stem}.txt"
    boxes = read_draft_label_file(label_path)
    return {
        "ground_truth": gt,
        "boxes": [
            {
                "class_id": b.class_id,
                "class_name": b.class_name,
                "cx": b.cx,
                "cy": b.cy,
                "w": b.w,
                "h": b.h,
            }
            for b in boxes
        ],
    }
