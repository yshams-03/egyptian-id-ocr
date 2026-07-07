"""
Import real front ID photos from Egyptian-ID-Detectr-3 into test_data/id_cards/.

Roboflow images already have YOLO box labels — copied to draft_labels/.
OCR text is pre-filled via prefill.py (human review required).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.labeling.prefill import DRAFT_LABELS_DIRNAME, DRAFTS_DIRNAME
from tests.labeling.sources import split_name_to_source
from tests.labeling.yolo_boxes import (
    FRONT_FIELD_NAMES,
    REQUIRED_FRONT_FIELDS,
    read_draft_label_file,
    write_draft_label_file,
    YoloBox,
)
import export_id_to_excel as eid

DATASET = ROOT / "egyptian_id_detectr" / "content" / "Egyptian-ID-Detectr-3"
DEFAULT_OUT = ROOT / "test_data" / "id_cards" / "real"
IMPORT_META_DIRNAME = "import_meta"
REAL_PREFIX = "real_"


def canonical_roboflow_stem(stem: str) -> str:
    """
    Normalize Roboflow/front-review prefixes to one canonical ID-card stem.

    Examples:
      reviewed_real_Front_... -> Front_...
      real_Front_...          -> Front_...
      Front_...               -> Front_...
    """
    out = stem
    if out.startswith("reviewed_"):
        out = out[len("reviewed_") :]
    if out.startswith(REAL_PREFIX):
        out = out[len(REAL_PREFIX) :]
    return out


def local_real_stem_for_dataset_stem(stem: str) -> str:
    """Map a dataset image stem to the canonical local `real_<stem>` name."""
    return f"{REAL_PREFIX}{canonical_roboflow_stem(stem)}"


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" not in parts:
        raise ValueError(f"Not a dataset image path: {image_path}")
    i = parts.index("images")
    return Path(*parts[:i], "labels", *parts[i + 1 :]).with_suffix(".txt")


def filter_front_boxes(label_path: Path) -> list[YoloBox]:
    names = eid.load_class_names()
    boxes = read_draft_label_file(label_path, names)
    return [b for b in boxes if b.class_name in FRONT_FIELD_NAMES]


def is_likely_back_card(src_image: Path, label_path: Path) -> bool:
    name = src_image.stem.lower()
    if "back" in name or "_id_back" in name:
        return True
    if not label_path.is_file():
        return False
    names = eid.load_class_names()
    boxes = read_draft_label_file(label_path, names)
    front = {b.class_name for b in boxes if b.class_name in FRONT_FIELD_NAMES}
    back_markers = {"job", "nid_back", "expiry", "issue", "demo", "watermark_tut"}
    return not front and bool(back_markers & {b.class_name for b in boxes})


def has_required_front_boxes(label_path: Path) -> bool:
    if not label_path.is_file():
        return False
    names = eid.load_class_names()
    boxes = read_draft_label_file(label_path, names)
    found = {b.class_name for b in boxes}
    return REQUIRED_FRONT_FIELDS <= found


def discover_dataset_images(split: str) -> list[Path]:
    img_dir = DATASET / split / "images"
    if not img_dir.is_dir():
        return []
    return sorted(
        p
        for p in img_dir.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def cleanup_real_imports(data_dir: Path) -> int:
    """
    Remove prior real_* imports from real/, drafts/, draft_labels/, import_meta/.
    Does NOT touch verified non-draft JSON (e.g. synthetic labeled_*.json at parent).
    """
    data_dir = data_dir.expanduser().resolve()
    removed = 0
    patterns = [
        (data_dir / "real", f"{REAL_PREFIX}*"),
        (data_dir / DRAFTS_DIRNAME, f"{REAL_PREFIX}*.json"),
        (data_dir / DRAFT_LABELS_DIRNAME, f"{REAL_PREFIX}*.txt"),
        (data_dir / IMPORT_META_DIRNAME, f"{REAL_PREFIX}*.json"),
    ]
    for folder, glob in patterns:
        if not folder.is_dir():
            continue
        for p in folder.glob(glob):
            p.unlink()
            removed += 1
    return removed


def import_one(
    src_image: Path,
    out_dir: Path,
    *,
    source: str,
    use_symlink: bool = False,
    overwrite: bool = True,
) -> dict[str, Any] | None:
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = local_real_stem_for_dataset_stem(src_image.stem)
    dest_image = out_dir / f"{stem}{src_image.suffix.lower()}"

    if dest_image.is_file() and not overwrite:
        return {"stem": stem, "status": "exists", "source": source}

    if dest_image.is_file():
        dest_image.unlink()

    if use_symlink:
        dest_image.symlink_to(src_image.resolve())
    else:
        shutil.copy2(src_image, dest_image)

    labels_root = out_dir.parent / DRAFT_LABELS_DIRNAME
    labels_root.mkdir(parents=True, exist_ok=True)
    src_lbl = label_path_for_image(src_image)
    front_boxes = filter_front_boxes(src_lbl) if src_lbl.is_file() else []
    write_draft_label_file(front_boxes, labels_root / f"{stem}.txt")

    meta_dir = out_dir.parent / IMPORT_META_DIRNAME
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "stem": stem,
        "source": source,
        "source_image": str(src_image.resolve()),
        "source_split": src_image.parent.parent.name,
        "source_label": str(src_lbl) if src_lbl.is_file() else "",
        "yolo_box_count": len(front_boxes),
        "front_ready": has_required_front_boxes(src_lbl),
    }
    (meta_dir / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "stem": stem,
        "image": dest_image.name,
        "source": source,
        "boxes": len(front_boxes),
        "front_ready": meta["front_ready"],
        "status": "imported",
    }


def collect_front_ready_candidates(
    split: str,
    *,
    front_only: bool = True,
    require_boxes: bool = True,
) -> list[Path]:
    out: list[Path] = []
    for img in discover_dataset_images(split):
        lbl = label_path_for_image(img)
        if front_only and is_likely_back_card(img, lbl):
            continue
        if require_boxes and not has_required_front_boxes(lbl):
            continue
        out.append(img)
    return out


def import_all_unverified(
    out_dir: Path,
    *,
    data_dir: Path | None = None,
    front_only: bool = True,
    require_boxes: bool = False,
    use_symlink: bool = False,
    skip_existing: bool = True,
) -> list[dict[str, Any]]:
    """
    Import every Roboflow front image that lacks verified ground truth.
    Preserves roboflow_train / roboflow_valid / roboflow_test source tags via import_meta/.
    """
    out_dir = out_dir.expanduser().resolve()
    data_root = data_dir.expanduser().resolve() if data_dir else out_dir.parent
    results: list[dict[str, Any]] = []

    from tests.labeling.inventory import classify_ground_truth

    for split in ("train", "valid", "test"):
        for img in collect_front_ready_candidates(
            split, front_only=front_only, require_boxes=require_boxes
        ):
            rob_stem = canonical_roboflow_stem(img.stem)
            cat, _ = classify_ground_truth(data_root, rob_stem)
            if cat == "verified":
                continue
            dest_stem = local_real_stem_for_dataset_stem(img.stem)
            dest_image = out_dir / f"{dest_stem}{img.suffix.lower()}"
            if skip_existing and dest_image.is_file():
                results.append(
                    {
                        "stem": dest_stem,
                        "image": dest_image.name,
                        "source": split_name_to_source(split),
                        "status": "exists",
                    }
                )
                continue
            row = import_one(
                img,
                out_dir,
                source=split_name_to_source(split),
                use_symlink=use_symlink,
                overwrite=True,
            )
            if row:
                results.append(row)
    return results


def run_reimport(
    out_dir: Path,
    *,
    target_total: int = 33,
    train_extra: int = 20,
    use_symlink: bool = False,
    cleanup: bool = True,
) -> list[dict[str, Any]]:
    data_root = out_dir.parent
    if cleanup:
        n = cleanup_real_imports(data_root)
        print(f"Cleaned {n} prior real_* import file(s)")

    results: list[dict[str, Any]] = []

    for split in ("valid", "test"):
        for img in collect_front_ready_candidates(split):
            row = import_one(
                img,
                out_dir,
                source=split_name_to_source(split),
                use_symlink=use_symlink,
            )
            if row:
                results.append(row)

    held_out = len(results)
    need = max(0, target_total - held_out)
    train_limit = min(train_extra, need) if need else train_extra

    train_candidates = collect_front_ready_candidates("train")
    for img in train_candidates[:train_limit]:
        row = import_one(
            img,
            out_dir,
            source=split_name_to_source("train"),
            use_symlink=use_symlink,
        )
        if row:
            results.append(row)

    return results


def print_import_summary(results: list[dict[str, Any]]) -> None:
    by_source = Counter(r["source"] for r in results if r.get("status") == "imported")
    front_ready = sum(1 for r in results if r.get("front_ready"))
    print(f"\n{'='*60}")
    print("FRONT-READY IMPORT SUMMARY")
    print(f"{'='*60}")
    print(f"Total imported (front-ready): {len(results)}")
    print(f"With all required YOLO field boxes: {front_ready}")
    print()
    print(f"{'Source':<22} {'Count':>6}")
    print("-" * 30)
    for src in ("roboflow_valid", "roboflow_test", "roboflow_train"):
        print(f"{src:<22} {by_source.get(src, 0):>6}")
    print(f"\nImages -> {DEFAULT_OUT}")
    print(f"YOLO draft labels -> {DEFAULT_OUT.parent / DRAFT_LABELS_DIRNAME}")
    print(f"Source metadata -> {DEFAULT_OUT.parent / IMPORT_META_DIRNAME}")
    print("\nNext:")
    print(f"  py -m tests.labeling.prefill --data-dir {DEFAULT_OUT.parent} --only real")
    print("  py -m tests.labeling.review_app")


def main() -> int:
    parser = argparse.ArgumentParser(description="Import front-ready Roboflow ID photos")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-total", type=int, default=33)
    parser.add_argument("--train-extra", type=int, default=20, help="Max train-split fronts to add")
    parser.add_argument("--symlink", action="store_true")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep existing real_* imports")
    parser.add_argument(
        "--all-unverified",
        action="store_true",
        help="Import all Roboflow front images without verified ground truth (skip cleanup).",
    )
    parser.add_argument(
        "--require-boxes",
        action="store_true",
        help="With --all-unverified: only import images with all required YOLO field boxes.",
    )
    args = parser.parse_args()

    if args.all_unverified:
        results = import_all_unverified(
            args.out_dir,
            front_only=True,
            require_boxes=args.require_boxes,
            use_symlink=args.symlink,
        )
        print_import_summary(results)
        return 0

    results = run_reimport(
        args.out_dir,
        target_total=args.target_total,
        train_extra=args.train_extra,
        use_symlink=args.symlink,
        cleanup=not args.no_cleanup,
    )
    print_import_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
