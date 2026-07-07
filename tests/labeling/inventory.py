"""
Shared inventory: Roboflow front images vs verified / draft / unlabeled ground truth.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from tests.ground_truth import IMAGE_EXTS, resolve_ground_truth_path
from tests.labeling.import_roboflow import (
    REAL_PREFIX,
    canonical_roboflow_stem,
    discover_dataset_images,
    has_required_front_boxes,
    is_likely_back_card,
    local_real_stem_for_dataset_stem,
    label_path_for_image,
)
from tests.labeling.prefill import DRAFTS_DIRNAME, discover_front_images, is_verified_ground_truth
from tests.labeling.sources import split_name_to_source

GtCategory = Literal["verified", "draft", "none"]

FIXTURES_GT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "ground_truth"


@dataclass
class FrontImageRow:
    rob_stem: str
    split: str
    source: str
    roboflow_path: Path
    front_ready: bool
    imported: bool = False
    local_stem: str = ""
    local_image: Path | None = None
    gt_category: GtCategory = "none"
    gt_path: Path | None = None


@dataclass
class InventoryReport:
    roboflow_fronts: list[FrontImageRow] = field(default_factory=list)
    extra_id_cards: list[FrontImageRow] = field(default_factory=list)

    @property
    def verified(self) -> list[FrontImageRow]:
        return [r for r in self.roboflow_fronts if r.gt_category == "verified"]

    @property
    def draft(self) -> list[FrontImageRow]:
        return [r for r in self.roboflow_fronts if r.gt_category == "draft"]

    @property
    def unlabeled(self) -> list[FrontImageRow]:
        return [r for r in self.roboflow_fronts if r.gt_category == "none"]

    def counts_by_source(self, rows: list[FrontImageRow]) -> dict[str, int]:
        return dict(Counter(r.source for r in rows))


def _candidate_stems(rob_stem: str, local_stem: str | None) -> list[str]:
    stems: list[str] = []
    canon = canonical_roboflow_stem(rob_stem)
    if local_stem:
        stems.append(local_stem)
    stems.append(local_real_stem_for_dataset_stem(rob_stem))
    if canon != rob_stem:
        stems.append(f"{REAL_PREFIX}{rob_stem}")
    stems.append(canon)
    if canon != rob_stem:
        stems.append(rob_stem)
    return stems


def classify_ground_truth(
    data_dir: Path,
    rob_stem: str,
    local_stem: str | None = None,
) -> tuple[GtCategory, Path | None]:
    """Return verified | draft | none for a Roboflow stem."""
    data_dir = data_dir.expanduser().resolve()
    for stem in _candidate_stems(rob_stem, local_stem):
        for base in (data_dir / "real", data_dir / "archive", data_dir):
            for ext in IMAGE_EXTS:
                img = base / f"{stem}{ext}"
                if not img.is_file():
                    continue
                gt_path = resolve_ground_truth_path(img, data_dir)
                if not gt_path:
                    continue
                data = json.loads(gt_path.read_text(encoding="utf-8"))
                if data.get("review_status") == "needs_review":
                    return "draft", gt_path
                if is_verified_ground_truth(gt_path):
                    return "verified", gt_path

    for stem in _candidate_stems(rob_stem, local_stem):
        draft = data_dir / DRAFTS_DIRNAME / f"{stem}.json"
        if draft.is_file():
            return "draft", draft

    fix = FIXTURES_GT / f"{rob_stem}.json"
    if fix.is_file():
        data = json.loads(fix.read_text(encoding="utf-8"))
        if data.get("review_status") == "needs_review":
            return "draft", fix
        if is_verified_ground_truth(fix):
            return "verified", fix

    return "none", None


def has_draft_json(data_dir: Path, stem: str) -> bool:
    return (data_dir / DRAFTS_DIRNAME / f"{stem}.json").is_file()


def build_inventory(data_dir: Path) -> InventoryReport:
    data_dir = data_dir.expanduser().resolve()
    report = InventoryReport()

    id_card_by_rob: dict[str, Path] = {}
    for img in discover_front_images(data_dir):
        rob_stem = canonical_roboflow_stem(img.stem)
        id_card_by_rob[rob_stem] = img

    for split in ("train", "valid", "test"):
        for img in discover_dataset_images(split):
            lbl = label_path_for_image(img)
            if is_likely_back_card(img, lbl):
                continue
            rob_stem = canonical_roboflow_stem(img.stem)
            local = id_card_by_rob.get(rob_stem)
            local_stem = local.stem if local else None
            cat, gt_path = classify_ground_truth(data_dir, rob_stem, local_stem)
            report.roboflow_fronts.append(
                FrontImageRow(
                    rob_stem=rob_stem,
                    split=split,
                    source=split_name_to_source(split),
                    roboflow_path=img,
                    front_ready=has_required_front_boxes(lbl),
                    imported=local is not None,
                    local_stem=local_stem or "",
                    local_image=local,
                    gt_category=cat,
                    gt_path=gt_path,
                )
            )

    rob_stems = {r.rob_stem for r in report.roboflow_fronts}
    for rob_stem, img in sorted(id_card_by_rob.items()):
        if rob_stem in rob_stems:
            continue
        cat, gt_path = classify_ground_truth(data_dir, rob_stem, img.stem)
        report.extra_id_cards.append(
            FrontImageRow(
                rob_stem=rob_stem,
                split="",
                source="",
                roboflow_path=img,
                front_ready=False,
                imported=True,
                local_stem=img.stem,
                local_image=img,
                gt_category=cat,
                gt_path=gt_path,
            )
        )
    return report


def print_inventory_report(report: InventoryReport) -> None:
    total = len(report.roboflow_fronts)
    front_ready = sum(1 for r in report.roboflow_fronts if r.front_ready)
    print("=== Roboflow front images (all splits, excl backs) ===")
    print(f"Total front candidates: {total}")
    print(f"  Front-ready (all required YOLO field boxes): {front_ready}")
    print()
    print("=== Ground truth classification (roboflow fronts) ===")
    print(f"  (a) Verified — SKIP: {len(report.verified)}")
    print(f"      by source: {report.counts_by_source(report.verified)}")
    print(f"  (b) Draft/needs_review: {len(report.draft)}")
    print(f"      by source: {report.counts_by_source(report.draft)}")
    print(f"  (c) No ground truth: {len(report.unlabeled)}")
    print(f"      by source: {report.counts_by_source(report.unlabeled)}")
    print(
        f"      front_ready: {sum(1 for r in report.unlabeled if r.front_ready)} / "
        f"not ready: {sum(1 for r in report.unlabeled if not r.front_ready)}"
    )
    print(
        f"      already imported: {sum(1 for r in report.unlabeled if r.imported)} / "
        f"not imported: {sum(1 for r in report.unlabeled if not r.imported)}"
    )
    extra_v = sum(1 for r in report.extra_id_cards if r.gt_category == "verified")
    extra_d = sum(1 for r in report.extra_id_cards if r.gt_category == "draft")
    extra_n = sum(1 for r in report.extra_id_cards if r.gt_category == "none")
    print()
    print("=== Extra id_cards images (not in roboflow scan) ===")
    print(f"  verified: {extra_v}, draft: {extra_d}, none: {extra_n}")
