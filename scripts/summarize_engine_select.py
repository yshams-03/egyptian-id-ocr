#!/usr/bin/env python
"""Summarize local engine selection stats vs ground truth (evaluation only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_engine_select import get_engine_select_stats
from tests.ground_truth import discover_test_cases, load_ground_truth
from tests.id_metrics import cer, exact_match, normalize_arabic_text, DEFAULT_CER_THRESHOLD
from tests.labeling.sources import HELD_OUT_SOURCES


def _expected_for_field(gt: dict, field_label: str) -> str:
    if field_label == "firstName":
        return (gt.get("first_name") or "").strip()
    if field_label == "lastName":
        return (gt.get("last_name") or "").strip()
    if field_label == "address":
        return (gt.get("address") or "").strip()
    return ""


def summarize(data_dir: Path, out_path: Path | None = None) -> str:
    stats = get_engine_select_stats()
    gt_by_image = {c["front"].name: c["ground_truth"] for c in discover_test_cases(data_dir)}

    lines = ["# Local engine select — evaluation summary", ""]
    tess_picks = 0
    improved = worse = same = 0
    held_tess = held_imp = held_worse = 0

    lines.append("| image | field | chosen | easy CER | tess CER | chosen CER | vs easy |")
    lines.append("|-------|-------|--------|----------|----------|------------|---------|")

    for row in stats.selections:
        img = row["image"]
        if isinstance(img, str) and ("/" in img or "\\" in img):
            img = Path(img).name
        gt = gt_by_image.get(img, {})
        exp = _expected_for_field(gt, row["field"])
        if not exp and row["field"] in ("firstName", "lastName"):
            full = (
                (gt.get("full_name") or "").strip()
                or f"{gt.get('first_name', '')} {gt.get('last_name', '')}".strip()
            )
            parts = full.split()
            if row["field"] == "firstName" and parts:
                exp = parts[0]
            elif row["field"] == "lastName" and len(parts) > 1:
                exp = " ".join(parts[1:])

        easy_c = cer(exp, row["easyocr"]) if exp else None
        tess_c = cer(exp, row["tesseract"]) if exp and row["tesseract"] else None
        chosen_text = row["tesseract"] if row["chosen"] == "tesseract" else row["easyocr"]
        chosen_c = cer(exp, chosen_text) if exp else None

        vs = ""
        if easy_c is not None and chosen_c is not None:
            if chosen_c + 0.001 < easy_c:
                vs = "improved"
                improved += 1
            elif chosen_c > easy_c + 0.001:
                vs = "worse"
                worse += 1
            else:
                vs = "same"
                same += 1

        if row["chosen"] == "tesseract":
            tess_picks += 1
            src = str(gt.get("source") or "")
            if src in HELD_OUT_SOURCES:
                held_tess += 1
                if vs == "improved":
                    held_imp += 1
                elif vs == "worse":
                    held_worse += 1

        ec = f"{easy_c:.3f}" if easy_c is not None else "—"
        tc = f"{tess_c:.3f}" if tess_c is not None else "—"
        cc = f"{chosen_c:.3f}" if chosen_c is not None else "—"
        lines.append(f"| `{img[:28]}…` | {row['field']} | {row['chosen']} | {ec} | {tc} | {cc} | {vs} |")

    lines.extend(
        [
            "",
            f"- Total field selections: **{len(stats.selections)}**",
            f"- Tesseract chosen: **{tess_picks}**",
            f"- vs EasyOCR baseline: improved **{improved}**, worse **{worse}**, same **{same}**",
            f"- Held-out Tesseract picks: **{held_tess}** (improved **{held_imp}**, worse **{held_worse}**)",
        ]
    )
    text = "\n".join(lines)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    data = ROOT / "test_data" / "id_cards"
    out = ROOT / "runs" / "diagnose_heldout_failures" / "engine_select_summary.md"
    print(summarize(data, out))
    print(f"Wrote {out}")
