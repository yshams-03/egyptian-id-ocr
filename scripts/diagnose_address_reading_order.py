#!/usr/bin/env python
"""Diagnose address EasyOCR box order vs y-sorted / RTL join (Steps 1-2)."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import export_id_to_excel as eid
import extract_name_address as ena
from extract_id_all import ExtractConfig, _init_ocr, extract_front
from tests.ground_truth import discover_test_cases
from tests.id_metrics import cer, exact_match, normalize_arabic_text
from tests.labeling.sources import HELD_OUT_SOURCES

OUT_DIR = ROOT / "runs" / "diagnose_address_reading"
MIN_SIDE = 120
MAX_SIDE = 520


@dataclass
class BoxRow:
    text: str
    conf: float
    xc: float
    yc: float
    x1: float
    y1: float
    x2: float
    y2: float


def _boxes_from_readtext(results) -> list[BoxRow]:
    rows: list[BoxRow] = []
    for item in results:
        if len(item) < 3:
            continue
        bbox, text, conf = item[0], str(item[1]).strip(), float(item[2])
        if not text:
            continue
        pts = np.asarray(bbox, dtype=float)
        rows.append(
            BoxRow(
                text=text,
                conf=conf,
                xc=float(np.mean(pts[:, 0])),
                yc=float(np.mean(pts[:, 1])),
                x1=float(np.min(pts[:, 0])),
                y1=float(np.min(pts[:, 1])),
                x2=float(np.max(pts[:, 0])),
                y2=float(np.max(pts[:, 1])),
            )
        )
    return rows


def _join_y_lines_rtl(boxes: list[BoxRow], *, line_gap: float = 0.35) -> str:
    """Group by y (top-to-bottom), within line sort by x descending (RTL)."""
    if not boxes:
        return ""
    heights = [b.y2 - b.y1 for b in boxes if b.y2 > b.y1]
    med_h = float(np.median(heights)) if heights else 20.0
    gap = max(8.0, med_h * line_gap)
    sorted_boxes = sorted(boxes, key=lambda b: (b.yc, -b.xc))
    lines: list[list[BoxRow]] = []
    for b in sorted_boxes:
        if not lines or b.yc - lines[-1][-1].yc > gap:
            lines.append([b])
        else:
            lines[-1].append(b)
    out_lines: list[str] = []
    for line in lines:
        line_sorted = sorted(line, key=lambda b: -b.xc)
        out_lines.append(" ".join(x.text for x in line_sorted))
    return "\n".join(out_lines).strip()


def _paragraph_join(results) -> str:
    texts: list[str] = []
    for item in results:
        text = item[1] if len(item) > 1 else ""
        conf = float(item[2]) if len(item) > 2 else 1.0
        if conf >= 0.05 and text.strip():
            texts.append(text.strip())
    return " ".join(texts).strip()


def inspect_address_crop(bgr, reader) -> dict:
    big = eid.upscale_crop(bgr, min_side=max(MIN_SIDE, 120))
    big = eid.resize_for_speed(big, max_side=MAX_SIDE)
    rgb = cv2.cvtColor(big, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    raw_para = reader.readtext(rgb, paragraph=True)
    raw_detail = reader.readtext(rgb, detail=1, paragraph=False)

    boxes = _boxes_from_readtext(raw_detail)
    y_sorted = sorted(boxes, key=lambda b: (b.yc, -b.xc))
    join_default = _paragraph_join(raw_para)
    join_detail_space = " ".join(b.text for b in boxes)
    join_y_rtl = _join_y_lines_rtl(boxes)
    join_y_only = "\n".join(
        " ".join(b.text for b in sorted(line, key=lambda x: -x.xc))
        for line in (
            [sorted([b], key=lambda x: x.yc)]
            if len(boxes) == 1
            else []
        )
    )
    # rebuild y-only groups properly
    if boxes:
        med_h = float(np.median([b.y2 - b.y1 for b in boxes if b.y2 > b.y1] or [20]))
        gap = max(8.0, med_h * 0.35)
        sb = sorted(boxes, key=lambda b: b.yc)
        groups: list[list[BoxRow]] = []
        for b in sb:
            if not groups or b.yc - groups[-1][-1].yc > gap:
                groups.append([b])
            else:
                groups[-1].append(b)
        join_y_only = "\n".join(
            " ".join(x.text for x in sorted(g, key=lambda b: -b.xc)) for g in groups
        )

    return {
        "crop_size": [w, h],
        "paragraph_true": {
            "n_blocks": len(raw_para),
            "blocks": [
                {
                    "text": str(it[1]).strip(),
                    "conf": float(it[2]) if len(it) > 2 else None,
                    "has_bbox": len(it) > 0 and it[0] is not None,
                }
                for it in raw_para
            ],
            "joined": join_default,
        },
        "detail_false": {
            "n_boxes": len(boxes),
            "raw_order": [
                {"i": i, "text": b.text, "conf": b.conf, "xc": round(b.xc, 1), "yc": round(b.yc, 1)}
                for i, b in enumerate(boxes)
            ],
            "y_then_rtl_order": [
                {"text": b.text, "xc": round(b.xc, 1), "yc": round(b.yc, 1)} for b in y_sorted
            ],
            "joined_space": join_detail_space,
            "joined_y_rtl_newlines": join_y_rtl,
            "joined_y_rtl_flat": normalize_arabic_text(join_y_rtl),
        },
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "0"
    try:
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"

    import easyocr
    from ultralytics import YOLO

    reader = easyocr.Reader(["ar", "en"], gpu=device != "cpu", verbose=False)
    field_yolo = YOLO(str(ROOT / "runs" / "train_id_detectr_hyper" / "weights" / "best.pt"))
    cfg = ExtractConfig(image=ROOT / "x.jpg", quiet=True, fast_mode=True, engine="easyocr")
    eng, tl, _ = _init_ocr(cfg, device)

    held = [
        c
        for c in discover_test_cases(ROOT / "test_data" / "id_cards")
        if str(c["ground_truth"].get("source") or "") in HELD_OUT_SOURCES
        and (c["ground_truth"].get("address") or "").strip()
    ]

    rows = []
    for case in held:
        gt = case["ground_truth"]
        exp = (gt["address"] or "").strip()
        row = extract_front(case["front"], cfg, device=device, engine=eng, tess_langs=tl, easyocr_reader=reader, dw=None)
        act = eid.clean_address_text(row.get("address", ""), strip_digits=False)
        c = cer(exp, act)
        ok = exact_match(exp, act, field="address") or c <= 0.15
        rows.append(
            {
                "image": case["front"].name,
                "passed": ok,
                "cer": c,
                "expected": exp,
                "expected_has_newline": "\n" in exp,
                "actual": act,
                "expected_normalized": normalize_arabic_text(exp),
                "actual_normalized": normalize_arabic_text(act),
            }
        )

    failures = sorted([r for r in rows if not r["passed"]], key=lambda x: -x["cer"])[:8]

    inspections = []
    for f in failures:
        case = next(c for c in held if c["front"].name == f["image"])
        img = cv2.imread(str(case["front"]))
        img = eid.resize_for_speed(img, 880)
        pred = field_yolo.predict(source=img, conf=0.25, device=device, imgsz=480, verbose=False)[0]
        if pred.boxes is None:
            continue
        best = eid.best_boxes_by_label(
            pred.boxes.xyxy.cpu().numpy(),
            pred.boxes.cls.cpu().numpy().astype(int),
            pred.boxes.conf.cpu().numpy(),
            eid.load_class_names(),
        )
        if "address" not in best:
            continue
        cr = eid.crop_xyxy(img, best["address"][0], 6)
        insp = inspect_address_crop(cr, reader)
        insp["image"] = f["image"]
        insp["expected"] = f["expected"]
        insp["pipeline_actual"] = f["actual"]
        insp["cer"] = f["cer"]
        insp["expected_has_newline"] = f["expected_has_newline"]
        # score joins
        exp_n = normalize_arabic_text(f["expected"])
        for key in (
            "paragraph_true.joined",
            "detail_false.joined_space",
            "detail_false.joined_y_rtl_newlines",
        ):
            parts = key.split(".")
            val = insp
            for p in parts:
                val = val[p]
            insp.setdefault("join_cer", {})[key] = cer(f["expected"], val)
            insp.setdefault("join_match_norm", {})[key] = exp_n == normalize_arabic_text(val)
        inspections.append(insp)

    payload = {
        "scoring_note": (
            "normalize_arabic_text collapses newlines to spaces via _PUNCT_SPACE; "
            "multi-line GT is compared as flattened single line for CER."
        ),
        "current_join": "ocr_text_field_easyocr: paragraph=True -> space-join blocks in EasyOCR return order",
        "held_out_failures_inspected": len(inspections),
        "inspections": inspections,
        "newline_stats": {
            "held_with_newline_in_gt": sum(1 for r in rows if r["expected_has_newline"]),
            "held_total": len(rows),
        },
    }
    out_json = OUT_DIR / "reading_order_inspection.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # markdown summary
    lines = [
        "# Address reading-order diagnosis",
        "",
        "## Step 2 — Current join logic",
        "",
        "- **Path:** `extract_id_all.extract_front` → `ena.ocr_text_field_easyocr` → `reader.readtext(..., paragraph=True)`",
        "- **Join:** each paragraph block's text collected in **EasyOCR return order**, joined with **single spaces** (`' '.join(texts)`).",
        "- **No explicit y-sort or RTL sort.** No `\\n` inserted.",
        "- **Post-process:** `eid.clean_address_text` splits on spaces only (isolated short digit tokens dropped).",
        "",
        "### Scoring vs multi-line ground truth",
        "",
        f"- Held-out samples with `\\n` in GT address: **{payload['newline_stats']['held_with_newline_in_gt']}/{payload['newline_stats']['held_total']}**",
        "- `normalize_arabic_text` maps whitespace/newlines/punctuation runs to a **single space** before CER.",
        "- So GT `line1\\nline2` and OCR `line1 line2` are **scoring-equivalent** after normalization — newline mismatch is **not** a separate scoring penalty.",
        "- Remaining failures are **word content/order errors**, not a newline-only artifact.",
        "",
        "## Step 1 — Raw EasyOCR boxes (worst held-out address failures)",
        "",
    ]
    for insp in inspections:
        lines.append(f"### `{insp['image']}` (CER={insp['cer']:.3f}, GT newline={insp['expected_has_newline']})")
        lines.append(f"- **Expected:** `{insp['expected'].replace(chr(10), ' / ')}`")
        lines.append(f"- **Pipeline:** `{insp['pipeline_actual']}`")
        pt = insp["paragraph_true"]
        lines.append(f"- **paragraph=True:** {pt['n_blocks']} block(s) → `{pt['joined']}`")
        df = insp["detail_false"]
        lines.append(f"- **detail=1 raw order ({df['n_boxes']} boxes):**")
        for b in df["raw_order"]:
            lines.append(f"  - [{b['i']}] yc={b['yc']} xc={b['xc']} conf={b['conf']:.2f} `{b['text']}`")
        lines.append("- **y-sort then RTL (proposed order):**")
        for b in df["y_then_rtl_order"]:
            lines.append(f"  - yc={b['yc']} xc={b['xc']} `{b['text']}`")
        lines.append(f"- **Joined (raw space):** `{df['joined_space']}`")
        lines.append(f"- **Joined (y-line RTL + newlines):** `{df['joined_y_rtl_newlines'].replace(chr(10), ' | ')}`")
        jc = insp.get("join_cer", {})
        lines.append(
            f"- **CER vs GT:** paragraph={jc.get('paragraph_true.joined', 0):.3f} "
            f"raw_space={jc.get('detail_false.joined_space', 0):.3f} "
            f"y_rtl={jc.get('detail_false.joined_y_rtl_newlines', 0):.3f}"
        )
        lines.append("")

    out_md = OUT_DIR / "report.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
