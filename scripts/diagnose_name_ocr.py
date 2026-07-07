#!/usr/bin/env python
"""
Diagnosis-only: name OCR failure modes on held-out and roboflow_train samples.
Does not modify pipeline config. Writes report + worst-case crops under runs/diagnose_name/.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import export_id_to_excel as eid
import extract_name_address as ena
from extract_id_all import DEFAULT_FIELD_WEIGHTS, ExtractConfig, _init_ocr, extract_front
from tests.ground_truth import discover_test_cases, load_ground_truth
from tests.id_metrics import cer, exact_match, normalize_arabic_text
from tests.labeling.sources import HELD_OUT_SOURCES, SOURCE_ROBOFLOW_TRAIN

OUT_DIR = ROOT / "runs" / "diagnose_name"
CROP_DIR = OUT_DIR / "worst_crops"


@dataclass
class NameRow:
    image: str
    source: str
    expected: str
    actual: str
    first_name: str
    last_name: str
    cer: float
    passed: bool
    category: str = ""
    batch_full: str = ""
    individual_full: str = ""
    batch_differs: bool = False
    missing_labels: list[str] = field(default_factory=list)


def _expected_name(gt: dict) -> str:
    return (
        (gt.get("full_name") or "").strip()
        or (gt.get("name") or "").strip()
        or f"{gt.get('first_name', '')} {gt.get('last_name', '')}".strip()
    )


def _levenshtein_backtrace_substitutions(ref: str, hyp: str) -> list[tuple[str, str]]:
    """Substitution pairs from optimal Levenshtein alignment (same normalize as CER)."""
    if not ref or not hyp:
        return []
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    pairs: list[tuple[str, str]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] != hyp[j - 1]:
                pairs.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def _categorize_failure(expected: str, actual: str) -> str:
    ref = normalize_arabic_text(expected)
    hyp = normalize_arabic_text(actual)
    if not hyp:
        return "empty"
    if ref == hyp:
        return "pass"
    ref_w = ref.split()
    hyp_w = hyp.split()
    hyp_set = Counter(hyp_w)
    missing = False
    for w in ref_w:
        if hyp_set[w] > 0:
            hyp_set[w] -= 1
        else:
            missing = True
            break
    if missing:
        return "missing_words"
    if sorted(ref_w) == sorted(hyp_w) and ref_w != hyp_w:
        return "word_order"
    if sorted(ref) == sorted(hyp) and ref != hyp and len(ref_w) <= 2:
        return "word_order"
    return "substitution"


def _compare_batch_vs_individual(
    img,
    best: dict,
    pad: int,
    reader,
    min_side: int,
) -> tuple[str, str, bool]:
    batch_labels = [
        ("firstName", True),
        ("lastName", True),
        ("address", True),
        ("dob", False),
        ("serial", False),
    ]
    labeled = [
        (lab, eid.crop_xyxy(img, best[lab][0], pad))
        for lab, _ in batch_labels
        if lab in best
    ]
    batched = ena.ocr_fields_batch_easyocr(labeled, reader, min_side=min_side)
    b_first = batched.get("firstName", "")
    b_last = batched.get("lastName", "")
    batch_full = f"{b_first} {b_last}".strip()

    i_first = ""
    i_last = ""
    if "firstName" in best:
        cr = eid.crop_xyxy(img, best["firstName"][0], pad)
        i_first = ena.ocr_text_field_easyocr(cr, reader, min_side=min_side)
    if "lastName" in best:
        cr = eid.crop_xyxy(img, best["lastName"][0], pad)
        i_last = ena.ocr_text_field_easyocr(cr, reader, min_side=min_side)
    ind_full = f"{i_first} {i_last}".strip()
    differs = normalize_arabic_text(batch_full) != normalize_arabic_text(ind_full)
    return batch_full, ind_full, differs


def _run_rows(
  field_yolo,
  reader,
  device: str,
  cases: list,
) -> list[NameRow]:
    rows: list[NameRow] = []
    cfg = ExtractConfig(image=ROOT / "x.jpg", quiet=True, fast_mode=True, engine="easyocr")
    engine, tess_langs, _ = _init_ocr(cfg, device)
    fw = DEFAULT_FIELD_WEIGHTS

    for case in cases:
        front = case["front"]
        gt = case["ground_truth"]
        source = str(gt.get("source") or "")
        exp = _expected_name(gt)
        if not exp:
            continue

        img = cv2.imread(str(front))
        if img is None:
            continue
        img = eid.resize_for_speed(img, max_side=880)
        r = field_yolo.predict(source=img, conf=0.25, device=device, imgsz=480, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            rows.append(
                NameRow(
                    image=front.name,
                    source=source,
                    expected=exp,
                    actual="",
                    first_name="",
                    last_name="",
                    cer=1.0,
                    passed=False,
                    category="empty",
                    missing_labels=["firstName", "lastName", "address", "nid"],
                )
            )
            continue

        id_to_name = eid.load_class_names()
        best = eid.best_boxes_by_label(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int),
            r.boxes.conf.cpu().numpy(),
            id_to_name,
        )
        missing = sorted({"firstName", "lastName", "address", "nid"} - set(best))

        row = extract_front(
            front,
            cfg,
            device=device,
            engine=engine,
            tess_langs=tess_langs,
            easyocr_reader=reader,
            dw=None,
        )
        act = row.get("full_name") or ""
        c = cer(exp, act)
        passed = exact_match(exp, act, field="name") or c <= 0.15
        cat = _categorize_failure(exp, act) if not passed else "pass"

        batch_full, ind_full, differs = "", "", False
        if "firstName" in best or "lastName" in best:
            batch_full, ind_full, differs = _compare_batch_vs_individual(
                img, best, pad=6, reader=reader, min_side=120
            )

        rows.append(
            NameRow(
                image=front.name,
                source=source,
                expected=exp,
                actual=act,
                first_name=row.get("first_name", ""),
                last_name=row.get("last_name", ""),
                cer=c,
                passed=passed,
                category=cat,
                batch_full=batch_full,
                individual_full=ind_full,
                batch_differs=differs,
                missing_labels=missing,
            )
        )
    return rows


def _confusion_for_rows(rows: list[NameRow]) -> list[tuple[str, str, int]]:
    c: Counter[tuple[str, str]] = Counter()
    for r in rows:
        if r.passed:
            continue
        ref = normalize_arabic_text(r.expected)
        hyp = normalize_arabic_text(r.actual)
        if not ref or not hyp:
            continue
        c.update(_levenshtein_backtrace_substitutions(ref, hyp))
    return [(a, b, n) for (a, b), n in c.most_common(30)]


def _category_counts(rows: list[NameRow]) -> dict[str, int]:
    failed = [r for r in rows if not r.passed]
    counts = Counter(r.category for r in failed)
    return dict(counts)


def _save_worst_crops(
    rows: list[NameRow],
    field_yolo,
    device: str,
    limit: int = 5,
) -> list[str]:
    held = [r for r in rows if r.source in HELD_OUT_SOURCES and not r.passed]
    held.sort(key=lambda r: r.cer, reverse=True)
    saved: list[str] = []
    CROP_DIR.mkdir(parents=True, exist_ok=True)

    for r in held[:limit]:
        case = next(c for c in discover_test_cases(ROOT / "test_data" / "id_cards") if c["front"].name == r.image)
        img = cv2.imread(str(case["front"]))
        if img is None:
            continue
        img = eid.resize_for_speed(img, max_side=880)
        pred = field_yolo.predict(source=img, conf=0.25, device=device, imgsz=480, verbose=False)[0]
        if pred.boxes is None:
            continue
        id_to_name = eid.load_class_names()
        best = eid.best_boxes_by_label(
            pred.boxes.xyxy.cpu().numpy(),
            pred.boxes.cls.cpu().numpy().astype(int),
            pred.boxes.conf.cpu().numpy(),
            id_to_name,
        )
        stem = Path(r.image).stem
        sub = CROP_DIR / stem
        sub.mkdir(parents=True, exist_ok=True)
        meta = {
            "expected": r.expected,
            "actual": r.actual,
            "cer": r.cer,
            "category": r.category,
            "batch_full": r.batch_full,
            "individual_full": r.individual_full,
            "batch_differs": r.batch_differs,
        }
        (sub / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        for lab in ("firstName", "lastName"):
            if lab not in best:
                continue
            cr = eid.crop_xyxy(img, best[lab][0], 6)
            up = eid.upscale_crop(cr, min_side=120)
            p = sub / f"{lab}.png"
            cv2.imwrite(str(p), cr)
            cv2.imwrite(str(sub / f"{lab}_upscaled.png"), up)
            saved.append(str(p))
        # composite strip preview
        if "firstName" in best or "lastName" in best:
            labeled = [(lab, eid.crop_xyxy(img, best[lab][0], 6)) for lab in ("firstName", "lastName") if lab in best]
            if labeled:
                max_w = max(c.shape[1] for _, c in labeled)
                parts = []
                for _, c in labeled:
                    h, w = c.shape[:2]
                    if w < max_w:
                        c = __import__("numpy").hstack([c, __import__("numpy").full((h, max_w - w, 3), 255, dtype="uint8")])
                    parts.append(c)
                strip = __import__("numpy").vstack(parts)
                cv2.imwrite(str(sub / "name_strip_preview.png"), strip)
    return saved


def _md_table_confusion(conf: list[tuple[str, str, int]]) -> list[str]:
    lines = ["| Expected | Read as | Count |", "|----------|---------|-------|"]
    for a, b, n in conf:
        lines.append(f"| {a} | {b} | {n} |")
    return lines


def _md_mismatch_table(rows: list[NameRow], title: str) -> list[str]:
    lines = [f"### {title}", "", "| Image | Expected | Actual | CER | Category |", "|-------|----------|--------|-----|----------|"]
    for r in sorted(rows, key=lambda x: (-x.cer, x.image)):
        if r.passed:
            continue
        lines.append(
            f"| `{r.image}` | {r.expected} | {r.actual or '(empty)'} | {r.cer:.3f} | {r.category} |"
        )
    if not any(not r.passed for r in rows):
        lines.append("| _all passed_ | | | | |")
    lines.append("")
    return lines


def build_report(
    held_rows: list[NameRow],
    train_rows: list[NameRow],
    batch_diff_held: list[NameRow],
    batch_diff_train: list[NameRow],
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Name OCR Diagnosis Report",
        "",
        f"Generated: {ts}",
        "",
        "Pipeline: `extract_id_all.extract_front` fast_mode + EasyOCR batched strip "
        "(firstName, lastName, address, dob, serial). Field weights: `train_id_detectr_hyper`.",
        "",
        "Scoring field: `full_name` fallback `first_name` + `last_name` (see `tests/id_metrics.score_fields`).",
        "",
    ]

    for label, rows in (("Held-out (roboflow_valid + roboflow_test)", held_rows), ("roboflow_train", train_rows)):
        n = len(rows)
        passed = sum(1 for r in rows if r.passed)
        failed = n - passed
        lines.append(f"## {label}")
        lines.append("")
        lines.append(f"- Samples with expected name: **{n}**")
        lines.append(f"- Name pass rate: **{passed}/{n}** ({100*passed/n:.1f}%)" if n else "- No samples")
        lines.append(f"- Failures: **{failed}**")
        lines.append("")
        cats = _category_counts(rows)
        if failed:
            lines.append("### Failure category breakdown (failed samples only)")
            lines.append("")
            for cat in ("empty", "missing_words", "word_order", "substitution"):
                c = cats.get(cat, 0)
                lines.append(f"- **{cat}**: {c}/{failed} ({100*c/failed:.1f}%)")
            lines.append("")
        conf = _confusion_for_rows(rows)
        lines.append("### Character confusion table (name field only, Levenshtein-aligned)")
        lines.append("")
        lines.extend(_md_table_confusion(conf) if conf else ["_No substitutions in non-empty failures._", ""])
        lines.extend(_md_mismatch_table(rows, "Raw mismatch table"))

    lines.extend(
        [
            "## Step 2 — Current name OCR config",
            "",
            "| Setting | Value for name crops |",
            "|---------|----------------------|",
            "| Engine (harness fast_mode) | EasyOCR only (`field_engine=easyocr`) |",
            "| Tesseract (non-fast / mixed) | `--oem 3`, PSM 6/7/11/13, lang `ara` + `ara+eng`, multi-preprocess variants |",
            "| EasyOCR single-field | `paragraph=True`, upscale min_side≥120, max_side 520 |",
            "| EasyOCR batched strip | `paragraph=False`, crops stacked vertically, y-center band assignment, max_side 1200 |",
            "| Reading-order correction | **None** — raw engine output; `full_name = first_name + ' ' + last_name` |",
            "| RTL handling | No post-OCR bidi reorder; `normalize_arabic_text` strips bidi marks for scoring only |",
            "| Name in batched strip? | **Yes** — firstName + lastName are bands 1–2 in the composite strip |",
            "",
            "## Batched vs individual EasyOCR (name only)",
            "",
        ]
    )
    for label, diffs in (
        ("Held-out failures where batch ≠ individual", batch_diff_held),
        ("roboflow_train failures where batch ≠ individual", batch_diff_train),
    ):
        lines.append(f"### {label}: **{len(diffs)}**")
        lines.append("")
        for r in diffs[:10]:
            lines.append(f"- `{r.image}` batch=`{r.batch_full}` individual=`{r.individual_full}`")
        lines.append("")

    lines.extend(
        [
            "## Step 4 — Summary",
            "",
            "See `runs/diagnose_name/worst_crops/` for 5 worst held-out name failure crops.",
            "",
        ]
    )
    return "\n".join(lines)


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
    fw = DEFAULT_FIELD_WEIGHTS.expanduser().resolve()
    field_yolo = YOLO(str(fw))

    cases = discover_test_cases(ROOT / "test_data" / "id_cards")
    held_cases = [c for c in cases if str(c["ground_truth"].get("source") or "") in HELD_OUT_SOURCES]
    train_cases = [c for c in cases if str(c["ground_truth"].get("source") or "") == SOURCE_ROBOFLOW_TRAIN]

    held_rows = _run_rows(field_yolo, reader, device, held_cases)
    train_rows = _run_rows(field_yolo, reader, device, train_cases)

    batch_diff_held = [r for r in held_rows if not r.passed and r.batch_differs]
    batch_diff_train = [r for r in train_rows if not r.passed and r.batch_differs]

    _save_worst_crops(held_rows, field_yolo, device, limit=5)

    report = build_report(held_rows, train_rows, batch_diff_held, batch_diff_train)

    h_failed = [r for r in held_rows if not r.passed]
    h_cats = _category_counts(held_rows)
    hf = len(h_failed) or 1
    dom = max(
        ((k, h_cats.get(k, 0)) for k in ("substitution", "missing_words", "word_order", "empty")),
        key=lambda x: x[1],
    )
    report += f"""
### 1. Dominant failure mode (held-out)

Primary category: **{dom[0]}** ({h_cats.get(dom[0], 0)}/{len(h_failed)} failed samples, {100*h_cats.get(dom[0], 0)/hf:.1f}%).

### 2. Plausible config levers (diagnosis only — not implemented)

- **substitution-heavy** → Arabic charset allowlist / Tesseract whitelist / stronger upscale (serial analogue: `--serial-charset-restrict`)
- **missing_words** → split firstName/lastName handling, compound-name token rules, or per-field OCR instead of merged full_name scoring
- **word_order** → explicit RTL reading-order sort on EasyOCR boxes before join; separate first/last fields
- **empty** → detection/crop issue (check missing_labels on those rows)

### 3. Batched strip risk for name

Held-out name failures where batched full name ≠ individual EasyOCR: **{len(batch_diff_held)}/{len(h_failed)}**.
{"**Flag:** batch strip may be contributing to name errors on some samples." if batch_diff_held else "**Low signal:** batch matches individual on all held-out name failures in this run."}
"""
    out = OUT_DIR / "report.md"
    out.write_text(report, encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Crops under {CROP_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
