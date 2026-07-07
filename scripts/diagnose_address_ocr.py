#!/usr/bin/env python
"""Diagnosis-only: address OCR failure modes (held-out + roboflow_train)."""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import export_id_to_excel as eid
import extract_name_address as ena
from extract_id_all import DEFAULT_FIELD_WEIGHTS, NAME_OCR_MAX_SIDE, NAME_OCR_MIN_SIDE
from tests.ground_truth import discover_test_cases
from tests.id_metrics import cer, exact_match, normalize_arabic_text
from tests.labeling.sources import HELD_OUT_SOURCES, SOURCE_ROBOFLOW_TRAIN

OUT_DIR = ROOT / "runs" / "diagnose_address"


@dataclass
class AddrRow:
    image: str
    source: str
    expected: str
    actual: str
    pipeline_cer: float
    passed: bool
    category: str
    batch_text: str
    individual_text: str
    batch_cer: float
    individual_cer: float
    batch_worse: str


def _levenshtein_backtrace_substitutions(ref: str, hyp: str) -> list[tuple[str, str]]:
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
    if sorted(ref) == sorted(hyp) and ref != hyp and len(ref_w) <= 3:
        return "word_order"
    return "substitution"


def _batch_spacer_for(img, best, lab: str, pad: int, min_side: int):
    if lab not in best:
        return None
    cr = eid.crop_xyxy(img, best[lab][0], pad)
    if cr.size == 0:
        return None
    up = eid.upscale_crop(cr, min_side=min_side)
    return eid.resize_for_speed(up, max_side=480)


def _current_batch_address(img, best, pad, reader, min_side) -> str:
    spacers = [
        sp
        for lab in ("firstName", "lastName")
        if (sp := _batch_spacer_for(img, best, lab, pad, min_side)) is not None
    ]
    labeled = [
        (lab, eid.crop_xyxy(img, best[lab][0], pad))
        for lab in ("address", "dob", "serial")
        if lab in best
    ]
    batched = ena.ocr_fields_batch_easyocr(
        labeled, reader, min_side=min_side, leading_spacers=spacers or None
    )
    return eid.clean_address_text(batched.get("address", ""), strip_digits=False)


def _individual_address(img, best, pad, reader, min_side=120, max_side=520) -> str:
    if "address" not in best:
        return ""
    cr = eid.crop_xyxy(img, best["address"][0], pad)
    raw = ena.ocr_text_field_easyocr(cr, reader, min_side=min_side, max_side=max_side)
    return eid.clean_address_text(raw, strip_digits=False)


def _run_rows(cases, reader, field_yolo, device) -> list[AddrRow]:
    from extract_id_all import ExtractConfig, _init_ocr, extract_front

    cfg = ExtractConfig(image=ROOT / "x.jpg", quiet=True, fast_mode=True, engine="easyocr")
    engine, tess_langs, _ = _init_ocr(cfg, device)
    pad = 6
    min_side = 120
    rows: list[AddrRow] = []

    for case in cases:
        gt = case["ground_truth"]
        source = str(gt.get("source") or "")
        exp = (gt.get("address") or "").strip()
        if not exp:
            continue

        row = extract_front(
            case["front"],
            cfg,
            device=device,
            engine=engine,
            tess_langs=tess_langs,
            easyocr_reader=reader,
            dw=None,
        )
        act = eid.clean_address_text(row.get("address", ""), strip_digits=False)
        pc = cer(exp, act)
        passed = exact_match(exp, act, field="address") or pc <= 0.15

        img = cv2.imread(str(case["front"]))
        if img is None:
            continue
        img = eid.resize_for_speed(img, max_side=880)
        pred = field_yolo.predict(source=img, conf=0.25, device=device, imgsz=480, verbose=False)[0]
        if pred.boxes is None:
            batch_t = ind_t = ""
        else:
            best = eid.best_boxes_by_label(
                pred.boxes.xyxy.cpu().numpy(),
                pred.boxes.cls.cpu().numpy().astype(int),
                pred.boxes.conf.cpu().numpy(),
                eid.load_class_names(),
            )
            batch_t = _current_batch_address(img, best, pad, reader, min_side)
            ind_t = _individual_address(img, best, pad, reader)

        bc, ic = cer(exp, batch_t), cer(exp, ind_t)
        if bc > ic + 0.01:
            bw = "yes"
        elif ic > bc + 0.01:
            bw = "no"
        else:
            bw = "tie"

        rows.append(
            AddrRow(
                image=case["front"].name,
                source=source,
                expected=exp,
                actual=act,
                pipeline_cer=pc,
                passed=passed,
                category=_categorize_failure(exp, act) if not passed else "pass",
                batch_text=batch_t,
                individual_text=ind_t,
                batch_cer=bc,
                individual_cer=ic,
                batch_worse=bw,
            )
        )
    return rows


def _confusion(rows: list[AddrRow]) -> list[tuple[str, str, int]]:
    c: Counter[tuple[str, str]] = Counter()
    for r in rows:
        if r.passed:
            continue
        ref = normalize_arabic_text(r.expected)
        hyp = normalize_arabic_text(r.actual)
        if ref and hyp:
            c.update(_levenshtein_backtrace_substitutions(ref, hyp))
    return [(a, b, n) for (a, b), n in c.most_common(25)]


def _category_counts(rows: list[AddrRow]) -> dict[str, int]:
    failed = [r for r in rows if not r.passed]
    return dict(Counter(r.category for r in failed))


def _section_report(title: str, rows: list[AddrRow]) -> list[str]:
    n = len(rows)
    passed = sum(1 for r in rows if r.passed)
    failed = n - passed
    lines = [
        f"## {title}",
        "",
        f"- Samples: **{n}** | pass **{passed}/{n}** ({100*passed/n:.1f}%)" if n else f"## {title}",
    ]
    if failed:
        cats = _category_counts(rows)
        lines.append(f"- Failures: **{failed}**")
        lines.append("")
        lines.append("### Failure categories (failed only)")
        for cat in ("empty", "missing_words", "word_order", "substitution"):
            c = cats.get(cat, 0)
            lines.append(f"- **{cat}**: {c}/{failed} ({100*c/failed:.1f}%)")
    lines.append("")
    conf = _confusion(rows)
    lines.append("### Character confusion (address, Levenshtein-aligned)")
    lines.append("")
    if conf:
        lines.append("| Expected | Read as | Count |")
        lines.append("|----------|---------|-------|")
        for a, b, cnt in conf:
            lines.append(f"| {a} | {b} | {cnt} |")
    else:
        lines.append("_No substitutions in non-empty failures._")
    lines.append("")
    lines.append("### Raw mismatch table (failures)")
    lines.append("")
    lines.append("| Image | Expected | Actual (pipeline) | CER | Category |")
    lines.append("|-------|----------|-------------------|-----|----------|")
    for r in sorted([x for x in rows if not x.passed], key=lambda x: -x.pipeline_cer):
        lines.append(
            f"| `{r.image}` | {r.expected} | {r.actual or '(empty)'} | {r.pipeline_cer:.3f} | {r.category} |"
        )
    lines.append("")
    return lines


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
    field_yolo = YOLO(str(DEFAULT_FIELD_WEIGHTS))
    cases = discover_test_cases(ROOT / "test_data" / "id_cards")
    held = [c for c in cases if str(c["ground_truth"].get("source") or "") in HELD_OUT_SOURCES and (c["ground_truth"].get("address") or "").strip()]
    train = [c for c in cases if str(c["ground_truth"].get("source") or "") == SOURCE_ROBOFLOW_TRAIN and (c["ground_truth"].get("address") or "").strip()]

    held_rows = _run_rows(held, reader, field_yolo, device)
    train_rows = _run_rows(train, reader, field_yolo, device)

    h_fail = [r for r in held_rows if not r.passed]
    batch_diff = [r for r in h_fail if normalize_arabic_text(r.batch_text) != normalize_arabic_text(r.individual_text)]
    batch_worse_n = sum(1 for r in h_fail if r.batch_worse == "yes")
    ind_better_big = sum(1 for r in h_fail if r.individual_cer + 0.05 < r.batch_cer)

    lines = [
        "# Address OCR Diagnosis Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "Pipeline: address in batched strip (name spacers + address/dob/serial). "
        "Individual comparison uses `ocr_text_field_easyocr` paragraph mode 120/520.",
        "",
    ]
    lines.extend(_section_report("Held-out (roboflow_valid + roboflow_test)", held_rows))
    lines.extend(_section_report("roboflow_train", train_rows))

    lines.extend(
        [
            "## Batch vs individual (held-out failures)",
            "",
            f"- Failures: **{len(h_fail)}**",
            f"- Batch text ≠ individual text: **{len(batch_diff)}/{len(h_fail)}**",
            f"- Individual clearly better (batch CER > individual + 0.05): **{ind_better_big}/{len(h_fail)}**",
            f"- Batch worse (CER): **{batch_worse_n}/{len(h_fail)}**",
            "",
            "| Image | Pipeline CER | Batch CER | Individual CER | Batch worse? |",
            "|-------|--------------|-----------|----------------|--------------|",
        ]
    )
    for r in sorted(h_fail, key=lambda x: -x.pipeline_cer):
        lines.append(
            f"| `{r.image[:45]}` | {r.pipeline_cer:.3f} | {r.batch_cer:.3f} | {r.individual_cer:.3f} | {r.batch_worse} |"
        )
    lines.append("")
    lines.append("### Batch vs individual text pairs (held-out failures where they differ)")
    lines.append("")
    for r in batch_diff[:12]:
        lines.append(f"- `{r.image}`")
        lines.append(f"  - batch: `{r.batch_text}`")
        lines.append(f"  - individual: `{r.individual_text}`")
    lines.append("")

    out = OUT_DIR / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Held-out batch worse: {batch_worse_n}/{len(h_fail)}, individual better (+0.05): {ind_better_big}/{len(h_fail)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
