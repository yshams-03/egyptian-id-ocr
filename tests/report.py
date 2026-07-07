"""Generate markdown and HTML reports from extraction test results."""
from __future__ import annotations

import html
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from tests.id_metrics import (
    SampleResult,
    aggregate_cer,
    aggregate_field_accuracy,
    aggregate_stage_pass_rate,
    held_out_results,
    serial_full_match,
    serial_suffix_match,
)

_NID_RE = re.compile(r"\b\d{14}\b")
REAL_HELD_OUT_SOURCES = {"roboflow_valid", "roboflow_test"}
REAL_TRAIN_SOURCE = "roboflow_train"
SYNTHETIC_SOURCE = "synthetic_generated"
FIELDS = (
    "name",
    "address",
    "national_id",
    "dob",
    "serial",
    "job",
    "religion",
    "expiry_date",
    "back_nid",
    "decoded_birth_date",
    "decoded_governorate",
    "decoded_gender",
)


def redact_pii(text: str) -> str:
    return _NID_RE.sub("[NID-REDACTED]", text)


def _confusion_table(results: list[SampleResult]) -> list[tuple[str, str, int]]:
    c: Counter[tuple[str, str]] = Counter()
    for r in results:
        c.update(r.confusion_pairs())
    return [(a, b, n) for (a, b), n in c.most_common(20)]


def _pass_rate(results: list[SampleResult]) -> float | None:
    if not results:
        return None
    return sum(1 for r in results if r.passed) / len(results)


def _fmt_pct(rate: float | None) -> str:
    return "N/A" if rate is None else f"{100 * rate:.1f}%"


def _real_all(results: list[SampleResult]) -> list[SampleResult]:
    return [r for r in results if r.source != SYNTHETIC_SOURCE]


def _real_train(results: list[SampleResult]) -> list[SampleResult]:
    return [r for r in results if r.source == REAL_TRAIN_SOURCE]


def _synthetic(results: list[SampleResult]) -> list[SampleResult]:
    return [r for r in results if r.source == SYNTHETIC_SOURCE]


def _field_keys_for_no_box(field: str) -> set[str]:
    mapping = {
        "name": {"firstName", "lastName"},
        "address": {"address"},
        "national_id": {"nid"},
        "dob": {"dob"},
        "serial": {"serial"},
    }
    return mapping.get(field, set())


def _no_box_for_field(result: SampleResult, field: str) -> bool:
    required = _field_keys_for_no_box(field)
    if not required:
        return False
    return bool(set(result.stages.missing_detection_labels or []) & required)


def _failure_mode_counts(results: list[SampleResult]) -> tuple[int, int]:
    no_box = 0
    bad_ocr = 0
    for r in results:
        failed_fields = [f for f in r.fields if not f.skipped and not f.passed]
        if failed_fields and any(_no_box_for_field(r, f.field) for f in failed_fields):
            no_box += 1
        elif failed_fields or r.dob_nid_mismatch or r.nid_validation_errors or r.extraction_error:
            bad_ocr += 1
    return no_box, bad_ocr


def _worst(results: list[SampleResult], limit: int = 5) -> list[SampleResult]:
    failed = [r for r in results if not r.passed]
    failed.sort(
        key=lambda r: (
            len(r.stages.missing_detection_labels or []),
            sum(1 for f in r.fields if not f.skipped and not f.passed),
            max((f.cer for f in r.fields), default=0),
        ),
        reverse=True,
    )
    return failed[:limit]


def _append_field_table(lines: list[str], title: str, results: list[SampleResult]) -> None:
    acc = aggregate_field_accuracy(results)
    cers = aggregate_cer(results)
    lines.extend([f"### {title}", "", "| Field | Pass % | Mean CER |", "|-------|--------|----------|"])
    for field in FIELDS:
        if field not in acc and field not in cers:
            continue
        pct = 100 * acc.get(field, 0)
        cer_val = cers.get(field, 0)
        cer_s = f"{cer_val:.3f}" if field in cers else "—"
        lines.append(f"| {field} | {pct:.1f}% | {cer_s} |")
    lines.append("")
    serial_rows = []
    for r in results:
        for f in r.fields:
            if f.field == "serial" and not f.skipped:
                serial_rows.append(f)
                break
    if serial_rows:
        full = sum(1 for f in serial_rows if serial_full_match(f.expected, f.actual)) / len(serial_rows)
        suffix = sum(1 for f in serial_rows if serial_suffix_match(f.expected, f.actual)) / len(serial_rows)
        lines.append(f"- Serial full-match accuracy (pass criterion): {100 * full:.1f}%")
        lines.append(
            f"- Serial suffix-match (OCR tolerance, not official): {100 * suffix:.1f}%"
        )
        lines.append("")


def _append_worst_group(lines: list[str], title: str, results: list[SampleResult], *, redact: bool) -> None:
    def t(s: str) -> str:
        return redact_pii(s) if redact else s

    lines.extend([f"### {title}", ""])
    worst = _worst(results, limit=5)
    if not worst:
        lines.append("_No failed samples in this group._")
        lines.append("")
        return
    for r in worst:
        lines.append(f"#### `{Path(r.image_path).name}`")
        if r.extraction_error:
            lines.append(f"- Error: {t(r.extraction_error)}")
        if r.stages.missing_detection_labels:
            lines.append(f"- Missing YOLO labels: {r.stages.missing_detection_labels}")
        for f in r.fields:
            if f.skipped:
                continue
            status = "OK" if f.passed else "FAIL"
            if not f.passed and _no_box_for_field(r, f.field):
                lines.append(
                    f"- **{f.field}** [{status}] [NO BOX DETECTED]\n"
                    f"  - expected: `{t(f.expected)}`\n"
                    f"  - actual: `{t(f.actual)}`"
                )
            else:
                lines.append(
                    f"- **{f.field}** [{status}] CER={f.cer:.3f}\n"
                    f"  - expected: `{t(f.expected)}`\n"
                    f"  - actual: `{t(f.actual)}`"
                )
        lines.append("")


def build_markdown_report(
    results: list[SampleResult],
    *,
    title: str = "ID Extraction Test Report",
    redact: bool = False,
) -> str:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    stages = aggregate_stage_pass_rate(results)
    held_out = held_out_results(results)
    real_all = _real_all(results)
    real_train = _real_train(results)
    synthetic = _synthetic(results)

    def t(s: str) -> str:
        return redact_pii(s) if redact else s

    lines = [
        f"# {title}",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "> **Privacy:** Reports may contain PII. Keep under `runs/test/` (gitignored). "
        "Use `--redact` when sharing outside your machine.",
        "",
        "## Summary",
        "",
        f"- **Samples:** {total}",
        f"- **Real (held-out only):** {sum(1 for r in held_out if r.passed)}/{len(held_out)} passed ({_fmt_pct(_pass_rate(held_out))})",
        f"- **Real (all sources):** {sum(1 for r in real_all if r.passed)}/{len(real_all)} passed ({_fmt_pct(_pass_rate(real_all))})",
        f"- **Synthetic:** {sum(1 for r in synthetic if r.passed)}/{len(synthetic)} passed ({_fmt_pct(_pass_rate(synthetic))})",
        f"- **Blended:** {passed}/{total} passed ({_fmt_pct(_pass_rate(results))})",
        "",
        "## Failure-mode summary",
        "",
    ]

    for label, group in (
        ("REAL (held-out)", held_out),
        ("REAL (train-source)", real_train),
        ("REAL (all)", real_all),
        ("SYNTHETIC", synthetic),
    ):
        no_box, bad_ocr = _failure_mode_counts(group)
        lines.append(f"- **{label}:** no-box-detected={no_box}, box-found/OCR-wrong={bad_ocr}")

    lines.extend(["", "## Per-stage pass rate", "", "| Stage | Pass % |", "|-------|--------|"])
    for stage, rate in stages.items():
        lines.append(f"| {stage} | {100 * rate:.1f}% |")

    lines.extend(["", "## Per-field accuracy", ""])
    _append_field_table(lines, "REAL (held-out)", held_out)
    _append_field_table(lines, "REAL (train-source)", real_train)
    _append_field_table(lines, "REAL (all sources)", real_all)
    _append_field_table(lines, "SYNTHETIC", synthetic)
    _append_field_table(lines, "BLENDED", results)

    dob_mismatches = [r for r in results if r.dob_nid_mismatch]
    lines.extend(["", "## DOB / NID cross-validation mismatches", ""])
    if not dob_mismatches:
        lines.append("_No DOB vs NID decode mismatches._")
    else:
        for r in dob_mismatches:
            d = r.dob_nid
            if not d:
                continue
            lines.append(f"### `{Path(r.image_path).name}`")
            lines.append(f"- printed DOB: `{t(d.printed_dob)}`")
            lines.append(f"- NID decode DOB: `{t(d.decoded_from_nid)}`")
            lines.append(f"- message: {t(d.message)}")
            lines.append("")

    lines.extend(["", "## Worst samples (expected vs actual)", ""])
    _append_worst_group(lines, "Worst samples — REAL (held-out)", held_out, redact=redact)
    _append_worst_group(lines, "Worst samples — REAL (train-source)", real_train, redact=redact)
    _append_worst_group(lines, "Worst samples — SYNTHETIC", synthetic, redact=redact)

    conf = _confusion_table(results)
    lines.extend(["", "## Common character confusions (Arabic)", ""])
    if conf:
        lines.append("| Expected | Read as | Count |")
        lines.append("|----------|---------|-------|")
        for a, b, n in conf:
            lines.append(f"| {a} | {b} | {n} |")
    else:
        lines.append("_No character-level confusions recorded._")

    return "\n".join(lines)


def write_reports(
    results: list[SampleResult],
    out_dir: Path | None = None,
    *,
    redact: bool = False,
    timestamped: bool = True,
) -> tuple[Path, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if out_dir is None:
        base = Path("runs/test")
        out_dir = base / f"report_{ts}" if timestamped else base / "latest"
    else:
        out_dir = out_dir.expanduser().resolve()
        if timestamped and out_dir.name == "latest":
            out_dir = out_dir.parent / f"report_{ts}"

    out_dir.mkdir(parents=True, exist_ok=True)
    md = build_markdown_report(results, redact=redact)
    md_path = out_dir / "report.md"
    md_path.write_text(md, encoding="utf-8")

    if timestamped:
        flat = Path("runs/test") / f"report_{ts}.md"
        flat.parent.mkdir(parents=True, exist_ok=True)
        flat.write_text(md, encoding="utf-8")

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    acc = aggregate_field_accuracy(results)
    rows_html = "".join(
        f"<tr><td>{html.escape(field)}</td><td>{100*pct:.1f}%</td></tr>"
        for field, pct in acc.items()
    )
    html_body = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>ID Extraction Report</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;max-width:960px}}
table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:.4rem .8rem}}
</style></head><body>
<h1>ID Extraction Test Report</h1>
<p><strong>{passed}/{total}</strong> blended passed</p>
<h2>Per-field accuracy</h2>
<table><tr><th>Field</th><th>Pass %</th></tr>{rows_html}</table>
<pre>{html.escape(md)}</pre>
</body></html>"""
    html_path = out_dir / "report.html"
    html_path.write_text(html_body, encoding="utf-8")
    return md_path, html_path
