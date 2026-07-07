#!/usr/bin/env python
"""
CLI test runner for the national ID extraction pipeline.

Examples:
  py -m tests.run_suite --generate-template
  py -m tests.run_suite --generate-json-templates
  py -m tests.run_suite --data-dir test_data/id_cards
  py -m pytest tests/ -m "not slow" -q
  py -m pytest tests/ -m slow -q
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.harness import (  # noqa: E402
    csv_to_json_ground_truth,
    discover_image_pairs,
    generate_missing_templates,
    generate_template_csv,
    run_dataset,
)
from tests.report import write_reports  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="ID extraction test suite runner")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "test_data" / "id_cards",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Report directory (default: runs/test/report_<timestamp>/)",
    )
    parser.add_argument("--generate-template", action="store_true")
    parser.add_argument(
        "--generate-json-templates",
        action="store_true",
        help="Create empty .json per image (NID-decode prefill if national_id in CSV)",
    )
    parser.add_argument("--import-csv", type=Path, default=None)
    parser.add_argument("--engine", default="easyocr", choices=("easyocr", "mixed", "tesseract"))
    parser.add_argument("--no-fast", action="store_true")
    parser.add_argument("--auto-card-crop", action="store_true")
    parser.add_argument(
        "--redact",
        action="store_true",
        help="Redact 14-digit NIDs in report (safer if sharing)",
    )
    parser.add_argument(
        "--field-weights",
        type=Path,
        default=None,
        help="Override field YOLO weights (default: runs/train_id_detectr_hyper/weights/best.pt)",
    )
    parser.add_argument(
        "--no-serial-charset-restrict",
        action="store_true",
        help="Disable A-Z0-9 serial OCR allowlist (default: restrict enabled).",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()

    if args.generate_template:
        out = generate_template_csv(data_dir)
        print(f"Wrote CSV template: {out}")
        pairs = discover_image_pairs(data_dir)
        print(f"Found {len(pairs)} images ({sum(1 for _, g in pairs if g)} with JSON)")
        return 0

    if args.generate_json_templates:
        paths = generate_missing_templates(data_dir)
        print(f"Wrote {len(paths)} JSON template(s)")
        for p in paths[:5]:
            print(f"  {p}")
        return 0

    if args.import_csv:
        n = csv_to_json_ground_truth(args.import_csv, data_dir)
        print(f"Imported {n} ground truth JSON file(s)")
        return 0

    if not data_dir.is_dir():
        print(f"Create {data_dir} — see test_data/id_cards/README.md")
        return 1

    reader = field_yolo = digit_yolo = None
    try:
        import export_id_to_excel as eid
        import torch

        fw = args.field_weights or (ROOT / "runs" / "train_id_detectr_hyper" / "weights" / "best.pt")
        if fw.is_file():
            field_yolo = eid.get_yolo(fw)
            dw = ROOT / "runs" / "train_arabic_numbers_v2" / "weights" / "best.pt"
            digit_yolo = eid.get_yolo(dw) if dw.is_file() else None
        import easyocr

        reader = easyocr.Reader(
            ["ar", "en"], gpu=torch.cuda.is_available(), verbose=False
        )
    except Exception as ex:
        print(f"Warning: model warmup partial ({ex})")

    results = run_dataset(
        data_dir,
        fast_mode=not args.no_fast,
        engine=args.engine,
        require_ground_truth=True,
        auto_card_crop=args.auto_card_crop,
        serial_charset_restrict=not args.no_serial_charset_restrict,
        easyocr_reader=reader,
        field_yolo=field_yolo,
        digit_yolo=digit_yolo,
    )
    if not results:
        print("No labeled image+JSON pairs. Run --generate-json-templates first.")
        return 1

    md_path, html_path = write_reports(
        results,
        args.report,
        redact=args.redact,
        timestamped=True,
    )
    passed = sum(1 for r in results if r.passed)
    print(f"Results: {passed}/{len(results)} passed")
    print(f"Report: {md_path}")
    print(f"HTML:   {html_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
