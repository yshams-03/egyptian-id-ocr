"""
Grow the labeled dataset: inventory → import unverified Roboflow fronts → prefill drafts.

  py -m tests.labeling.batch_grow --inventory-only
  py -m tests.labeling.batch_grow
  py -m tests.labeling.batch_grow --prefill-only
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.labeling.import_roboflow import DEFAULT_OUT, import_all_unverified
from tests.labeling.inventory import build_inventory, print_inventory_report
from tests.labeling.prefill import run_prefill, print_summary

DATA_DIR = ROOT / "test_data" / "id_cards"


def ensure_lexicon() -> None:
    lex_path = ROOT / "scripts" / "lexicon" / "egyptian_lexicon.json"
    if not lex_path.is_file():
        print("Building lexicon from train-source ground truth…")
        import subprocess

        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "lexicon" / "build_lexicon.py")])


def load_models():
    reader = field_yolo = digit_yolo = None
    try:
        import export_id_to_excel as eid
        import torch

        fw = ROOT / "runs" / "train_id_detectr_hyper" / "weights" / "best.pt"
        if fw.is_file():
            field_yolo = eid.get_yolo(fw)
        dw = ROOT / "runs" / "train_arabic_numbers_v2" / "weights" / "best.pt"
        if dw.is_file():
            digit_yolo = eid.get_yolo(dw)
        import easyocr

        reader = easyocr.Reader(["ar", "en"], gpu=torch.cuda.is_available(), verbose=False)
    except Exception as ex:
        print(f"Warning: partial model load ({ex})")
    return reader, field_yolo, digit_yolo


def estimate_review_minutes(n: int, *, cards_per_session: int = 33, session_minutes: float = 90.0) -> float:
    """Rough estimate from prior ~33-card review session (~90 min)."""
    if n <= 0:
        return 0.0
    return (n / cards_per_session) * session_minutes


def print_final_report(
    report,
    import_results: list | None,
    prefill_results: list | None,
) -> None:
    print("\n" + "=" * 60)
    print("FINAL BATCH REPORT")
    print("=" * 60)
    print(f"Roboflow front images total:     {len(report.roboflow_fronts)}")
    print(f"Already verified (skipped):        {len(report.verified)}")
    print(f"Existing drafts (left as-is):    {len(report.draft)}")
    if import_results is not None:
        imported = sum(1 for r in import_results if r.get("status") == "imported")
        existed = sum(1 for r in import_results if r.get("status") == "exists")
        print(f"Newly imported this run:         {imported}")
        print(f"Already on disk (import skip):   {existed}")
    if prefill_results is not None:
        by_source = Counter(r.get("source", "missing") for r in prefill_results)
        by_pri = Counter(r.get("review_priority", "low") for r in prefill_results)
        print(f"New draft JSON written:          {len(prefill_results)}")
        print(f"  by source: {dict(by_source)}")
        print(f"  review priority: {dict(by_pri)} (review_app default: fast-confirm / low first)")
        est = estimate_review_minutes(len(prefill_results))
        print(f"\nEstimated review time: ~{est:.0f} min ({est/60:.1f} h) at ~90 min / 33 cards")
    print("\nNext: py -m tests.labeling.review_app")
    print("  Queue default: fast-confirm first (low priority). Toggle to needs-work for boxing session.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Import + prefill all unlabeled real front cards")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--import-only", action="store_true")
    parser.add_argument("--prefill-only", action="store_true")
    parser.add_argument("--require-boxes", action="store_true")
    parser.add_argument("--symlink", action="store_true")
    parser.add_argument(
        "--re-draft",
        action="store_true",
        help="Re-run OCR on images that already have draft JSON.",
    )
    parser.add_argument(
        "--no-local-engine-select-name",
        action="store_true",
        help="Disable EasyOCR vs Tesseract(ara) scoring on firstName/lastName (default: enabled).",
    )
    args = parser.parse_args()
    local_engine_select_name = not args.no_local_engine_select_name

    report = build_inventory(args.data_dir)
    print_inventory_report(report)

    if args.inventory_only:
        return 0

    import_results = None
    prefill_results = None

    if not args.prefill_only:
        print("\n--- Importing unverified Roboflow front images ---")
        import_results = import_all_unverified(
            args.out_dir,
            data_dir=args.data_dir,
            require_boxes=args.require_boxes,
            use_symlink=args.symlink,
        )
        n_new = sum(1 for r in import_results if r.get("status") == "imported")
        print(f"Import complete: {n_new} new, {len(import_results) - n_new} already present")

    if not args.import_only:
        print("\n--- Running front OCR prefill (draft JSON, needs_review) ---")
        if local_engine_select_name:
            ensure_lexicon()
        reader, field_yolo, digit_yolo = load_models()
        prefill_results = run_prefill(
            args.data_dir,
            skip_drafts=not args.re_draft,
            only_subdir="real",
            field_yolo=field_yolo,
            easyocr_reader=reader,
            digit_yolo=digit_yolo,
            local_engine_select_name=local_engine_select_name,
        )
        print_summary(prefill_results)

    print_final_report(report, import_results, prefill_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
