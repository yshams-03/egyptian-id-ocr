"""
Compare field-detection weight versions on the labeled test set.

Example:
  py -m tests.labeling.compare_weights \\
    --v1 runs/train_id_detectr_hyper/weights/best.pt \\
    --v2 runs/train_id_detectr_hyper_v2/weights/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.harness import run_dataset
from tests.id_metrics import aggregate_field_accuracy, aggregate_cer, aggregate_pass_rate_by_source, held_out_results


def _load_models(field_weights: Path):
    import export_id_to_excel as eid
    import torch
    import easyocr

    field_yolo = eid.get_yolo(field_weights)
    dw = ROOT / "runs" / "train_arabic_numbers_v2" / "weights" / "best.pt"
    digit_yolo = eid.get_yolo(dw) if dw.is_file() else None
    reader = easyocr.Reader(["ar", "en"], gpu=torch.cuda.is_available(), verbose=False)
    return reader, field_yolo, digit_yolo


def _summarize(results, label: str) -> dict:
    acc = aggregate_field_accuracy(results)
    cer = aggregate_cer(results)
    passed = sum(1 for r in results if r.passed)
    by_source = aggregate_pass_rate_by_source(results)
    held = held_out_results(results)
    ho_rate = sum(1 for r in held if r.passed) / len(held) if held else None
    return {
        "label": label,
        "passed": passed,
        "total": len(results),
        "accuracy": acc,
        "cer": cer,
        "by_source": by_source,
        "held_out_pass_rate": ho_rate,
        "held_out_n": len(held),
    }


def print_delta(v1: dict, v2: dict) -> None:
    print(f"\n{'='*60}")
    print(f"{'Metric':<22} {'v1':>12} {'v2':>12} {'delta':>12}")
    print("-" * 60)
    print(f"{'samples passed':<22} {v1['passed']}/{v1['total']:>8} {v2['passed']}/{v2['total']:>8} {v2['passed']-v1['passed']:>+12}")
    fields = sorted(set(v1["accuracy"]) | set(v2["accuracy"]))
    for f in fields:
        a1 = v1["accuracy"].get(f, 0) * 100
        a2 = v2["accuracy"].get(f, 0) * 100
        print(f"{f+' acc %':<22} {a1:>11.1f}% {a2:>11.1f}% {a2-a1:>+11.1f}%")
    for f in ("name", "address"):
        c1 = v1["cer"].get(f, 0) * 100
        c2 = v2["cer"].get(f, 0) * 100
        print(f"{f+' CER %':<22} {c1:>11.1f}% {c2:>11.1f}% {c2-c1:>+11.1f}%")
    print("\nHeld-out pass rate (roboflow_valid + roboflow_test):")
    for label, s in (("v1", v1), ("v2", v2)):
        if s.get("held_out_n"):
            print(f"  {label}: {s['held_out_pass_rate']*100:.1f}% ({s['held_out_n']} samples)")
        else:
            print(f"  {label}: n/a (no held-out tagged samples)")
    print("\nPass rate by source (v2):")
    for src, rate in sorted(v2.get("by_source", {}).items()):
        print(f"  {src}: {rate*100:.1f}%")
    print(f"\nNote: if test images were also added to train/, gains may be overfitting.")
    print("Check Ultralytics valid mAP from training for an honest generalization signal.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare v1 vs v2 field weights on test_data/id_cards")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "test_data" / "id_cards")
    parser.add_argument("--v1", type=Path, default=ROOT / "runs" / "train_id_detectr_hyper" / "weights" / "best.pt")
    parser.add_argument("--v2", type=Path, default=ROOT / "runs" / "train_id_detectr_hyper_v2" / "weights" / "best.pt")
    args = parser.parse_args()

    if not args.v1.is_file():
        print(f"v1 weights missing: {args.v1}")
        return 1
    if not args.v2.is_file():
        print(f"v2 weights missing: {args.v2} — train with: py run_egyptian_id_ocr.py --stage field_detection --name train_id_detectr_hyper_v2 --force")
        return 1

    r1, fy1, dy1 = _load_models(args.v1)
    results_v1 = run_dataset(
        args.data_dir,
        easyocr_reader=r1,
        field_yolo=fy1,
        digit_yolo=dy1,
    )
    s1 = _summarize(results_v1, "v1")

    r2, fy2, dy2 = _load_models(args.v2)
    results_v2 = run_dataset(
        args.data_dir,
        easyocr_reader=r2,
        field_yolo=fy2,
        digit_yolo=dy2,
    )
    s2 = _summarize(results_v2, "v2")
    print_delta(s1, s2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
