"""
Run trained YOLO weights on your own ID photo (or any image).

Examples:
  python predict_my_id.py path\\to\\my_id.jpg
  python predict_my_id.py "path\\with spaces\\my id.jpg"
  python predict_my_id.py my_id.jpg --model detectr_hyper
  python predict_my_id.py my_id.jpg --model national --conf 0.35
  python predict_my_id.py my_id.jpg --weights "C:\\path\\to\\custom.pt"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent
RUNS = BASE / "runs"

PRESETS = {
    "detectr": RUNS / "train_id_detectr_v1" / "weights" / "best.pt",
    "detectr_hyper": RUNS / "train_id_detectr_hyper" / "weights" / "best.pt",
    "arabic": RUNS / "train_arabic_numbers_v2" / "weights" / "best.pt",
    "national": RUNS / "train_national_id_v7" / "weights" / "best.pt",
}

DATA_YAML = {
    "detectr": BASE / "egyptian_id_detectr" / "content" / "Egyptian-ID-Detectr-3" / "data.yaml",
    "detectr_hyper": BASE / "egyptian_id_detectr" / "content" / "Egyptian-ID-Detectr-3" / "data.yaml",
    "arabic": BASE / "arabic_numbers" / "content" / "arabic-numbers-2" / "data.yaml",
    "national": BASE / "national_id" / "content" / "National-ID-7" / "data.yaml",
}


def load_names(preset: str) -> dict[int, str]:
    path = DATA_YAML.get(preset)
    if not path or not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data.get("names") or []
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Test trained ID models on your image.")
    parser.add_argument("image", type=Path, help="Path to your ID photo (jpg/png/webp).")
    parser.add_argument(
        "--model",
        choices=list(PRESETS.keys()),
        default="detectr_hyper",
        help="Which trained checkpoint to use (default: detectr_hyper — Egyptian ID fields).",
    )
    parser.add_argument("--weights", type=Path, default=None, help="Override: path to any .pt file.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--device", type=str, default="0", help="0 for first GPU, cpu for CPU.")
    parser.add_argument(
        "--project",
        type=Path,
        default=BASE / "runs" / "predict_custom",
        help="Folder to save annotated images.",
    )
    parser.add_argument("--name", type=str, default="exp", help="Subfolder name under --project.")
    args = parser.parse_args()

    image = args.image.expanduser().resolve()
    if not image.is_file():
        raise SystemExit(f"Image not found: {image}")

    weights = args.weights.expanduser().resolve() if args.weights else PRESETS[args.model]
    if not weights.is_file():
        raise SystemExit(
            f"Weights not found: {weights}\nTrain first with run_egyptian_id_ocr.py or pass --weights."
        )

    import torch
    from ultralytics import YOLO

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device = "cpu"

    names = load_names(args.model)
    model = YOLO(str(weights))
    results = model.predict(
        source=str(image),
        conf=args.conf,
        save=True,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        device=device,
    )

    r = results[0]
    print(f"\nSaved to: {args.project / args.name}")
    if r.boxes is None or len(r.boxes) == 0:
        print("No detections above confidence threshold. Try lowering --conf.")
        return

    print(f"Detections ({len(r.boxes)}):\n")
    xyxy = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()
    for i in range(len(r.boxes)):
        c = int(cls[i])
        label = names.get(c, str(c))
        print(f"  {label:24s} conf={conf[i]:.3f}  box_xyxy={xyxy[i].round(1).tolist()}")


if __name__ == "__main__":
    main()
