"""
Run Egyptian ID OCR training pipeline locally with GPU.
Matches egyptian_id_ocr.ipynb using local dataset folders.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
RUNS = BASE / "runs"
DATASETS = {
    "egyptian_id_detectr": BASE / "egyptian_id_detectr" / "content" / "Egyptian-ID-Detectr-3" / "data.yaml",
    "arabic_numbers": BASE / "arabic_numbers" / "content" / "arabic-numbers-2" / "data.yaml",
    "national_id": BASE / "national_id" / "content" / "National-ID-7" / "data.yaml",
}

# Conservative default; override with --batch or --batch -1 for Ultralytics auto-batch.
STAGES = [
    ("egyptian_id_detectr", "yolov8n.pt", "train_id_detectr_v1", 10, 16, None),
    ("egyptian_id_detectr", "yolov8n.pt", "train_id_detectr_hyper", 30, 16, 0.005),
    ("arabic_numbers", "yolov8s.pt", "train_arabic_numbers_v2", 20, 16, None),
    ("national_id", "yolov8n.pt", "train_national_id_v7", 40, 16, None),
]
FIELD_DETECTION_DEFAULT_RUN = "train_id_detectr_hyper"

STAGE_ALIASES = {
    "field_detection": "egyptian_id_detectr",
    "fields": "egyptian_id_detectr",
    "digits": "arabic_numbers",
    "orientation": "national_id",
    "all": None,
}

# Windows: workers>0 spawns processes that each load torch/CUDA → WinError 1455
DATALOADER_WORKERS = 0 if sys.platform == "win32" else 8


def check_gpu() -> int:
    import torch

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU not available. Install PyTorch with CUDA support.")
        print(f"  torch version: {torch.__version__}")
        sys.exit(1)
    device = 0
    print(f"GPU active: CUDA:{device} — {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}")
    return device


def weights_path(run_name: str) -> Path:
    return RUNS / run_name / "weights" / "best.pt"


def train_stage(
    data_yaml: Path,
    model: str,
    name: str,
    epochs: int,
    device: int,
    batch: int = 16,
    lr0: float | None = None,
    *,
    train_overrides: dict[str, Any] | None = None,
    force: bool = False,
) -> Path:
    from ultralytics import YOLO

    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    best = weights_path(name)
    if best.is_file() and not force:
        print(f"Skipping training (exists): {best}")
        return best

    data_str = str(data_yaml.resolve())
    print(f"\n{'='*60}\nTraining: {name}\ndata: {data_str}\n{'='*60}")

    model_path = BASE / model
    yolo = YOLO(str(model_path if model_path.is_file() else model))
    kwargs = dict(
        data=data_str,
        epochs=epochs,
        imgsz=640,
        name=name,
        project=str(RUNS),
        device=device,
        batch=batch,
        workers=DATALOADER_WORKERS,
        exist_ok=True,
    )
    if lr0 is not None:
        kwargs["lr0"] = lr0
    if train_overrides:
        kwargs.update(train_overrides)
        print("Training overrides:")
        for key in sorted(train_overrides):
            print(f"  {key}: {train_overrides[key]}")

    results = yolo.train(**kwargs)
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training finished but weights missing: {best}")
    return best


def validate(best: Path, data_yaml: Path, device: int) -> None:
    from ultralytics import YOLO

    data_str = str(data_yaml.resolve())
    print(f"Validating: {best}")
    results = YOLO(str(best)).val(data=data_str, device=device)
    print(f"  Precision:   {results.box.mp:.4f}")
    print(f"  Recall:      {results.box.mr:.4f}")
    print(f"  mAP@50:      {results.box.map50:.4f}")
    print(f"  mAP@50-95:   {results.box.map:.4f}")


def field_detection_document_aug_overrides() -> dict[str, Any]:
    """Conservative phone-photo document augmentations for rigid card layout."""
    return {
        "degrees": 3.0,
        "translate": 0.03,
        "scale": 0.08,
        "shear": 0.0,
        "perspective": 0.0005,
        "hsv_h": 0.01,
        "hsv_s": 0.2,
        "hsv_v": 0.25,
        "fliplr": 0.0,
        "flipud": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "auto_augment": None,
        "erasing": 0.1,
    }


def field_detection_rotation_only_overrides() -> dict[str, Any]:
    return {
        "degrees": 3.0,
        "perspective": 0.0005,
    }


def field_detection_hsv_only_overrides() -> dict[str, Any]:
    return {
        "hsv_h": 0.01,
        "hsv_s": 0.2,
        "hsv_v": 0.25,
    }


def field_detection_translate_scale_only_overrides() -> dict[str, Any]:
    return {
        "translate": 0.03,
        "scale": 0.08,
    }


def field_detection_rotation_hsv_overrides() -> dict[str, Any]:
    out = {}
    out.update(field_detection_rotation_only_overrides())
    out.update(field_detection_hsv_only_overrides())
    return out


def field_detection_mosaic_off_overrides() -> dict[str, Any]:
    return {"mosaic": 0.0}


def field_detection_fliplr_off_overrides() -> dict[str, Any]:
    return {"fliplr": 0.0}


def field_detection_auto_augment_off_overrides() -> dict[str, Any]:
    return {"auto_augment": None}


def field_detection_erasing_low_overrides() -> dict[str, Any]:
    return {"erasing": 0.1}


def field_detection_mosaic_fliplr_off_overrides() -> dict[str, Any]:
    out = {}
    out.update(field_detection_mosaic_off_overrides())
    out.update(field_detection_fliplr_off_overrides())
    return out


def field_detection_fliplr_autoaugment_erasing_overrides() -> dict[str, Any]:
    """fliplr/auto_augment/erasing tweaks only; mosaic stays at Ultralytics default."""
    out = {}
    out.update(field_detection_fliplr_off_overrides())
    out.update(field_detection_auto_augment_off_overrides())
    out.update(field_detection_erasing_low_overrides())
    return out


def field_detection_ablation_overrides(preset: str) -> dict[str, Any] | None:
    presets: dict[str, dict[str, Any] | None] = {
        "default": None,
        "document": field_detection_document_aug_overrides(),
        "rotation_only": field_detection_rotation_only_overrides(),
        "hsv_only": field_detection_hsv_only_overrides(),
        "translate_scale_only": field_detection_translate_scale_only_overrides(),
        "rotation_hsv": field_detection_rotation_hsv_overrides(),
        "mosaic_off": field_detection_mosaic_off_overrides(),
        "fliplr_off": field_detection_fliplr_off_overrides(),
        "auto_augment_off": field_detection_auto_augment_off_overrides(),
        "erasing_low": field_detection_erasing_low_overrides(),
        "mosaic_fliplr_off": field_detection_mosaic_fliplr_off_overrides(),
        "fliplr_autoaugment_erasing": field_detection_fliplr_autoaugment_erasing_overrides(),
    }
    return presets[preset]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Egyptian ID YOLO stages")
    parser.add_argument(
        "--stage",
        choices=list(STAGE_ALIASES.keys()),
        default="all",
        help="Train one stage only (field_detection = Egyptian-ID-Detectr-3)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Run folder name under runs/ (default: stage default; use train_id_detectr_hyper_v2 for v2)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Override batch size. Use -1 to let Ultralytics auto-size batch from available VRAM.",
    )
    parser.add_argument("--force", action="store_true", help="Retrain even if best.pt exists")
    parser.add_argument(
        "--field-detector-augment",
        choices=(
            "default",
            "document",
            "rotation_only",
            "hsv_only",
            "translate_scale_only",
            "rotation_hsv",
            "mosaic_off",
            "fliplr_off",
            "auto_augment_off",
            "erasing_low",
            "mosaic_fliplr_off",
            "fliplr_autoaugment_erasing",
        ),
        default="default",
        help="Augmentation preset for the field-detection stage.",
    )
    args = parser.parse_args()

    os.chdir(BASE)
    device = check_gpu()

    ds_filter = STAGE_ALIASES.get(args.stage)
    ran = False
    train_overrides: dict[str, Any] | None = None
    if ds_filter == "egyptian_id_detectr" or args.stage == "field_detection":
        from tests.labeling.split_guard import assert_no_held_out_in_train

        dataset_root = DATASETS["egyptian_id_detectr"].parent
        try:
            assert_no_held_out_in_train(dataset_root)
        except RuntimeError as e:
            print(f"\n*** TRAIN GUARD: {e}\n")
            sys.exit(1)
        print("Train guard: no roboflow_valid/test labels in YOLO train/ OK")
        train_overrides = field_detection_ablation_overrides(args.field_detector_augment)

    for ds_key, model, default_name, epochs, batch, lr0 in STAGES:
        if ds_filter is not None and ds_key != ds_filter:
            continue
        if args.stage in ("field_detection", "fields") and ds_key == "egyptian_id_detectr":
            if default_name != FIELD_DETECTION_DEFAULT_RUN:
                continue
        name = args.name or default_name
        ep = args.epochs if args.epochs is not None else epochs
        bt = args.batch if args.batch is not None else batch
        yaml_path = DATASETS[ds_key]
        stage_overrides = train_overrides if ds_key == "egyptian_id_detectr" else None
        best = train_stage(
            yaml_path,
            model,
            name,
            ep,
            device,
            bt,
            lr0,
            train_overrides=stage_overrides,
            force=args.force,
        )
        validate(best, yaml_path, device)
        ran = True

    if not ran:
        print(f"Unknown stage configuration: {args.stage}")
        sys.exit(1)

    print("\nTraining finished. Weights under:", RUNS)


if __name__ == "__main__":
    main()
