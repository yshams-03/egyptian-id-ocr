"""
Isolate pipeline stages for per-stage regression tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

import export_id_to_excel as eid
import extract_back as eb
from extract_id_all import DEFAULT_DIGIT_WEIGHTS, ExtractConfig, extract_front, _init_ocr

from egypt_nid_decode import NidDecodeError, decode_egyptian_nid


def _device() -> str:
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def run_pipeline_row(
    front: Path,
    *,
    back: Path | None = None,
    fast_mode: bool = True,
    engine: str = "easyocr",
    auto_card_crop: bool = False,
    serial_charset_restrict: bool = True,
    local_engine_select_name: bool = True,
    easyocr_reader: object | None = None,
    field_yolo: object | None = None,
    digit_yolo: object | None = None,
) -> dict[str, str]:
    """Full extract_id_all path — same row as Excel/JSON export."""
    from extract_id_all import extract_all

    cfg = ExtractConfig(
        image=front,
        back_image=back,
        engine=engine,
        fast_mode=fast_mode,
        auto_card_crop=auto_card_crop,
        serial_charset_restrict=serial_charset_restrict,
        local_engine_select_name=local_engine_select_name,
        quiet=True,
        raise_on_empty=False,
        easyocr_reader=easyocr_reader,
        field_yolo=field_yolo,
        digit_yolo=digit_yolo,
    )
    return extract_all(cfg)


def run_front_ocr_row(
    front: Path,
    *,
    fast_mode: bool = True,
    engine: str = "easyocr",
    serial_charset_restrict: bool = True,
    local_engine_select_name: bool = True,
    easyocr_reader: object | None = None,
    field_yolo: object | None = None,
    digit_yolo: object | None = None,
) -> dict[str, str]:
    """Name, address, dob, serial, national_id (YOLO fields + digit YOLO)."""
    device = _device()
    cfg = ExtractConfig(
        image=front,
        fast_mode=fast_mode,
        engine=engine,
        serial_charset_restrict=serial_charset_restrict,
        local_engine_select_name=local_engine_select_name,
        quiet=True,
        raise_on_empty=False,
        easyocr_reader=easyocr_reader,
        field_yolo=field_yolo,
        digit_yolo=digit_yolo,
    )
    eng, tess_langs, reader = _init_ocr(cfg, device)
    reader = reader or easyocr_reader
    dw_path = DEFAULT_DIGIT_WEIGHTS
    if dw_path.is_file():
        dw = dw_path
    return extract_front(
        front,
        cfg,
        device=device,
        engine=eng,
        tess_langs=tess_langs,
        easyocr_reader=reader,
        dw=dw,
    )


def run_nid_decode_fields(national_id: str) -> dict[str, str]:
    """egypt_nid_decode.py outputs mapped to export column names."""
    empty = {
        "decoded_birth_date": "",
        "decoded_governorate": "",
        "decoded_gender": "",
        "decoded_century": "",
        "decoded_sequence": "",
        "decoded_check_digit": "",
    }
    if not (national_id or "").strip():
        return empty
    try:
        d = decode_egyptian_nid(national_id)
    except NidDecodeError:
        return empty
    dec = d.as_export_dict()
    return {
        "decoded_birth_date": dec["Birth Date"],
        "decoded_governorate": dec["Governorate"],
        "decoded_gender": dec["Gender"],
        "decoded_century": dec["Century"],
        "decoded_sequence": dec.get("Sequence", ""),
        "decoded_check_digit": dec.get("Check Digit", ""),
    }


def run_back_row(
    back: Path,
    *,
    front_nid: str = "",
    easyocr_reader: object | None = None,
    fast: bool = True,
) -> dict[str, str]:
    if easyocr_reader is None:
        raise RuntimeError("EasyOCR reader required for back extraction")
    bgr = cv2.imread(str(back))
    return eb.extract_back_fields(
        back, easyocr_reader, bgr=bgr, front_nid=front_nid, fast=fast
    )


def detection_labels_detected(front: Path, *, field_yolo: object | None = None) -> set[str]:
    """Which YOLO field labels were detected (field-detection stage)."""
    device = _device()
    fw = Path("runs/train_id_detectr_hyper/weights/best.pt")
    img = cv2.imread(str(front))
    if img is None:
        return set()
    model = field_yolo if field_yolo is not None else eid.get_yolo(fw)
    r = model.predict(source=img, conf=0.25, device=device, imgsz=640, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return set()
    id_to_name = eid.load_class_names()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    return {id_to_name.get(int(c), str(c)) for c in cls}
