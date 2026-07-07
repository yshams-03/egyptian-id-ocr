"""
Egyptian ID — front + back extraction (YOLO fields, digit YOLO, EasyOCR, Tesseract).

Examples:
  py extract_id_all.py "front.jpg" --output runs\\id_export\\out.xlsx
  py extract_id_all.py "back.jpg" --back --json-out runs\\id_export\\back.json
  py extract_id_all.py "front.jpg" --back-image "back.jpg" --auto-detect-side
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import export_id_to_excel as eid
import extract_back as eb
import extract_name_address as ena

BASE = Path(__file__).resolve().parent
RUNS = BASE / "runs"
DEFAULT_FIELD_WEIGHTS = RUNS / "train_id_detectr_hyper" / "weights" / "best.pt"
DEFAULT_DIGIT_WEIGHTS = RUNS / "train_arabic_numbers_v2" / "weights" / "best.pt"
DEFAULT_CARD_WEIGHTS = RUNS / "train_national_id_v7" / "weights" / "best.pt"

# Individual EasyOCR on name crops (A/B: up240_wide beat baseline on held-out name OCR).
NAME_OCR_MIN_SIDE = 240
NAME_OCR_MAX_SIDE = 880


def safe_print(msg: str) -> None:
    """Print on Windows consoles that lack UTF-8 (cp1252)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))


def empty_row(image_path: str = "") -> dict[str, str]:
    row = {k: "" for k in eid.EXCEL_ROW_COLUMNS}
    row["image_path"] = image_path
    return row


@dataclass
class ExtractConfig:
    image: Path
    back_image: Path | None = None
    field_weights: Path = DEFAULT_FIELD_WEIGHTS
    digit_weights: Path | None = None
    output: Path | None = None
    json_out: Path | None = None
    save_crops: Path | None = None
    conf: float = 0.25
    device: str = "0"
    pad: int = 6
    min_crop_side: int = 200
    nid_min_side: int = 220
    nid_expand_scale: float = 1.5
    digit_conf: float = 0.25
    digit_reading_order: str = "ltr"
    digit_dedupe_iou: float = 0.45
    engine: str = "mixed"
    lang_mode: str = "both"
    expect_arabic: bool = True
    serial_lang: str = "eng"
    serial_charset_restrict: bool = True
    decode_nid: bool = True
    dob_from_nid: bool = True
    strip_address_digits: bool = False
    use_notebook_field_ocr: bool = False
    card_weights: Path | None = None
    auto_card_crop: bool = False
    fallback_invalid_fields: bool = True
    raise_on_empty: bool = True
    quiet: bool = False
    easyocr_reader: object | None = None
    force_back: bool = False
    auto_detect_side: bool = False
    fast_mode: bool = False
    field_yolo: object | None = None
    digit_yolo: object | None = None


def _init_ocr(cfg: ExtractConfig, device: str) -> tuple[str, list[str], object | None]:
    if cfg.lang_mode == "ara":
        tess_langs = ["ara"]
    elif cfg.lang_mode == "ara+eng":
        tess_langs = ["ara+eng"]
    else:
        tess_langs = ["ara", "ara+eng"]

    engine = cfg.engine
    reader = cfg.easyocr_reader
    if reader is None and engine in ("easyocr", "mixed"):
        try:
            import easyocr  # type: ignore[import-not-found]

            if not cfg.quiet:
                print("Loading EasyOCR models (first time can take 1–2 minutes)…")
            reader = easyocr.Reader(["ar", "en"], gpu=device != "cpu", verbose=False)
        except ImportError:
            if engine == "easyocr":
                raise SystemExit("Install EasyOCR: py -m pip install easyocr") from None
            if not cfg.quiet:
                print("EasyOCR not installed — using Tesseract only for Arabic fields.")
            engine = "tesseract"
        except Exception as ex:  # noqa: BLE001
            if engine == "easyocr":
                raise SystemExit(f"EasyOCR failed: {ex}") from ex
            if not cfg.quiet:
                print(f"EasyOCR init failed ({ex}); using Tesseract only.")
            engine = "tesseract"
    if engine == "mixed" and reader is None:
        engine = "tesseract"

    if cfg.fast_mode:
        engine = "easyocr" if reader is not None else engine
    elif engine in ("tesseract", "mixed"):
        if not cfg.quiet:
            print("Using Tesseract:", eid.setup_tesseract())
    if not cfg.quiet:
        print(f"OCR engine for names/address/dob: {engine} (lang-mode={cfg.lang_mode})")
    return engine, tess_langs, reader


def extract_front(
    img_path: Path,
    cfg: ExtractConfig,
    *,
    device: str,
    engine: str,
    tess_langs: list[str],
    easyocr_reader: object | None,
    dw: Path | None,
) -> dict[str, str]:
    """Front card: YOLO fields + NID digits + photo + decode."""
    row = empty_row(str(img_path))
    row["nid_decode_error"] = ""
    serial_allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" if cfg.serial_charset_restrict else None
    serial_psm = 7 if cfg.serial_charset_restrict else 6

    fw = cfg.field_weights.expanduser().resolve()
    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"Could not read image: {img_path}")
    if cfg.fast_mode:
        img = eid.resize_for_speed(img, max_side=880)

    cw = cfg.card_weights
    if cw is None and cfg.auto_card_crop and DEFAULT_CARD_WEIGHTS.is_file():
        cw = DEFAULT_CARD_WEIGHTS
    if cw is not None:
        cw = Path(cw).expanduser().resolve()
        if cw.is_file():
            img, card_lab = eid.crop_id_card_region(img, cw, conf=cfg.conf, device=device)
            if not cfg.quiet and card_lab:
                print(f"Card crop ({card_lab}): {img.shape[1]}x{img.shape[0]} px")

    model = cfg.field_yolo if cfg.field_yolo is not None else eid.get_yolo(fw)
    field_imgsz = 480 if cfg.fast_mode else 640
    r = model.predict(
        source=img, conf=cfg.conf, device=device, imgsz=field_imgsz, verbose=False
    )[0]

    if r.boxes is None or len(r.boxes) == 0:
        row["nid_decode_error"] = "no detections"
        return row

    id_to_name = eid.load_class_names()
    xyxy = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()
    if cfg.fallback_invalid_fields:
        best, used_invalid = eid.best_boxes_with_invalid_fallback(
            xyxy, cls, conf, id_to_name
        )
        if used_invalid and not cfg.quiet:
            print(
                "Note: using invalid_* field boxes for OCR (augmented / difficult images)."
            )
    else:
        best = eid.best_boxes_by_label(xyxy, cls, conf, id_to_name)

    photo_dir = cfg.save_crops or (cfg.output.parent if cfg.output else RUNS / "id_export")
    photo_dir = Path(photo_dir).expanduser().resolve()
    photo_path = eid.extract_photo(img, best, photo_dir, img_path.stem, pad=cfg.pad)
    if photo_path:
        row["photo_path"] = photo_path

    if cfg.save_crops and not cfg.fast_mode:
        crop_dir = cfg.save_crops.expanduser().resolve()
        crop_dir.mkdir(parents=True, exist_ok=True)
        for lab in (
            "nid",
            "nid_back",
            "firstName",
            "lastName",
            "address",
            "dob",
            "serial",
            "photo",
        ):
            if lab not in best:
                continue
            c = eid.crop_xyxy(img, best[lab][0], cfg.pad)
            if c.size == 0:
                continue
            cv2.imwrite(str(crop_dir / f"{lab}.png"), c)

    field_engine = engine
    if cfg.fast_mode and easyocr_reader is not None:
        field_engine = "easyocr"

    min_side = 120 if cfg.fast_mode else cfg.min_crop_side

    def read_field(lab: str, *, expect_arabic: bool) -> str:
        if lab not in best:
            return ""
        if cfg.use_notebook_field_ocr:
            return eid.extract_text_notebook(img, best[lab][0], lang="ara")
        cr = eid.crop_xyxy(img, best[lab][0], cfg.pad)
        return ena.ocr_text_field(
            cr,
            engine=field_engine,
            min_side=min_side,
            langs=tess_langs,
            expect_arabic=expect_arabic,
            easyocr_reader=easyocr_reader,
        )

    if cfg.fast_mode and easyocr_reader is not None:
        min_side_batch = min_side

        def _batch_spacer_for(lab: str) -> np.ndarray | None:
            """Same upscale as batch bands — preserves strip geometry for dob/serial."""
            if lab not in best:
                return None
            cr = eid.crop_xyxy(img, best[lab][0], cfg.pad)
            if cr.size == 0:
                return None
            up = eid.upscale_crop(cr, min_side=min_side_batch)
            return eid.resize_for_speed(up, max_side=480)

        leading_spacers = [
            sp
            for lab in ("firstName", "lastName", "address")
            if (sp := _batch_spacer_for(lab)) is not None
        ]
        labeled = [
            (lab, eid.crop_xyxy(img, best[lab][0], cfg.pad))
            for lab in ("dob", "serial")
            if lab in best
        ]
        batched = ena.ocr_fields_batch_easyocr(
            labeled,
            easyocr_reader,
            min_side=min_side_batch,
            leading_spacers=leading_spacers or None,
        )
        if "firstName" in best:
            cr = eid.crop_xyxy(img, best["firstName"][0], cfg.pad)
            first = ena.ocr_text_field_easyocr(
                cr,
                easyocr_reader,
                min_side=NAME_OCR_MIN_SIDE,
                max_side=NAME_OCR_MAX_SIDE,
            )
        else:
            first = ""
        if "lastName" in best:
            cr = eid.crop_xyxy(img, best["lastName"][0], cfg.pad)
            last = ena.ocr_text_field_easyocr(
                cr,
                easyocr_reader,
                min_side=NAME_OCR_MIN_SIDE,
                max_side=NAME_OCR_MAX_SIDE,
            )
        else:
            last = ""
        if "address" in best:
            cr = eid.crop_xyxy(img, best["address"][0], cfg.pad)
            addr_raw = ena.ocr_text_field_easyocr(cr, easyocr_reader, min_side=min_side)
            addr = eid.clean_address_text(addr_raw, strip_digits=cfg.strip_address_digits)
        else:
            addr = ""
        dob = batched.get("dob", "")
        serial = eid.merge_serial_ocr("", batched.get("serial") or None)
    else:
        first = read_field("firstName", expect_arabic=cfg.expect_arabic)
        last = read_field("lastName", expect_arabic=cfg.expect_arabic)
        addr = read_field("address", expect_arabic=cfg.expect_arabic)
        addr = eid.clean_address_text(addr, strip_digits=cfg.strip_address_digits)
        dob = read_field("dob", expect_arabic=False)
        serial = ""
    if (
        not cfg.fast_mode
        and not (dob or "").strip()
        and "dob" in best
        and engine in ("tesseract", "mixed")
    ):
        scr = eid.crop_xyxy(img, best["dob"][0], cfg.pad)
        dob = eid.ocr_crop(eid.upscale_crop(scr, min_side=cfg.min_crop_side), "eng", 6)

    if not (cfg.fast_mode and easyocr_reader is not None and not cfg.serial_charset_restrict):
        serial = ""
        if "serial" in best:
            scr = eid.crop_xyxy(img, best["serial"][0], cfg.pad)
            tess_s = ""
            if engine in ("tesseract", "mixed"):
                tess_s = eid.ocr_crop(
                    eid.upscale_crop(scr, min_side=cfg.min_crop_side),
                    cfg.serial_lang,
                    serial_psm,
                    whitelist=serial_allowlist,
                )
            eo_s = ""
            if easyocr_reader is not None:
                eo_s = ena.ocr_text_field_easyocr(
                    scr,
                    easyocr_reader,
                    min_side=cfg.min_crop_side,
                    allowlist=serial_allowlist,
                )
            serial = eid.merge_serial_ocr(tess_s, eo_s or None)

    nid_raw = ""
    nid_back_raw = ""
    if not cfg.fast_mode:
        if "nid" in best:
            nc = eid.crop_xyxy(img, best["nid"][0], cfg.pad)
            nid_raw = eid.ocr_nid_crop(nc, min_side=cfg.nid_min_side)
        if "nid_back" in best:
            cr = eid.crop_xyxy(img, best["nid_back"][0], cfg.pad)
            nid_back_raw = eid.ocr_nid_crop(cr, min_side=cfg.nid_min_side)
    combined_nid = f"{nid_raw} {nid_back_raw}".strip()
    nid_digits = eid.normalize_nid_digits(combined_nid)

    nid_yolo = ""
    if dw is not None and "nid" in best:
        box, _ = best["nid"]
        x1, y1, x2, y2 = eid.expand_bbox_height_xyxy(box, cfg.nid_expand_scale, img.shape)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        if x2 > x1 and y2 > y1:
            nid_sub = img[y1:y2, x1:x2].copy()
            digit_model = cfg.digit_yolo
            if digit_model is None:
                digit_model = eid.get_yolo(dw)
            digit_imgsz = 512 if cfg.fast_mode else 640
            nid_yolo = eid.nid_digits_yolo_on_crop(
                nid_sub,
                dw,
                device=device,
                conf=cfg.digit_conf,
                dedupe_iou=cfg.digit_dedupe_iou,
                reading_order=cfg.digit_reading_order,
                model=digit_model,
                imgsz=digit_imgsz,
            )
    wy = eid.western_digits_only(nid_yolo)
    wt = eid.western_digits_only(nid_digits)
    if wy and (len(wy) == 14 or len(wy) > len(wt)):
        nid_digits = wy
    elif (
        not cfg.fast_mode
        and len(wy) < 14
        and easyocr_reader is not None
        and "nid" in best
    ):
        nc = eid.crop_xyxy(img, best["nid"][0], cfg.pad)
        eo_nid = eid.ocr_nid_easyocr_digits(nc, easyocr_reader, min_side=cfg.nid_min_side)
        if len(eo_nid) >= len(wy):
            nid_digits = eo_nid

    row["national_id"] = nid_digits or re.sub(r"[^\d]", "", combined_nid)
    row["first_name"] = first
    row["last_name"] = last
    row["full_name"] = f"{first} {last}".strip()
    row["address"] = addr
    row["dob"] = dob
    row["serial"] = serial

    if cfg.decode_nid:
        nid14 = eid.western_digits_only(row["national_id"])
        if len(nid14) != 14:
            row["nid_decode_error"] = (
                "no 14-digit national_id detected"
                if not nid14
                else f"national_id has {len(nid14)} digits, expected 14"
            )
        else:
            try:
                dec = eid.decode_egyptian_id(nid14)
                row["decoded_birth_date"] = dec["Birth Date"]
                row["decoded_century"] = dec.get("Century", "")
                row["decoded_governorate"] = dec["Governorate"]
                row["decoded_gender"] = dec["Gender"]
                row["decoded_sequence"] = dec.get("Sequence", "")
                row["decoded_check_digit"] = dec.get("Check Digit", "")
                row["nid_decode_error"] = ""
            except ValueError as e:
                row["nid_decode_error"] = str(e)

    bd = (row.get("decoded_birth_date") or "").strip()
    if cfg.dob_from_nid and cfg.decode_nid and bd:
        row["dob"] = bd

    return row


def extract_all(cfg: ExtractConfig) -> dict[str, str]:
    """Run front and/or back pipeline; return merged row for Excel/JSON."""
    import torch

    front_path = cfg.image.expanduser().resolve()
    if not front_path.is_file():
        raise SystemExit(f"Image not found: {front_path}")

    fw = cfg.field_weights.expanduser().resolve()
    if not fw.is_file():
        raise SystemExit(f"Field weights not found: {fw}")

    dw = cfg.digit_weights
    if dw is None and DEFAULT_DIGIT_WEIGHTS.is_file():
        dw = DEFAULT_DIGIT_WEIGHTS
    if dw is not None:
        dw = dw.expanduser().resolve()
        if not dw.is_file() and not cfg.quiet:
            print(f"Digit weights not found ({dw}); NID YOLO disabled.")
        dw = dw if dw and dw.is_file() else None

    out_xlsx = cfg.output
    if out_xlsx is None:
        out_xlsx = RUNS / "id_export" / f"{front_path.stem}_full.xlsx"
    out_xlsx = out_xlsx.expanduser().resolve()

    device = cfg.device
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"

    if not cfg.fast_mode and cfg.engine != "easyocr":
        try:
            import pytesseract  # noqa: F401
        except ImportError as ex:
            raise SystemExit("Install pytesseract: py -m pip install pytesseract") from ex

    if cfg.fast_mode and cfg.engine == "mixed":
        cfg.engine = "easyocr"

    engine, tess_langs, reader = _init_ocr(cfg, device)

    back_path = cfg.back_image.expanduser().resolve() if cfg.back_image else None
    run_front = not cfg.force_back
    run_back = cfg.force_back or back_path is not None

    if cfg.auto_detect_side and back_path is None:
        side = eb.detect_card_side(front_path, reader)
        if not cfg.quiet:
            print(f"Auto-detected card side: {side}")
        if side == "back":
            run_front, run_back = False, True
            back_path = front_path
        else:
            run_front, run_back = True, False

    front_row = empty_row(str(front_path))
    back_row = eb._empty_back_fields()

    if run_front:
        front_row = extract_front(
            front_path,
            cfg,
            device=device,
            engine=engine,
            tess_langs=tess_langs,
            easyocr_reader=reader,
            dw=dw,
        )
        if cfg.raise_on_empty and front_row.get("nid_decode_error") == "no detections":
            raise SystemExit("No field detections on front image.")

    if run_back:
        bp = back_path or front_path
        if reader is None:
            raise SystemExit("Back extraction requires EasyOCR (use --engine easyocr or mixed).")
        if not cfg.quiet:
            print(f"Extracting back fields from: {bp}")
        back_row = eb.extract_back_fields(
            bp,
            reader,
            front_nid=front_row.get("national_id", ""),
            fast=cfg.fast_mode,
        )
        if not run_front:
            front_row["image_path"] = str(bp)

    merged = eb.merge_front_back(front_row, back_row)
    if back_path and run_front:
        merged["image_path"] = f"{front_path} | {back_path}"

    _write_outputs(merged, out_xlsx, cfg.json_out)
    if not cfg.quiet:
        print(f"Wrote: {out_xlsx.resolve()}")
        if cfg.json_out:
            print(f"Wrote: {cfg.json_out.expanduser().resolve()}")
        for k in eid.EXCEL_ROW_COLUMNS:
            v = merged.get(k, "")
            if not v:
                continue
            prev = (str(v)[:100] + "…") if len(str(v)) > 100 else v
            safe_print(f"  {k}: {prev!r}")
    return merged


def _write_outputs(row: dict[str, str], out_xlsx: Path, json_out: Path | None) -> None:
    try:
        import pandas as pd
    except ImportError as ex:
        raise SystemExit("Install pandas openpyxl: py -m pip install pandas openpyxl") from ex

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    eid.write_excel_safe(pd.DataFrame([eid.row_for_excel(row)], columns=list(eid.EXCEL_ROW_COLUMNS)), out_xlsx)
    if json_out is not None:
        jp = json_out.expanduser().resolve()
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(
            json.dumps(eid.row_for_excel(row), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract Egyptian ID front + back fields to Excel/JSON."
    )
    p.add_argument("image", type=Path, help="Front image, back image, or single image with --auto-detect-side.")
    p.add_argument("--back-image", type=Path, default=None, help="Separate back card image; merged with front row.")
    p.add_argument("--back", action="store_true", help="Treat main image as back only (no front YOLO).")
    p.add_argument(
        "--auto-detect-side",
        action="store_true",
        help="Detect front vs back on a single image (use with one photo).",
    )
    p.add_argument("--field-weights", type=Path, default=DEFAULT_FIELD_WEIGHTS)
    p.add_argument("--weights", type=Path, default=None, help="Alias for --field-weights.")
    p.add_argument("--digit-weights", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--save-crops", type=Path, default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="0")
    p.add_argument("--pad", type=int, default=6)
    p.add_argument("--min-crop-side", type=int, default=200)
    p.add_argument("--nid-min-side", type=int, default=220)
    p.add_argument("--nid-expand-scale", type=float, default=1.5)
    p.add_argument("--digit-conf", type=float, default=0.25)
    p.add_argument("--digit-reading-order", choices=("auto", "ltr", "row_col"), default="ltr")
    p.add_argument("--digit-dedupe-iou", type=float, default=0.45)
    p.add_argument("--engine", choices=("tesseract", "easyocr", "mixed"), default="mixed")
    p.add_argument("--lang-mode", choices=("ara", "ara+eng", "both"), default="both")
    p.add_argument("--no-expect-arabic", action="store_true")
    p.add_argument("--serial-lang", default="eng")
    p.add_argument(
        "--no-serial-charset-restrict",
        action="store_true",
        help="Disable A-Z0-9 serial OCR allowlist (default: restrict enabled).",
    )
    p.add_argument("--decode-nid", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--strip-address-digits", action="store_true")
    p.add_argument("--no-dob-from-nid", action="store_true")
    p.add_argument("--use-notebook-field-ocr", action="store_true")
    p.add_argument("--card-weights", type=Path, default=None)
    p.add_argument("--auto-card-crop", action="store_true")
    p.add_argument("--no-fallback-invalid-fields", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> ExtractConfig:
    fw = args.weights if args.weights is not None else args.field_weights
    return ExtractConfig(
        image=args.image,
        back_image=args.back_image,
        field_weights=fw,
        digit_weights=args.digit_weights,
        output=args.output,
        json_out=args.json_out,
        save_crops=args.save_crops,
        conf=args.conf,
        device=args.device,
        pad=args.pad,
        min_crop_side=args.min_crop_side,
        nid_min_side=args.nid_min_side,
        nid_expand_scale=args.nid_expand_scale,
        digit_conf=args.digit_conf,
        digit_reading_order=args.digit_reading_order,
        digit_dedupe_iou=args.digit_dedupe_iou,
        engine=args.engine,
        lang_mode=args.lang_mode,
        expect_arabic=not args.no_expect_arabic,
        serial_lang=args.serial_lang,
        serial_charset_restrict=not args.no_serial_charset_restrict,
        decode_nid=args.decode_nid,
        dob_from_nid=not args.no_dob_from_nid,
        strip_address_digits=args.strip_address_digits,
        use_notebook_field_ocr=args.use_notebook_field_ocr,
        card_weights=args.card_weights,
        auto_card_crop=args.auto_card_crop,
        fallback_invalid_fields=not args.no_fallback_invalid_fields,
        force_back=args.back,
        auto_detect_side=args.auto_detect_side,
    )


def main(argv: list[str] | None = None) -> None:
    extract_all(config_from_args(build_arg_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
