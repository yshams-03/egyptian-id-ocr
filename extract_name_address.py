"""
Extract Arabic first name, last name, and address from an ID photo using the
same field detector as the working NID digit pipeline (train_id_detectr_hyper)
plus Tesseract and/or EasyOCR.

Requires: py -m pip install ultralytics opencv-python pytesseract pyyaml pandas openpyxl
          Tesseract with Arabic traineddata (same as export_id_to_excel.py)
Optional: py -m pip install easyocr  (often much better Arabic on screenshots)

Examples:
  py extract_name_address.py "C:\\Users\\yassi\\Downloads\\my_id.jpg"
  py extract_name_address.py id.jpg --engine easyocr
  py extract_name_address.py id.jpg --digit-weights runs\\train_arabic_numbers_v2\\weights\\best.pt --decode-nid
  py extract_name_address.py id.jpg --output runs\\id_export\\name_address.xlsx --save-crops runs\\id_export\\crops
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

import export_id_to_excel as eid

BASE = Path(__file__).resolve().parent
RUNS = BASE / "runs"
DEFAULT_FIELD_WEIGHTS = RUNS / "train_id_detectr_hyper" / "weights" / "best.pt"
DEFAULT_DIGIT_WEIGHTS = RUNS / "train_arabic_numbers_v2" / "weights" / "best.pt"

# Arabic + Arabic supplement + presentation forms (common on IDs)
_ARABIC = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)


def _arabic_char_count(s: str) -> int:
    return sum(1 for c in s if _ARABIC.match(c))


def _latin_letter_count(s: str) -> int:
    return sum(1 for c in s if "A" <= c <= "Z" or "a" <= c <= "z")


def _score_reading(s: str, expect_arabic: bool) -> float:
    """Prefer Arabic script; penalize long Latin gibberish typical of failed Tesseract."""
    s = " ".join(s.split()).strip()
    if not s:
        return -1e18
    if not expect_arabic:
        return float(len(s))
    n = len(s)
    ar = _arabic_char_count(s)
    lat = _latin_letter_count(s)
    ar_ratio = ar / max(n, 1)
    lat_ratio = lat / max(n, 1)
    # Strong Arabic lines win; long Latin-only lines lose hard
    if ar >= 2:
        return 2000.0 + ar_ratio * 500.0 + n * 0.5 - lat_ratio * 200.0
    if ar == 1:
        return 800.0 + n * 0.3 - lat * 1.5
    # No Arabic detected — allow short Latin (edge case) but crush "Cpeaitleg" style noise
    if lat_ratio > 0.55 and n > 6:
        return -500.0 - lat_ratio * 100.0
    # Punctuation / OCR noise with no letters (e.g. "—_-") should never beat Arabic reads
    letters = sum(1 for c in s if c.isalpha())
    if ar == 0 and letters == 0:
        return -800.0
    return n * 0.15 - lat * 0.4


def _field_preprocess_variants(gray: np.ndarray) -> list[np.ndarray]:
    """Several single-channel views; screenshots often need more than plain Otsu."""
    out: list[np.ndarray] = []
    if gray.size == 0:
        return out
    out.append(gray)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gc = clahe.apply(gray)
    out.append(gc)
    blur = cv2.GaussianBlur(gc, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(otsu)
    out.append(cv2.bitwise_not(otsu))
    try:
        # Some builds expose constants only as ints (ADAPTIVE_GAUSSIAN_C=1, THRESH_BINARY=0).
        ag = getattr(cv2, "ADAPTIVE_GAUSSIAN_C", 1)
        tb = getattr(cv2, "THRESH_BINARY", 0)
        adapt = cv2.adaptiveThreshold(gray, 255, ag, tb, 35, 11)
        out.append(adapt)
        out.append(cv2.bitwise_not(adapt))
    except (cv2.error, AttributeError):
        pass
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    out.append(cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, k))
    return out


def _tesseract_strings_on_variants(
    gray_u8: np.ndarray, langs: list[str], psms: tuple[int, ...]
) -> list[str]:
    import pytesseract

    found: list[str] = []
    for prep in _field_preprocess_variants(gray_u8):
        for lang in langs:
            for psm in psms:
                cfg = f"--oem 3 --psm {psm}"
                try:
                    t = pytesseract.image_to_string(prep, lang=lang, config=cfg)
                except Exception:
                    continue
                t = " ".join(t.split()).strip()
                if t:
                    found.append(t)
    return found


def ocr_text_field_tesseract(
    bgr: np.ndarray,
    *,
    min_side: int,
    langs: list[str],
    expect_arabic: bool,
    psms: tuple[int, ...] = (6, 7, 11, 13),
) -> str:
    if bgr.size == 0:
        return ""
    big = eid.upscale_crop(bgr, min_side=min_side)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    candidates = _tesseract_strings_on_variants(gray, langs, psms)
    if not candidates:
        return ""
    best_s = max(candidates, key=lambda s: _score_reading(s, expect_arabic))
    if expect_arabic and _arabic_char_count(best_s) == 0:
        return ""
    return best_s


def ocr_text_field_easyocr(
    bgr: np.ndarray,
    reader,
    *,
    min_side: int,
    max_side: int = 520,
    allowlist: str | None = None,
) -> str:
    if bgr.size == 0:
        return ""
    big = eid.upscale_crop(bgr, min_side=max(min_side, 120))
    big = eid.resize_for_speed(big, max_side=max_side)
    rgb = cv2.cvtColor(big, cv2.COLOR_BGR2RGB)
    kwargs = {"paragraph": True}
    if allowlist:
        kwargs["allowlist"] = allowlist
    parts = reader.readtext(rgb, **kwargs)
    texts: list[str] = []
    for item in parts:
        text = item[1] if len(item) > 1 else ""
        conf = float(item[2]) if len(item) > 2 else 1.0
        if conf >= 0.05 and text.strip():
            texts.append(text.strip())
    return " ".join(texts).strip()


def ocr_fields_batch_easyocr(
    labeled_crops: list[tuple[str, np.ndarray]],
    reader,
    *,
    min_side: int = 120,
    leading_spacers: list[np.ndarray] | None = None,
) -> dict[str, str]:
    """
    One EasyOCR pass for several field crops (stacked vertically) — much faster than N separate calls.

    leading_spacers: optional blank bands stacked above labeled_crops to preserve strip geometry
    (same upscale path as crops) without OCR assignment — used when name fields are OCR'd individually.
    """
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for sp in leading_spacers or []:
        if sp is None or sp.size == 0:
            continue
        rows.append(sp)
        labels.append("")  # geometry only; excluded from OCR output keys
    for lab, bgr in labeled_crops:
        if bgr is None or bgr.size == 0:
            continue
        up = eid.upscale_crop(bgr, min_side=min_side)
        up = eid.resize_for_speed(up, max_side=480)
        rows.append(up)
        labels.append(lab)

    if not rows:
        return {}

    max_w = max(r.shape[1] for r in rows)
    gap = 6
    bands: list[tuple[str, int, int]] = []
    stack_parts: list[np.ndarray] = []
    y = 0
    for lab, r in zip(labels, rows):
        h, w = r.shape[:2]
        if w < max_w:
            r = np.hstack([r, np.full((h, max_w - w, 3), 255, dtype=np.uint8)])
        stack_parts.append(r)
        if lab:
            bands.append((lab, y, y + h))
        y += h + gap
        stack_parts.append(np.full((gap, max_w, 3), 255, dtype=np.uint8))

    strip = np.vstack(stack_parts)
    strip = eid.resize_for_speed(strip, max_side=1200)
    rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
    ocr_labels = [lab for lab in labels if lab]
    out: dict[str, list[str]] = {lab: [] for lab in ocr_labels}

    try:
        results = reader.readtext(rgb, detail=1, paragraph=False)
    except Exception:
        results = []

    for item in results:
        if len(item) < 3:
            continue
        bbox, text, conf = item[0], str(item[1]).strip(), float(item[2])
        if not text or conf < 0.08:
            continue
        pts = np.asarray(bbox, dtype=float)
        yc = float(np.mean(pts[:, 1]))
        for lab, y0, y1 in bands:
            if y0 <= yc <= y1:
                out[lab].append(text)
                break

    return {lab: " ".join(parts).strip() for lab, parts in out.items() if parts}


def ocr_text_field(
    bgr: np.ndarray,
    *,
    engine: str,
    min_side: int,
    langs: list[str],
    expect_arabic: bool,
    easyocr_reader,
) -> str:
    if bgr.size == 0:
        return ""
    if engine == "easyocr":
        if easyocr_reader is None:
            raise SystemExit("easyocr engine requested but reader failed to load.")
        return ocr_text_field_easyocr(bgr, easyocr_reader, min_side=min_side)
    if engine == "mixed":
        t = ocr_text_field_tesseract(
            bgr, min_side=min_side, langs=langs, expect_arabic=expect_arabic
        )
        e = ""
        if easyocr_reader is not None:
            e = ocr_text_field_easyocr(bgr, easyocr_reader, min_side=min_side)
        st = _score_reading(t, expect_arabic)
        se = _score_reading(e, expect_arabic)
        if se > st + 5:
            return e
        return t or e
    return ocr_text_field_tesseract(
        bgr, min_side=min_side, langs=langs, expect_arabic=expect_arabic
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCR firstName, lastName, and address from Egyptian ID image."
    )
    parser.add_argument("image", type=Path, help="Path to ID image (quote if spaces).")
    parser.add_argument(
        "--field-weights",
        type=Path,
        default=DEFAULT_FIELD_WEIGHTS,
        help="YOLO field detector .pt (default: train_id_detectr_hyper).",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--pad", type=int, default=6)
    parser.add_argument(
        "--min-crop-side",
        type=int,
        default=200,
        help="Minimum short side for text crops before OCR (higher helps Arabic).",
    )
    parser.add_argument(
        "--lang-mode",
        choices=("ara", "ara+eng", "both"),
        default="both",
        help="Tesseract language: ara only, ara+eng, or try both and pick best-scoring text.",
    )
    parser.add_argument(
        "--engine",
        choices=("tesseract", "easyocr", "mixed"),
        default="mixed",
        help="OCR backend. 'mixed' prefers EasyOCR when installed, else Tesseract; "
        "Tesseract-only path uses stronger preprocessing + Arabic scoring.",
    )
    parser.add_argument(
        "--no-expect-arabic",
        action="store_true",
        help="Score reads by length only (for Latin-only test images).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional .xlsx path (one row: first_name, last_name, address, full_name).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the same fields as JSON.",
    )
    parser.add_argument(
        "--save-crops",
        type=Path,
        default=None,
        help="If set, save firstName/lastName/address PNG crops here.",
    )
    parser.add_argument(
        "--digit-weights",
        type=Path,
        default=None,
        help="Arabic-digit YOLO .pt; with nid box, reads 14-digit NID (egyptian_id_ocr.ipynb flow).",
    )
    parser.add_argument("--nid-expand-scale", type=float, default=1.5)
    parser.add_argument("--digit-conf", type=float, default=0.25)
    parser.add_argument(
        "--digit-reading-order",
        choices=("auto", "ltr", "row_col"),
        default="ltr",
    )
    parser.add_argument("--digit-dedupe-iou", type=float, default=0.45)
    parser.add_argument(
        "--decode-nid",
        action="store_true",
        help="Print and export decoded birth date / governorate / gender from 14-digit NID.",
    )
    parser.add_argument(
        "--strip-address-digits",
        action="store_true",
        help="Remove digit runs from address (notebook remove_numbers).",
    )
    parser.add_argument(
        "--include-serial",
        action="store_true",
        help="OCR serial field with eng Tesseract when a serial box is detected.",
    )
    args = parser.parse_args()

    img_path = args.image.expanduser().resolve()
    if not img_path.is_file():
        raise SystemExit(f"Image not found: {img_path}")

    wpath = args.field_weights.expanduser().resolve()
    if not wpath.is_file():
        raise SystemExit(f"Weights not found: {wpath}")

    import torch
    from ultralytics import YOLO

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"

    if args.engine != "easyocr":
        try:
            import pytesseract  # noqa: F401
        except ImportError as ex:
            raise SystemExit("Install pytesseract: py -m pip install pytesseract") from ex

    expect_arabic = not args.no_expect_arabic
    if args.lang_mode == "ara":
        tess_langs = ["ara"]
    elif args.lang_mode == "ara+eng":
        tess_langs = ["ara+eng"]
    else:
        tess_langs = ["ara", "ara+eng"]

    engine = args.engine
    easyocr_reader = None
    if engine in ("easyocr", "mixed"):
        try:
            import easyocr  # type: ignore[import-not-found]

            easyocr_reader = easyocr.Reader(["ar", "en"], gpu=device != "cpu", verbose=False)
        except ImportError as ex:
            if engine == "easyocr":
                raise SystemExit(
                    "Install EasyOCR: py -m pip install easyocr"
                ) from ex
            print("EasyOCR not installed — mixed mode will use Tesseract only.")
            engine = "tesseract"
        except Exception as ex:  # noqa: BLE001
            if engine == "easyocr":
                raise SystemExit(f"EasyOCR failed to initialize: {ex}") from ex
            print(f"EasyOCR init failed ({ex}); mixed mode will use Tesseract only.")
            engine = "tesseract"

    if engine == "mixed" and easyocr_reader is not None:
        pass  # keep mixed
    elif engine == "mixed":
        engine = "tesseract"

    if engine in ("tesseract", "mixed"):
        tess = eid.setup_tesseract()
        print(f"Using Tesseract: {tess}")
    else:
        print("Using Tesseract: (skipped — EasyOCR only)")

    if engine == "easyocr":
        print("OCR engine: EasyOCR")
    elif engine == "mixed":
        print("OCR engine: mixed (Tesseract + EasyOCR, best Arabic-plausible read)")
    else:
        print(f"OCR engine: Tesseract (lang-mode={args.lang_mode})")

    id_to_name = eid.load_class_names()
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise SystemExit(f"Could not read image: {img_path}")

    model = YOLO(str(wpath))
    r = model.predict(source=bgr, conf=args.conf, device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        raise SystemExit("No detections — lower --conf or check image/weights.")

    xyxy = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()
    best = eid.best_boxes_by_label(xyxy, cls, conf, id_to_name)

    if args.save_crops:
        outd = args.save_crops.expanduser().resolve()
        outd.mkdir(parents=True, exist_ok=True)
        for lab in ("firstName", "lastName", "address", "nid", "serial"):
            if lab not in best:
                continue
            cr = eid.crop_xyxy(bgr, best[lab][0], args.pad)
            if cr.size == 0:
                continue
            cv2.imwrite(str(outd / f"{lab}.png"), cr)
        print(f"Saved crops to: {outd}")

    def read_label(lab: str) -> str:
        if lab not in best:
            return ""
        cr = eid.crop_xyxy(bgr, best[lab][0], args.pad)
        return ocr_text_field(
            cr,
            engine=engine,
            min_side=args.min_crop_side,
            langs=tess_langs,
            expect_arabic=expect_arabic,
            easyocr_reader=easyocr_reader,
        )

    first = read_label("firstName")
    last = read_label("lastName")
    addr = read_label("address")
    if args.strip_address_digits:
        addr = eid.remove_numbers(addr)
    full = f"{first} {last}".strip()

    serial = ""
    if args.include_serial and "serial" in best and engine in ("tesseract", "mixed"):
        scr = eid.crop_xyxy(bgr, best["serial"][0], args.pad)
        bigs = eid.upscale_crop(scr, min_side=args.min_crop_side)
        tess_s = eid.ocr_crop(bigs, "eng", 6)
        eo_s = ""
        if easyocr_reader is not None:
            eo_s = ocr_text_field_easyocr(scr, easyocr_reader, min_side=args.min_crop_side)
        serial = eid.merge_serial_ocr(tess_s, eo_s or None)
    elif args.include_serial and "serial" in best and engine == "easyocr" and easyocr_reader is not None:
        scr = eid.crop_xyxy(bgr, best["serial"][0], args.pad)
        eo_s = ocr_text_field_easyocr(scr, easyocr_reader, min_side=args.min_crop_side)
        serial = eid.merge_serial_ocr("", eo_s or None)

    national_id = ""
    national_id_yolo = ""
    dw = args.digit_weights
    if dw is not None:
        dw = dw.expanduser().resolve()
    if dw is not None and dw.is_file() and "nid" in best:
        box, _ = best["nid"]
        x1, y1, x2, y2 = eid.expand_bbox_height_xyxy(box, args.nid_expand_scale, bgr.shape)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(bgr.shape[1], x2), min(bgr.shape[0], y2)
        if x2 > x1 and y2 > y1:
            nid_sub = bgr[y1:y2, x1:x2].copy()
            national_id_yolo = eid.nid_digits_yolo_on_crop(
                nid_sub,
                dw,
                device=device,
                conf=args.digit_conf,
                dedupe_iou=args.digit_dedupe_iou,
                reading_order=args.digit_reading_order,
            )
            national_id = eid.western_digits_only(national_id_yolo)

    decode_err = ""
    decoded: dict[str, str] = {}
    if args.decode_nid:
        try:
            decoded = eid.decode_egyptian_id(national_id)
        except ValueError as e:
            decode_err = str(e)

    print()
    print("first_name: ", first)
    print("last_name:  ", last)
    print("full_name:  ", full)
    print("address:    ", addr)
    if args.include_serial:
        print("serial:     ", serial)
    if national_id or national_id_yolo:
        print("national_id (YOLO digits):", national_id or "(empty)")
    if args.decode_nid:
        if decoded:
            for key, value in decoded.items():
                print(f"{key}: {value}")
        elif decode_err:
            print(f"(NID decode: {decode_err})")

    payload = {
        "first_name": first,
        "last_name": last,
        "full_name": full,
        "address": addr,
        "serial": serial,
        "national_id": national_id,
        "national_id_yolo_digits": national_id_yolo,
        "image_path": str(img_path),
    }
    if args.decode_nid:
        payload["nid_decode_error"] = decode_err
        payload.update({f"decoded_{k.lower().replace(' ', '_')}": v for k, v in decoded.items()})

    if args.json_out:
        p = args.json_out.expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {p}")

    if args.output:
        try:
            import pandas as pd
        except ImportError as ex:
            raise SystemExit("Install pandas openpyxl: py -m pip install pandas openpyxl") from ex
        out = args.output.expanduser().resolve()
        written = eid.write_excel_safe(pd.DataFrame([payload]), out)
        print(f"\nWrote Excel: {written}")


if __name__ == "__main__":
    main()
