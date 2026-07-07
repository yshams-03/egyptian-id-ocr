"""
Export Egyptian ID fields to Excel (wrapper around extract_id_all.py).

Default: mixed EasyOCR + Tesseract for Arabic name/address, YOLO digits for NID,
DOB filled from NID decode when the printed dob line is not read.

For the same pipeline explicitly:
  py extract_id_all.py "path\\to\\id.jpg"

Legacy Tesseract-only (no EasyOCR):
  py export_id_to_excel.py id.jpg --tesseract-only

Requires:
  py -m pip install pandas openpyxl pytesseract opencv-python ultralytics
  Optional: py -m pip install easyocr
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from egypt_nid_decode import EGYPT_NID_GOVERNORATES

BASE = Path(__file__).resolve().parent
RUNS = BASE / "runs"
DEFAULT_WEIGHTS = RUNS / "train_id_detectr_hyper" / "weights" / "best.pt"
DEFAULT_DIGIT_WEIGHTS = RUNS / "train_arabic_numbers_v2" / "weights" / "best.pt"
DATA_YAML = BASE / "egyptian_id_detectr" / "content" / "Egyptian-ID-Detectr-3" / "data.yaml"

# Arabic-Indic → ASCII (shared with extract_back.py)
ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Reuse loaded YOLO weights (reload per request costs 5–15s each).
_YOLO_CACHE: dict[str, object] = {}


def get_yolo(weights: Path | str):
    """Return a cached Ultralytics YOLO instance for this weights path."""
    from ultralytics import YOLO

    key = str(Path(weights).expanduser().resolve())
    if key not in _YOLO_CACHE:
        _YOLO_CACHE[key] = YOLO(key)
    return _YOLO_CACHE[key]


def resize_for_speed(bgr: np.ndarray, max_side: int = 880) -> np.ndarray:
    """Downscale large photos so YOLO + EasyOCR stay under ~30s total."""
    if bgr is None or bgr.size == 0:
        return bgr
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / m
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)

# Final merged Excel column order (front + back + decode)
EXCEL_ROW_COLUMNS: tuple[str, ...] = (
    "first_name",
    "last_name",
    "full_name",
    "address",
    "national_id",
    "dob",
    "serial",
    "photo_path",
    "job",
    "religion",
    "gender",
    "marital_status",
    "expiry_date",
    "issue_date",
    "back_nid",
    "decoded_birth_date",
    "decoded_century",
    "decoded_governorate",
    "decoded_gender",
    "decoded_sequence",
    "decoded_check_digit",
    "nid_mismatch_warning",
    "image_path",
)

def load_class_names() -> dict[int, str]:
    data = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    names = data.get("names") or []
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def preprocess_for_ocr(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def upscale_crop(bgr: np.ndarray, min_side: int = 96) -> np.ndarray:
    if bgr.size == 0:
        return bgr
    h, w = bgr.shape[:2]
    m = min(h, w)
    if m >= min_side:
        return bgr
    scale = min_side / max(m, 1)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)


def ocr_crop(
    bgr: np.ndarray,
    lang: str,
    psm: int = 6,
    *,
    whitelist: str | None = None,
) -> str:
    import pytesseract

    if bgr.size == 0:
        return ""
    prep = preprocess_for_ocr(bgr)
    cfg = f"--oem 3 --psm {psm}"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    text = pytesseract.image_to_string(prep, lang=lang, config=cfg)
    return " ".join(text.split()).strip()


def ocr_nid_crop(bgr: np.ndarray, min_side: int = 220) -> str:
    """Digit-only OCR on NID-like crops: strong upscale, Otsu + invert, adaptive thresh."""
    import pytesseract

    if bgr.size == 0:
        return ""
    bgr = upscale_crop(bgr, min_side=min_side)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(otsu)
    try:
        adapt = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10
        )
    except Exception:
        adapt = otsu
    images: list[np.ndarray] = [otsu, inv, blur, adapt]
    pieces: list[str] = []
    digit_cfg = "0123456789"
    for im in images:
        for psm in (7, 8, 11, 13):
            cfg = f"--oem 3 --psm {psm} -c tessedit_char_whitelist={digit_cfg}"
            try:
                t = pytesseract.image_to_string(im, lang="eng", config=cfg)
                if t.strip():
                    pieces.append(t.strip())
            except Exception:
                continue
    # de-duplicate while keeping order
    seen: set[str] = set()
    uniq: list[str] = []
    for p in pieces:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return " | ".join(uniq) if uniq else ""


def best_digit_sequence(text: str) -> str:
    """Pick best Western-digit run (Egyptian national ID is 14 digits)."""
    runs = re.findall(r"\d+", text)
    if not runs:
        return ""
    for r in runs:
        if len(r) == 14:
            return r
    runs.sort(key=len, reverse=True)
    for r in runs:
        if len(r) >= 8:
            return r
    # Prefer longer runs even if < 8 (e.g. 6–7 digit fragments) over tiny noise
    return runs[0]


def normalize_nid_digits(text: str) -> str:
    """Prefer a 14-digit or long digit run from multi-pass NID OCR."""
    best = best_digit_sequence(text)
    if best:
        return best
    return re.sub(r"[^\d]", "", text)


def _candidate_tesseract_paths() -> list[Path]:
    """Common install locations on Windows (and PATH)."""
    out: list[Path] = []
    env = os.environ.get("TESSERACT_CMD")
    if env:
        out.append(Path(env))
    which = shutil.which("tesseract")
    if which:
        out.append(Path(which))
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    pfx86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    for base in (pf, pfx86):
        out.append(Path(base) / "Tesseract-OCR" / "tesseract.exe")
    local = os.environ.get("LOCALAPPDATA")
    if local:
        out.append(Path(local) / "Programs" / "Tesseract-OCR" / "tesseract.exe")
    scoop = Path.home() / "scoop" / "shims" / "tesseract.exe"
    out.append(scoop)
    return out


def setup_tesseract() -> Path:
    """Set pytesseract's executable; raise SystemExit if Tesseract is not installed."""
    import pytesseract

    for candidate in _candidate_tesseract_paths():
        if candidate and candidate.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            try:
                subprocess.run(
                    [str(candidate), "--version"],
                    capture_output=True,
                    check=True,
                    timeout=10,
                )
            except (subprocess.CalledProcessError, FileNotFoundError, OSError):
                continue
            return candidate

    raise SystemExit(
        "Tesseract OCR is not installed (or not found).\n\n"
        "Install it, then re-run this script:\n"
        "  • Windows installer: https://github.com/UB-Mannheim/tesseract/wiki\n"
        "    During setup, enable the Arabic language pack.\n"
        "  • Or try: winget install UB-Mannheim.TesseractOCR\n\n"
        "If Tesseract is installed in a custom folder, set:\n"
        '  $env:TESSERACT_CMD = "C:\\\\path\\\\to\\\\tesseract.exe"\n'
    )


EXPORT_FIELD_LABELS = (
    "firstName",
    "lastName",
    "address",
    "nid",
    "nid_back",
    "dob",
    "serial",
)


def best_boxes_by_label(
    xyxy: np.ndarray, cls: np.ndarray, conf: np.ndarray, id_to_name: dict[int, str]
) -> dict[str, tuple[np.ndarray, float]]:
    """One box per canonical label (highest conf). Skips invalid_* keys for export keys."""
    by_label: dict[str, list[tuple[float, np.ndarray]]] = defaultdict(list)
    for i in range(len(cls)):
        name = id_to_name.get(int(cls[i]), "")
        if not name or name.startswith("invalid_"):
            continue
        by_label[name].append((float(conf[i]), xyxy[i].copy()))

    out: dict[str, tuple[np.ndarray, float]] = {}
    for label, pairs in by_label.items():
        pairs.sort(key=lambda x: -x[0])
        best_conf, best_xy = pairs[0]
        out[label] = (best_xy, best_conf)
    return out


def best_boxes_with_invalid_fallback(
    xyxy: np.ndarray, cls: np.ndarray, conf: np.ndarray, id_to_name: dict[int, str]
) -> tuple[dict[str, tuple[np.ndarray, float]], bool]:
    """If no valid export fields, use highest-conf ``invalid_<field>`` boxes for OCR."""
    best = best_boxes_by_label(xyxy, cls, conf, id_to_name)
    if any(k in best for k in EXPORT_FIELD_LABELS):
        return best, False

    by_canon: dict[str, list[tuple[float, np.ndarray]]] = defaultdict(list)
    for i in range(len(cls)):
        name = id_to_name.get(int(cls[i]), "")
        if not name.startswith("invalid_"):
            continue
        canon = name[len("invalid_") :]
        if canon not in EXPORT_FIELD_LABELS:
            continue
        by_canon[canon].append((float(conf[i]), xyxy[i].copy()))

    if not by_canon:
        return best, False

    out = dict(best)
    for label, pairs in by_canon.items():
        pairs.sort(key=lambda x: -x[0])
        best_conf, best_xy = pairs[0]
        out[label] = (best_xy, best_conf)
    return out, True


def crop_id_card_region(
    bgr: np.ndarray, card_weights: Path, *, conf: float, device: str
) -> tuple[np.ndarray, str]:
    """Crop best front/back card box from national-id card detector (orientation model)."""
    from ultralytics import YOLO

    model = get_yolo(card_weights)
    r = model.predict(source=bgr, conf=conf, device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return bgr, ""

    names = model.names
    best_i: int | None = None
    best_c = -1.0
    best_name = ""
    for i in range(len(r.boxes)):
        c = int(r.boxes.cls[i])
        label = names.get(c, names[c]) if isinstance(names, dict) else str(c)
        label = str(label)
        if not (label.startswith("front-") or label.startswith("back-")):
            continue
        score = float(r.boxes.conf[i])
        if score > best_c:
            best_c, best_i, best_name = score, i, label
    if best_i is None:
        best_i = int(np.argmax(r.boxes.conf.cpu().numpy()))
        c = int(r.boxes.cls[best_i])
        best_name = names.get(c, names[c]) if isinstance(names, dict) else str(c)

    xyxy = r.boxes.xyxy[best_i].cpu().numpy()
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return bgr, str(best_name)
    return bgr[y1:y2, x1:x2].copy(), str(best_name)


def crop_xyxy(img: np.ndarray, xyxy: np.ndarray, pad: int) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return np.array([])
    return img[y1:y2, x1:x2]


# --- Helpers aligned with egyptian_id_ocr.ipynb (Colab) — decode, bbox expand, simple OCR, YOLO digits ---

_ARABIC_INDIC_TO_WESTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def western_digits_only(s: str) -> str:
    s = str(s).translate(_ARABIC_INDIC_TO_WESTERN)
    return re.sub(r"[^\d]", "", s)


def plausible_dob_ocr(text: str) -> bool:
    """True when printed DOB OCR looks date-like (not punctuation noise)."""
    if not (text or "").strip():
        return False
    if re.search(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", text):
        return True
    if re.search(r"\d{1,2}[-/.]\d{1,2}[-/.]\d{4}", text):
        return True
    digits = western_digits_only(text)
    if len(digits) >= 6:
        return True
    ar_digit_count = sum(1 for c in text if "\u0660" <= c <= "\u0669")
    return ar_digit_count >= 6


def merge_serial_ocr(tesseract: str, easyocr_text: str | None = None) -> str:
    """Serial often starts with Roman 'II'; Tesseract reads as '11'. Merge Tesseract + EasyOCR."""
    t = re.sub(r"\s+", "", (tesseract or "").strip())
    e = re.sub(r"\s+", "", (easyocr_text or "").strip()) if easyocr_text else ""

    def ii_plus_digits(s: str) -> str | None:
        m = re.match(r"^([1lI|]{2})(\d+)$", s, re.I)
        if m and len(m.group(2)) >= 6:
            return "II" + m.group(2)
        return None

    if e:
        x = ii_plus_digits(e)
        if x:
            return x
        if len(e) >= 3 and e[:2].upper() == "II" and e[2:].isdigit():
            return "II" + e[2:]
    x = ii_plus_digits(t)
    if x:
        return _fix_serial_prefix(x)
    return _fix_serial_prefix(e or t)


def _fix_serial_prefix(s: str) -> str:
    """Common OCR confusions at start of serial: O→0, lone I→1 (not II)."""
    if not s:
        return s
    u = s.upper()
    if len(s) >= 2 and u[0] == "O" and s[1].isdigit():
        return "0" + s[1:]
    if len(s) >= 2 and u[0] == "I" and s[1].isdigit() and not u.startswith("II"):
        return "1" + s[1:]
    return s


def clean_address_text(address: str, *, strip_digits: bool = False) -> str:
    """Normalize whitespace; optionally strip all digit runs from address OCR."""
    if not address:
        return ""
    if strip_digits:
        return remove_numbers(address)
    return " ".join(address.split()).strip()


def row_for_excel(row: dict[str, str]) -> dict[str, str]:
    """Project row dict onto the canonical Excel schema."""
    out: dict[str, str] = {}
    for k in EXCEL_ROW_COLUMNS:
        out[k] = str(row.get(k, "") or "")
    return out


def extract_photo(
    bgr: np.ndarray,
    best_boxes: dict,
    output_dir: Path,
    stem: str,
    *,
    pad: int = 4,
) -> str | None:
    """Save face crop from YOLO `photo` box or left-region fallback."""
    h, w = bgr.shape[:2]
    if "photo" in best_boxes:
        crop = crop_xyxy(bgr, best_boxes["photo"][0], pad)
    else:
        x1, y1 = 0, int(h * 0.15)
        x2, y2 = int(w * 0.28), int(h * 0.72)
        crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}_photo.jpg"
    cv2.imwrite(str(out_path), crop)
    return str(out_path.resolve())


def ocr_nid_easyocr_digits(bgr: np.ndarray, reader: object, *, min_side: int = 220) -> str:
    """Fallback NID read: EasyOCR on nid strip, Arabic-Indic → ASCII, keep digits."""
    if bgr.size == 0:
        return ""
    big = upscale_crop(bgr, min_side=min_side)
    rgb = cv2.cvtColor(big, cv2.COLOR_BGR2RGB)
    pieces: list[str] = []
    for item in reader.readtext(rgb, detail=1, paragraph=False):
        if len(item) < 2:
            continue
        t = str(item[1]).translate(ARABIC_INDIC)
        d = re.sub(r"[^\d]", "", t)
        if d:
            pieces.append(d)
    return "".join(pieces)


def remove_numbers(text: str) -> str:
    """Strip Western and Arabic-Indic digit runs (notebook `remove_numbers`)."""
    if not text:
        return ""
    t = text.translate(_ARABIC_INDIC_TO_WESTERN)
    return re.sub(r"\d+", "", t).strip()


def decode_egyptian_id(id_number: str) -> dict[str, str]:
    """
    Decode 14-digit Egyptian national ID (Western digits).
    Logic aligned with https://github.com/Eslam2014/extract-information-from-eg-national-id
    """
    from egypt_nid_decode import NidDecodeError, decode_egyptian_nid

    try:
        return decode_egyptian_nid(id_number).as_export_dict()
    except NidDecodeError as e:
        raise ValueError(str(e)) from e


def expand_bbox_height_xyxy(
    xyxy: np.ndarray,
    scale: float = 1.2,
    image_shape: tuple[int, ...] | None = None,
) -> tuple[int, int, int, int]:
    """Expand box height around vertical center; clamp to image height (notebook logic)."""
    x1, y1, x2, y2 = [int(round(float(v))) for v in np.asarray(xyxy).ravel()[:4]]
    h_img = int(image_shape[0]) if image_shape is not None and len(image_shape) >= 1 else 10**9
    height = y2 - y1
    center_y = y1 + height // 2
    new_height = max(1, int(height * scale))
    new_y1 = max(center_y - new_height // 2, 0)
    new_y2 = min(center_y + new_height // 2, h_img)
    return x1, new_y1, x2, new_y2


def preprocess_image_notebook(cropped_bgr: np.ndarray) -> np.ndarray:
    """Fixed 127 threshold (same idea as original Colab `preprocess_image`)."""
    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary


def extract_text_notebook(image_bgr: np.ndarray, bbox, lang: str = "ara") -> str:
    """Notebook-style: threshold 127 + Tesseract PSM 6 on the crop."""
    import pytesseract

    x1, y1, x2, y2 = [int(round(float(v))) for v in np.asarray(bbox).ravel()[:4]]
    cropped = image_bgr[y1:y2, x1:x2]
    if cropped.size == 0:
        return ""
    prep = preprocess_image_notebook(cropped)
    text = pytesseract.image_to_string(prep, lang=lang, config=r"--oem 3 --psm 6")
    return " ".join(text.split()).strip()


def nid_digits_yolo_on_crop(
    nid_crop_bgr: np.ndarray,
    digit_weights: Path,
    *,
    device: str,
    conf: float = 0.25,
    imgsz: int = 640,
    dedupe_iou: float = 0.45,
    reading_order: str = "ltr",
    model: object | None = None,
) -> str:
    """Run Arabic-digit YOLO on an nid strip crop; return sorted digit string (like notebook `detect_national_id`)."""
    from extract_nid_digits import dedupe_detections, reading_order_indices

    if nid_crop_bgr.size == 0 or not digit_weights.is_file():
        return ""
    if model is None:
        model = get_yolo(digit_weights)
    r = model.predict(
        source=nid_crop_bgr, conf=conf, device=device, imgsz=imgsz, verbose=False
    )[0]
    if r.boxes is None or len(r.boxes) == 0:
        return ""
    xyxy = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    cconf = r.boxes.conf.cpu().numpy()
    if dedupe_iou > 0:
        xyxy, cls, cconf = dedupe_detections(xyxy, cls, cconf, dedupe_iou)
    order = reading_order_indices(xyxy, reading_order)
    names = model.names
    chars: list[str] = []
    for idx in order:
        c = int(cls[idx])
        label = names.get(c, names[c]) if isinstance(names, dict) else str(c)
        chars.append(str(label))
    return "".join(chars)


def write_excel_safe(df, out_path: Path) -> Path:
    """Write xlsx; if target is locked (e.g. open in Excel), use a timestamped filename."""
    import pandas as pd

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_excel(out_path, index=False, engine="openpyxl")
        return out_path
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = out_path.parent / f"{out_path.stem}_{stamp}{out_path.suffix}"
        df.to_excel(alt, index=False, engine="openpyxl")
        print(
            "Could not write (file may be open in Excel). Saved a new copy:\n"
            f"  {alt}\n"
            "Close the workbook and delete the old file if you no longer need it."
        )
        return alt


def main() -> None:
    parser = argparse.ArgumentParser(description="Export national ID, name, address (OCR) to Excel.")
    parser.add_argument("image", type=Path, help="Path to ID image (quote if spaces).")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Field YOLO weights .pt")
    parser.add_argument("--output", type=Path, default=None, help=".xlsx output path.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON export.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--pad", type=int, default=6, help="Pixels to pad each crop.")
    parser.add_argument(
        "--lang",
        default="ara+eng",
        help="(Tesseract-only) languages for text fields.",
    )
    parser.add_argument("--min-crop-side", type=int, default=200)
    parser.add_argument("--nid-min-side", type=int, default=220)
    parser.add_argument("--save-crops", type=Path, default=None)
    parser.add_argument("--digit-weights", type=Path, default=None)
    parser.add_argument("--nid-expand-scale", type=float, default=1.5)
    parser.add_argument("--digit-conf", type=float, default=0.25)
    parser.add_argument("--digit-reading-order", choices=("auto", "ltr", "row_col"), default="ltr")
    parser.add_argument("--digit-dedupe-iou", type=float, default=0.45)
    parser.add_argument(
        "--decode-nid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decode birth date / governorate / gender from NID (default: on).",
    )
    parser.add_argument(
        "--no-dob-from-nid",
        action="store_true",
        help="Leave dob empty when printed dob OCR fails.",
    )
    parser.add_argument("--address-strip-digits", action="store_true")
    parser.add_argument("--serial-lang", default="eng")
    parser.add_argument("--use-notebook-field-ocr", action="store_true")
    parser.add_argument(
        "--tesseract-only",
        action="store_true",
        help="Skip EasyOCR; use Tesseract only (old behavior; poor Arabic on screenshots).",
    )
    parser.add_argument(
        "--engine",
        choices=("tesseract", "easyocr", "mixed"),
        default="mixed",
        help="Name/address OCR (ignored when --tesseract-only).",
    )
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

    if not args.tesseract_only:
        from extract_id_all import DEFAULT_DIGIT_WEIGHTS, ExtractConfig, extract_all

        out_path = args.output
        if out_path is None:
            out_path = BASE / "runs" / "id_export" / (image_path.stem + "_fields.xlsx")
        digit = args.digit_weights
        if digit is None and DEFAULT_DIGIT_WEIGHTS.is_file():
            digit = DEFAULT_DIGIT_WEIGHTS
        extract_all(
            ExtractConfig(
                image=image_path,
                field_weights=args.weights,
                digit_weights=digit,
                output=out_path,
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
                decode_nid=args.decode_nid,
                dob_from_nid=not args.no_dob_from_nid,
                strip_address_digits=args.address_strip_digits,
                use_notebook_field_ocr=args.use_notebook_field_ocr,
                serial_lang=args.serial_lang,
            )
        )
        return

    _main_tesseract_only(args, image_path)


def _main_tesseract_only(args: argparse.Namespace, image_path: Path) -> None:
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}")

    out_path = args.output
    if out_path is None:
        out_path = BASE / "runs" / "id_export" / (image_path.stem + "_fields.xlsx")
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit("Install pandas: pip install pandas openpyxl") from e

    import torch
    from ultralytics import YOLO

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"

    try:
        import pytesseract  # noqa: F401
    except ImportError as e:
        raise SystemExit("Install pytesseract: py -m pip install pytesseract") from e

    tess = setup_tesseract()
    print(f"Using Tesseract: {tess}")

    id_to_name = load_class_names()
    model = YOLO(str(weights))
    results = model.predict(source=str(image_path), conf=args.conf, device=device, verbose=False)
    r = results[0]
    img = r.orig_img
    if img is None:
        img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit("Could not read image.")

    if r.boxes is None or len(r.boxes) == 0:
        row = {k: "" for k in ("national_id", "first_name", "last_name", "full_name", "address", "dob", "serial")}
        row["image_path"] = str(image_path)
        if args.decode_nid:
            row["decoded_birth_date"] = ""
            row["decoded_century"] = ""
            row["decoded_governorate"] = ""
            row["decoded_gender"] = ""
            row["nid_decode_error"] = "no detections"
        written = write_excel_safe(pd.DataFrame([row]), out_path)
        raise SystemExit(f"No detections. Wrote empty row to {written}")

    xyxy = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()
    best = best_boxes_by_label(xyxy, cls, conf, id_to_name)

    if args.save_crops:
        crop_dir = args.save_crops.expanduser().resolve()
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
            c = crop_xyxy(img, best[lab][0], args.pad)
            if c.size == 0:
                continue
            cv2.imwrite(str(crop_dir / f"{lab}.png"), c)
        print(f"Saved field crops to: {crop_dir}")

    def ocr_label(label: str) -> str:
        if label not in best:
            return ""
        box, _ = best[label]
        crop = crop_xyxy(img, box, args.pad)
        if label == "nid":
            return ocr_nid_crop(crop, min_side=args.nid_min_side)
        if label == "serial":
            big = upscale_crop(crop, min_side=args.min_crop_side)
            return ocr_crop(big, args.serial_lang)
        if args.use_notebook_field_ocr and label in ("firstName", "lastName", "address", "dob"):
            return extract_text_notebook(img, box, lang="ara")
        big = upscale_crop(crop, min_side=args.min_crop_side)
        return ocr_crop(big, args.lang)

    first = ocr_label("firstName")
    last = ocr_label("lastName")
    addr = ocr_label("address")
    if args.address_strip_digits:
        addr = remove_numbers(addr)
    nid_raw = ocr_label("nid")
    nid_back_raw = ""
    if "nid_back" in best:
        bb, _ = best["nid_back"]
        cr = crop_xyxy(img, bb, args.pad)
        nid_back_raw = ocr_nid_crop(cr, min_side=args.nid_min_side)
    combined_nid = f"{nid_raw} {nid_back_raw}".strip()
    nid_digits = normalize_nid_digits(combined_nid)

    nid_yolo = ""
    dw = args.digit_weights
    if dw is not None:
        dw = dw.expanduser().resolve()
    if dw is not None and dw.is_file() and "nid" in best:
        box, _ = best["nid"]
        x1, y1, x2, y2 = expand_bbox_height_xyxy(box, args.nid_expand_scale, img.shape)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)
        if x2 > x1 and y2 > y1:
            nid_sub = img[y1:y2, x1:x2].copy()
            nid_yolo = nid_digits_yolo_on_crop(
                nid_sub,
                dw,
                device=device,
                conf=args.digit_conf,
                dedupe_iou=args.digit_dedupe_iou,
                reading_order=args.digit_reading_order,
            )
    wy = western_digits_only(nid_yolo)
    wt = western_digits_only(nid_digits)
    if wy and (len(wy) == 14 or len(wy) > len(wt)):
        nid_digits = wy

    dob = ocr_label("dob")
    serial = merge_serial_ocr(ocr_label("serial"), None)
    row = {
        "national_id": nid_digits or re.sub(r"[^\d]", "", combined_nid),
        "national_id_raw_ocr": combined_nid or nid_raw,
        "national_id_yolo_digits": nid_yolo,
        "first_name": first,
        "last_name": last,
        "full_name": f"{first} {last}".strip(),
        "address": addr,
        "dob": dob,
        "serial": serial,
        "image_path": str(image_path),
    }
    if args.decode_nid:
        try:
            dec = decode_egyptian_id(row["national_id"])
            row["decoded_birth_date"] = dec["Birth Date"]
            row["decoded_century"] = dec["Century"]
            row["decoded_governorate"] = dec["Governorate"]
            row["decoded_gender"] = dec["Gender"]
            row["decoded_sequence"] = dec.get("Sequence", "")
            row["decoded_check_digit"] = dec.get("Check Digit", "")
            row["nid_decode_error"] = ""
        except ValueError as e:
            row["decoded_birth_date"] = ""
            row["decoded_century"] = ""
            row["decoded_governorate"] = ""
            row["decoded_gender"] = ""
            row["decoded_sequence"] = ""
            row["decoded_check_digit"] = ""
            row["nid_decode_error"] = str(e)

    bd = (row.get("decoded_birth_date") or "").strip()
    if args.decode_nid and bd and not args.no_dob_from_nid:
        row["dob"] = bd

    written = write_excel_safe(pd.DataFrame([row]), out_path)
    print(f"Wrote: {written}")
    if len(nid_digits) < 10:
        print(
            "Tip: few NID digits read — the `nid` box may be misaligned. "
            "Run with --save-crops runs\\id_export\\debug and open nid.png; "
            "compare with predict_my_id.py overlay."
        )
    for k, v in row.items():
        if k == "image_path":
            continue
        preview = (v[:120] + "…") if len(str(v)) > 120 else v
        print(f"  {k}: {preview!r}")
    if args.decode_nid and row.get("nid_decode_error"):
        print(f"  (decode skipped: {row['nid_decode_error']})")
    if args.decode_nid and bd and not args.no_dob_from_nid:
        print("  (dob taken from national_id decode)")


if __name__ == "__main__":
    main()
