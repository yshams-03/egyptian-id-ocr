"""
Egyptian national ID card — back side extraction (EasyOCR + regex + layout heuristics).

Used by extract_id_all.py when --back, --back-image, or --auto-detect-side routes to back.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Arabic-Indic digits → ASCII (٠١٢٣٤٥٦٧٨٩ → 0123456789)
ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

GENDER_KEYWORDS = {"ذكر": "ذكر", "أنثى": "أنثى", "انثى": "أنثى"}
RELIGION_KEYWORDS = {"مسلم": "مسلم", "مسيحي": "مسيحي", "مسيحى": "مسيحي"}
MARITAL_KEYWORDS = {
    "أعزب": "أعزب",
    "اعزب": "أعزب",
    "متزوج": "متزوج",
    "مطلق": "مطلق",
    "أرمل": "أرمل",
    "ارمل": "أرمل",
}

_STATUS_WORDS = set(GENDER_KEYWORDS) | set(RELIGION_KEYWORDS) | set(MARITAL_KEYWORDS)
_STATUS_WORDS.update({"اعزب", "انثى", "مسم", "مسلم", "مسيحي", "مسيحى"})

# Front header phrases (جمهورية مصر العربية)
_FRONT_MARKERS = ("جمهورية", "بطاقة", "تحقيق", "الشخصية")
# Back markers (البطاقة سارية حتى = valid until)
_BACK_MARKERS = ("سارية", "البطاقة", "مهندس", "مسلم", "مسيح", "طالب")


def western_digits(s: str) -> str:
    return re.sub(r"[^\d]", "", str(s).translate(ARABIC_INDIC))


def normalize_date_iso(raw: str) -> str:
    """Convert YYYY/MM or YYYY-MM (Arabic-Indic ok) to ISO date or YYYY-MM."""
    t = str(raw).strip().translate(ARABIC_INDIC)
    t = t.replace(".", "/").replace("-", "/")
    parts = [p for p in t.split("/") if p]
    if len(parts) >= 3:
        y, m, d = parts[0].zfill(4)[:4], parts[1].zfill(2)[:2], parts[2].zfill(2)[:2]
        if len(y) == 4 and y.isdigit() and m.isdigit() and d.isdigit():
            return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
    if len(parts) == 2 and len(parts[0]) == 4 and parts[0].isdigit():
        return f"{parts[0]}-{parts[1].zfill(2)}"
    return t


def enhance_for_ocr(bgr: np.ndarray) -> np.ndarray:
    """CLAHE on grayscale — helps glare / low contrast on back cards."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _bbox_center(bbox: Any) -> tuple[float, float]:
    pts = np.asarray(bbox, dtype=float)
    xc = float(np.mean(pts[:, 0]))
    yc = float(np.mean(pts[:, 1]))
    return xc, yc


def _sort_blocks_rtl_top_bottom(
    results: list[tuple[Any, str, float]], img_h: int
) -> list[tuple[Any, str, float]]:
    """Sort OCR blocks: top→bottom, then right→left (RTL reading order)."""
    return sorted(
        results,
        key=lambda r: (
            round(_bbox_center(r[0])[1] / max(img_h * 0.05, 1)),
            -_bbox_center(r[0])[0],
        ),
    )


def _is_status_token(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if t in _STATUS_WORDS:
        return True
    for kw in GENDER_KEYWORDS:
        if kw in t:
            return True
    for kw in MARITAL_KEYWORDS:
        if kw in t:
            return True
    if _normalize_religion_token(t):
        return True
    return False


def _normalize_religion_token(text: str) -> str:
    t = text.strip()
    if t in RELIGION_KEYWORDS:
        return RELIGION_KEYWORDS[t]
    if re.fullmatch(r"م[سص][مم]", t) or t in ("مسم", "مسى", "مسل"):
        return "مسلم"
    if "مسل" in t or (t.startswith("مس") and len(t) <= 4):
        return "مسلم"
    if "مسيح" in t:
        return "مسيحي"
    return ""


def _normalize_marital_token(text: str) -> str:
    t = text.strip()
    if t in MARITAL_KEYWORDS:
        return MARITAL_KEYWORDS[t]
    if "عزب" in t or t == "اعزب":
        return "أعزب"
    return ""


def _extract_job_line(
    blocks: list[tuple[Any, str, float]], img_h: int, img_w: int
) -> str:
    """
    Job title sits above the gender/religion/marital row (e.g. طالب, مهندس بترول).
    """
    status_ys = [
        _bbox_center(b[0])[1] / img_h
        for b in blocks
        if _is_status_token(b[1]) and 0.18 < _bbox_center(b[0])[1] / img_h < 0.5
    ]
    status_cut = min(status_ys) - 0.04 if status_ys else 0.28

    candidates: list[tuple[float, float, str]] = []
    for bbox, text, conf in blocks:
        yc = _bbox_center(bbox)[1] / img_h
        if yc >= status_cut or yc < 0.04:
            continue
        t = text.strip()
        if not _ARABIC_RE.search(t) or _is_status_token(t):
            continue
        if len(western_digits(t)) >= 8:
            continue
        candidates.append((yc, -conf, t))

    if not candidates:
        return ""

    candidates.sort()
    # Prefer the block just above the status row (largest y among candidates)
    line_y = candidates[-1][0]
    words = [t for yc, _, t in candidates if abs(yc - line_y) <= 0.06]
    return " ".join(words).strip()


def _extract_status_row(
    blocks: list[tuple[Any, str, float]], img_h: int, img_w: int
) -> dict[str, str]:
    """Gender / religion / marital from the middle row (spatial + keyword)."""
    out = {"gender": "", "religion": "", "marital_status": ""}
    row_blocks = []
    for bbox, text, conf in blocks:
        yc = _bbox_center(bbox)[1] / img_h
        xc = _bbox_center(bbox)[0] / img_w
        if 0.18 <= yc <= 0.48 and conf >= 0.2:
            row_blocks.append((xc, yc, text.strip(), conf))

    if not row_blocks:
        return out

    for xc, _yc, text, _ in sorted(row_blocks, key=lambda r: -r[0]):
        for kw, val in GENDER_KEYWORDS.items():
            if kw in text:
                out["gender"] = val
        rel = _normalize_religion_token(text)
        if rel:
            out["religion"] = rel
        mar = _normalize_marital_token(text)
        if mar:
            out["marital_status"] = mar

    # Spatial fallback when OCR splits columns (RTL: right=gender, center=religion, left=marital)
    if not out["gender"]:
        for xc, _, text, _ in row_blocks:
            if xc > 0.62 and any(k in text for k in GENDER_KEYWORDS):
                out["gender"] = GENDER_KEYWORDS.get(
                    next(k for k in GENDER_KEYWORDS if k in text), "ذكر"
                )
    if not out["religion"]:
        for xc, _, text, _ in row_blocks:
            if 0.35 < xc < 0.72:
                rel = _normalize_religion_token(text)
                if rel:
                    out["religion"] = rel
    if not out["marital_status"]:
        for xc, _, text, _ in row_blocks:
            if xc < 0.48:
                mar = _normalize_marital_token(text)
                if mar:
                    out["marital_status"] = mar

    return out


def _stitch_back_nid(
    blocks: list[tuple[Any, str, float]], img_h: int, img_w: int
) -> str:
    """Concatenate digit runs in the top band (RTL), preferring 14-digit NID."""
    digit_parts: list[tuple[float, str]] = []
    for bbox, text, conf in blocks:
        yc = _bbox_center(bbox)[1] / img_h
        xc = _bbox_center(bbox)[0] / img_w
        if yc > 0.38:
            continue
        digits = western_digits(text)
        if len(digits) >= 4:
            digit_parts.append((xc, digits))

    if not digit_parts:
        return ""

    digit_parts.sort(key=lambda x: -x[0])
    stitched = "".join(d for _, d in digit_parts)
    if len(stitched) >= 14:
        return stitched[:14] if len(stitched) >= 14 else stitched
    m = re.search(r"\d{13,15}", stitched)
    return m.group(0) if m else stitched


def _ocr_crop(
    bgr: np.ndarray, reader: Any, y0: float, y1: float, *, scale: float = 2.0
) -> list[tuple[Any, str, float]]:
    h, w = bgr.shape[:2]
    y_a, y_b = int(h * y0), int(h * y1)
    crop = bgr[max(0, y_a) : min(h, y_b), :]
    if crop.size == 0:
        return []
    if scale != 1.0:
        crop = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    out: list[tuple[Any, str, float]] = []
    for item in reader.readtext(rgb, detail=1, paragraph=False):
        if len(item) < 3:
            continue
        bbox, text, conf = item[0], str(item[1]).strip(), float(item[2])
        if text and conf >= 0.25:
            out.append((bbox, text, conf))
    return out


def read_easyocr_blocks(
    bgr: np.ndarray,
    reader: Any,
    *,
    min_conf: float = 0.3,
    retry_enhanced: bool = True,
    fast: bool = False,
) -> list[tuple[Any, str, float]]:
    """Run EasyOCR; optional upscale / CLAHE retry (skipped in fast mode)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _read(im: np.ndarray) -> list[tuple[Any, str, float]]:
        out: list[tuple[Any, str, float]] = []
        for item in reader.readtext(im, detail=1, paragraph=False):
            if len(item) < 3:
                continue
            bbox, text, conf = item[0], str(item[1]).strip(), float(item[2])
            if text and conf >= min_conf:
                out.append((bbox, text, conf))
        return out

    blocks = _read(rgb)
    if fast:
        return blocks

    if len(blocks) < 12:
        up = cv2.resize(rgb, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        blocks2 = _read(up)
        if len(blocks2) > len(blocks):
            blocks = blocks2

    if retry_enhanced and blocks:
        mean_c = sum(b[2] for b in blocks) / len(blocks)
        if mean_c < 0.45 or len(blocks) < 10:
            g = enhance_for_ocr(bgr)
            g3 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            upg = cv2.resize(g3, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            blocks2 = _read(cv2.cvtColor(upg, cv2.COLOR_BGR2RGB))
            if len(blocks2) > len(blocks):
                blocks = blocks2
    return blocks


def detect_card_side(
    image_path: Path | str,
    reader: Any | None = None,
    *,
    bgr: np.ndarray | None = None,
) -> str:
    """
    Return 'front' or 'back' from layout / quick OCR cues.
    Back: dense barcode strip (high variance in bottom 25%) or back keywords.
    """
    if bgr is None:
        bgr = cv2.imread(str(image_path))
    if bgr is None:
        return "front"

    h = bgr.shape[0]
    bottom = bgr[int(h * 0.75) :, :]
    gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
    if float(gray.std()) > 60:
        return "back"

    if reader is not None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            sample = reader.readtext(rgb, detail=0, paragraph=True)
            text = " ".join(sample) if sample else ""
        except Exception:
            text = ""
        if any(m in text for m in _BACK_MARKERS):
            return "back"
        if any(m in text for m in _FRONT_MARKERS):
            return "front"

    return "front"


def extract_back_fields(
    image_path: Path | str,
    reader: Any,
    *,
    bgr: np.ndarray | None = None,
    front_nid: str = "",
    fast: bool = False,
) -> dict[str, str]:
    """
    Extract back-card fields via full-image EasyOCR + spatial/regex parsing.
    """
    path = Path(image_path)
    if bgr is None:
        bgr = cv2.imread(str(path))
    if bgr is None:
        return _empty_back_fields()

    if fast:
        import export_id_to_excel as eid

        bgr = eid.resize_for_speed(bgr, max_side=880)

    blocks = read_easyocr_blocks(bgr, reader, fast=fast, min_conf=0.35 if fast else 0.3)
    if not blocks:
        return _empty_back_fields()

    img_h, img_w = bgr.shape[:2]
    sorted_blocks = _sort_blocks_rtl_top_bottom(blocks, img_h)
    lines = [t for _, t, c in sorted_blocks if c > 0.3]
    all_text = " ".join(lines)

    fields: dict[str, str] = _empty_back_fields()

    # Expiry line — full text then cropped retry below "حتى"
    expiry_match = re.search(
        r"البطاقة\s*سارية\s*حتى\s*([\d٠-٩]{2,4}[/\.\-][\d٠-٩]{1,2}(?:[/\.\-][\d٠-٩]{1,2})?)",
        all_text,
    )
    if expiry_match:
        fields["expiry_date"] = normalize_date_iso(expiry_match.group(1))

    date_pat = re.compile(
        r"([\d٠-٩]{4})[/\.\-]([\d٠-٩]{1,2})(?:[/\.\-]([\d٠-٩]{1,2}))?"
    )
    dated: list[tuple[float, float, str]] = []
    for bbox, text, conf in sorted_blocks:
        for m in date_pat.finditer(text):
            raw = m.group(0)
            yc, xc = _bbox_center(bbox)[1], _bbox_center(bbox)[0]
            dated.append((yc, -xc, raw))
    dated.sort(key=lambda x: (x[0], x[1]))
    if dated and not fields["expiry_date"]:
        fields["expiry_date"] = normalize_date_iso(dated[-1][2])
    if len(dated) >= 2:
        fields["issue_date"] = normalize_date_iso(dated[0][2])
    elif len(dated) == 1 and not fields["issue_date"]:
        fields["issue_date"] = normalize_date_iso(dated[0][2])

    if not fast and not fields["expiry_date"] and "حتى" in all_text:
        extra = _ocr_crop(bgr, reader, 0.40, 0.58, scale=2.5)
        extra_text = " ".join(t for _, t, _ in extra)
        em = re.search(
            r"حتى\s*([\d٠-٩]{2,4}[/\.\-][\d٠-٩]{1,2}(?:[/\.\-][\d٠-٩]{1,2})?)",
            extra_text,
        )
        if em:
            fields["expiry_date"] = normalize_date_iso(em.group(1))
        else:
            for _, t, _ in extra:
                for m in date_pat.finditer(t):
                    fields["expiry_date"] = normalize_date_iso(m.group(0))
                    break
                if fields["expiry_date"]:
                    break

    fields["back_nid"] = _stitch_back_nid(sorted_blocks, img_h, img_w)
    if not fields["back_nid"]:
        for _, text, _ in sorted_blocks:
            m = re.search(r"[\d٠-٩]{13,15}", text.replace(" ", ""))
            if m:
                fields["back_nid"] = western_digits(m.group())
                break
    fn_hint = western_digits(front_nid)
    if len(western_digits(fields["back_nid"])) < 13 and len(fn_hint) == 14:
        fields["back_nid"] = fn_hint

    fields["job"] = _extract_job_line(sorted_blocks, img_h, img_w)
    fields.update(_extract_status_row(sorted_blocks, img_h, img_w))

    return fields


def _empty_back_fields() -> dict[str, str]:
    return {
        "job": "",
        "religion": "",
        "gender": "",
        "marital_status": "",
        "expiry_date": "",
        "issue_date": "",
        "back_nid": "",
    }


def merge_front_back(
    front: dict[str, str],
    back: dict[str, str],
) -> dict[str, str]:
    """Merge front + back dicts; set nid_mismatch_warning when IDs differ."""
    row = {**front, **back}
    fn = western_digits(front.get("national_id", ""))
    bn = western_digits(back.get("back_nid", ""))
    if fn and bn and fn != bn:
        row["nid_mismatch_warning"] = "true"
    else:
        row["nid_mismatch_warning"] = ""
    return row
