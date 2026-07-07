#!/usr/bin/env python
"""Deep diagnosis of remaining held-out name/address failures (report_20260707_143537)."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import export_id_to_excel as eid
import extract_name_address as ena
from extract_id_all import (
    DEFAULT_FIELD_WEIGHTS,
    NAME_OCR_MAX_SIDE,
    NAME_OCR_MIN_SIDE,
    ExtractConfig,
    _init_ocr,
    extract_front,
)
from tests.ground_truth import discover_test_cases, load_ground_truth
from tests.id_metrics import DEFAULT_CER_THRESHOLD, cer, exact_match, normalize_arabic_text
from tests.labeling.sources import HELD_OUT_SOURCES

OUT_DIR = ROOT / "runs" / "diagnose_heldout_failures"
PAD = 6
MIN_SIDE_NAME = NAME_OCR_MIN_SIDE
MAX_SIDE_NAME = NAME_OCR_MAX_SIDE
MIN_SIDE_ADDR = 120
MAX_SIDE_ADDR = 520


@dataclass
class EasyBox:
    text: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class FieldDiag:
    field: str
    expected: str
    pipeline_actual: str
    easyocr_joined: str
    easyocr_boxes: list[EasyBox]
    tess_best: str
    tess_by_psm: dict[str, str]
    easyocr_cer: float
    tess_cer: float
    batch_text: str | None = None
    batch_differs: bool | None = None
    crop_raw_shape: tuple[int, int] = (0, 0)
    crop_upscaled_shape: tuple[int, int] = (0, 0)
    box_xyxy: list[float] = field(default_factory=list)
    visual_notes: str = ""
    boundary_notes: str = ""
    verdict: str = ""
    verdict_reason: str = ""


@dataclass
class SampleDiag:
    image: str
    name_fail: bool
    address_fail: bool
    fields: list[FieldDiag] = field(default_factory=list)
    code_path: str = ""


def _expected_name(gt: dict) -> str:
    return (
        (gt.get("full_name") or "").strip()
        or (gt.get("name") or "").strip()
        or f"{gt.get('first_name', '')} {gt.get('last_name', '')}".strip()
    )


def _upscale_for_field(bgr: np.ndarray, *, min_side: int, max_side: int) -> np.ndarray:
    big = eid.upscale_crop(bgr, min_side=max(min_side, 120))
    return eid.resize_for_speed(big, max_side=max_side)


def _easyocr_detail(bgr: np.ndarray, reader, *, min_side: int, max_side: int) -> tuple[str, list[EasyBox]]:
    up = _upscale_for_field(bgr, min_side=min_side, max_side=max_side)
    rgb = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
    boxes: list[EasyBox] = []
    for item in reader.readtext(rgb, detail=1, paragraph=False):
        if len(item) < 2:
            continue
        bbox, text, conf = item[0], str(item[1]), float(item[2]) if len(item) > 2 else 1.0
        pts = np.asarray(bbox, dtype=float)
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        boxes.append(EasyBox(text=text.strip(), conf=conf, x1=x1, y1=y1, x2=x2, y2=y2))
    joined = " ".join(b.text for b in boxes if b.text).strip()
    return joined, boxes


def _tesseract_grid(bgr: np.ndarray, *, min_side: int, max_side: int) -> tuple[str, dict[str, str]]:
    import pytesseract

    up = _upscale_for_field(bgr, min_side=min_side, max_side=max_side)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    langs = ["ara", "ara+eng", "eng"]
    psms = (6, 7, 11, 13)
    by_key: dict[str, str] = {}
    candidates: list[str] = []
    for prep_i, prep in enumerate(ena._field_preprocess_variants(gray)):
        for lang in langs:
            for psm in psms:
                cfg = f"--oem 3 --psm {psm}"
                key = f"v{prep_i}_{lang}_psm{psm}"
                try:
                    t = pytesseract.image_to_string(prep, lang=lang, config=cfg)
                except Exception:
                    t = ""
                t = " ".join(t.split()).strip()
                by_key[key] = t
                if t:
                    candidates.append(t)
    best = ""
    if candidates:
        best = max(candidates, key=lambda s: ena._score_reading(s, True))
        if ena._arabic_char_count(best) == 0:
            best = max(candidates, key=len)
    return best, by_key


def _assess_visual(up_bgr: np.ndarray) -> str:
    if up_bgr.size == 0:
        return "empty crop"
    gray = cv2.cvtColor(up_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    mean = float(gray.mean())
    std = float(gray.std())
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    notes: list[str] = []
    if lap < 80:
        notes.append("soft/blur (low Laplacian variance)")
    elif lap < 200:
        notes.append("moderate sharpness")
    else:
        notes.append("fairly sharp")
    if std < 35:
        notes.append("low contrast")
    elif std > 70:
        notes.append("high local contrast (security pattern?)")
    notes.append(f"size={w}x{h} mean={mean:.0f} std={std:.0f} lap={lap:.0f}")
    return "; ".join(notes)


def _boundary_check(
    img: np.ndarray,
    box: list[float],
    pad: int,
    field: str,
) -> str:
    """Heuristic: ink near crop edges suggests tight YOLO box."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    cr = eid.crop_xyxy(img, box, pad)
    if cr.size == 0:
        return "empty crop"
    gray = cv2.cvtColor(cr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ch, cw = bw.shape[:2]
    margin = max(2, min(ch, cw) // 20)
    top = bw[:margin, :].mean()
    bot = bw[-margin:, :].mean()
    left = bw[:, :margin].mean()
    right = bw[:, -margin:].mean()
    edge_thresh = 8.0
    hits: list[str] = []
    if top > edge_thresh:
        hits.append("top")
    if bot > edge_thresh:
        hits.append("bottom")
    if left > edge_thresh:
        hits.append("left")
    if right > edge_thresh:
        hits.append("right")
    raw_w = x2 - x1
    raw_h = y2 - y1
    notes = [f"raw_box={raw_w}x{raw_h}px pad={pad}"]
    if hits:
        notes.append(f"ink_at_crop_edge({','.join(hits)}) — possible tight box")
    else:
        notes.append("no strong edge ink — box padding likely OK")
    # Check if box touches image border (detector clipped card)
    if x1 < 5:
        notes.append("box flush left image edge")
    if y1 < 5:
        notes.append("box flush top image edge")
    if x2 > w - 5:
        notes.append("box flush right image edge")
    if y2 > h - 5:
        notes.append("box flush bottom image edge")
    return "; ".join(notes)


def _batch_name_text(img, best: dict, reader, pad: int) -> tuple[str, bool]:
    """Legacy full batch (first+last in strip) vs current individual path."""
    if "firstName" not in best and "lastName" not in best:
        return "", False
    labeled = [
        (lab, eid.crop_xyxy(img, best[lab][0], pad))
        for lab in ("firstName", "lastName")
        if lab in best
    ]
    batched = ena.ocr_fields_batch_easyocr(labeled, reader, min_side=120)
    b_first = batched.get("firstName", "")
    b_last = batched.get("lastName", "")
    batch_full = f"{b_first} {b_last}".strip()

    parts: list[str] = []
    if "firstName" in best:
        cr = eid.crop_xyxy(img, best["firstName"][0], pad)
        parts.append(
            ena.ocr_text_field_easyocr(cr, reader, min_side=MIN_SIDE_NAME, max_side=MAX_SIDE_NAME)
        )
    if "lastName" in best:
        cr = eid.crop_xyxy(img, best["lastName"][0], pad)
        parts.append(
            ena.ocr_text_field_easyocr(cr, reader, min_side=MIN_SIDE_NAME, max_side=MAX_SIDE_NAME)
        )
    ind = " ".join(p for p in parts if p).strip()
    differs = normalize_arabic_text(batch_full) != normalize_arabic_text(ind)
    return batch_full, differs


def _batch_address_text(img, best: dict, reader, pad: int) -> tuple[str, bool]:
    if "address" not in best:
        return "", False
    labeled = [("address", eid.crop_xyxy(img, best["address"][0], pad))]
    batched = ena.ocr_fields_batch_easyocr(labeled, reader, min_side=120)
    batch_raw = batched.get("address", "")
    cr = eid.crop_xyxy(img, best["address"][0], pad)
    ind_raw = ena.ocr_text_field_easyocr(cr, reader, min_side=MIN_SIDE_ADDR, max_side=MAX_SIDE_ADDR)
    batch_clean = eid.clean_address_text(batch_raw)
    ind_clean = eid.clean_address_text(ind_raw)
    differs = normalize_arabic_text(batch_clean) != normalize_arabic_text(ind_clean)
    return batch_clean, differs


def _classify_verdict(
    fd: FieldDiag,
    *,
    field: str,
) -> tuple[str, str]:
    exp = fd.expected
    eo_cer = fd.easyocr_cer
    te_cer = fd.tess_cer
    avg_conf = (
        sum(b.conf for b in fd.easyocr_boxes) / len(fd.easyocr_boxes) if fd.easyocr_boxes else 0.0
    )
    low_conf = avg_conf < 0.45 and fd.easyocr_boxes
    high_conf_wrong = avg_conf >= 0.55 and eo_cer > 0.1

    if fd.batch_differs:
        return "a", "batch vs individual OCR still differs — possible code-path regression"

    if "ink_at_crop_edge" in fd.boundary_notes and eo_cer > 0.12:
        return "a", "ink at crop boundary — try increasing cfg.pad before blaming OCR engine"

    if te_cer + 0.05 < eo_cer and te_cer <= DEFAULT_CER_THRESHOLD:
        return "b", f"Tesseract CER {te_cer:.3f} beats EasyOCR {eo_cer:.3f} under threshold"

    if te_cer + 0.08 < eo_cer:
        return "b", f"Tesseract meaningfully lower CER ({te_cer:.3f} vs {eo_cer:.3f}) — merge candidate"

    if low_conf:
        return "c", f"low EasyOCR confidence (avg {avg_conf:.2f}) + poor CER — legibility/resolution limit"

    if high_conf_wrong:
        return "c", f"high-confidence misread (avg {avg_conf:.2f}) — char confusion on security pattern, not engine swap"

    if "soft/blur" in fd.visual_notes and eo_cer > 0.15:
        return "c", "soft/blur crop + high CER — resolution ceiling"

    if eo_cer > 0.15 and te_cer > 0.15:
        return "c", f"both engines fail (EasyOCR {eo_cer:.3f}, Tesseract {te_cer:.3f}) — genuine OCR ceiling"

    return "c", f"no structural bug found; marginal CER {eo_cer:.3f}"


def _diag_field(
    img,
    best: dict,
    reader,
    field: str,
    expected: str,
    pipeline_actual: str,
    *,
    min_side: int,
    max_side: int,
) -> FieldDiag:
    lab = {"name": None, "address": "address"}[field if field == "address" else "name"]
    if field == "name":
        # composite name strip for visual (stack first+last vertically)
        parts_img: list[np.ndarray] = []
        box_list: list[float] = []
        for ln in ("firstName", "lastName"):
            if ln in best:
                cr = eid.crop_xyxy(img, best[ln][0], PAD)
                parts_img.append(cr)
                if not box_list:
                    box_list = list(best[ln][0])
        if parts_img:
            max_w = max(p.shape[1] for p in parts_img)
            rows = []
            for p in parts_img:
                if p.shape[1] < max_w:
                    pad_r = max_w - p.shape[1]
                    p = np.pad(p, ((0, 0), (0, pad_r), (0, 0)), constant_values=255)
                rows.append(p)
            bgr = np.vstack(rows)
            box = box_list
        else:
            bgr = np.zeros((1, 1, 3), dtype=np.uint8)
            box = []
        batch_text, batch_differs = _batch_name_text(img, best, reader, PAD)
        eo_joined_parts: list[str] = []
        eo_boxes: list[EasyBox] = []
        y_off = 0
        for ln in ("firstName", "lastName"):
            if ln not in best:
                continue
            cr = eid.crop_xyxy(img, best[ln][0], PAD)
            j, boxes = _easyocr_detail(cr, reader, min_side=MIN_SIDE_NAME, max_side=MAX_SIDE_NAME)
            if j:
                eo_joined_parts.append(j)
            for b in boxes:
                eo_boxes.append(
                    EasyBox(
                        text=f"[{ln}] {b.text}",
                        conf=b.conf,
                        x1=b.x1,
                        y1=b.y1 + y_off,
                        x2=b.x2,
                        y2=b.y2 + y_off,
                    )
                )
            up = _upscale_for_field(cr, min_side=MIN_SIDE_NAME, max_side=MAX_SIDE_NAME)
            y_off += up.shape[0]
        eo_joined = " ".join(eo_joined_parts).strip()
        tess_parts: list[str] = []
        tess_grid: dict[str, str] = {}
        for ln in ("firstName", "lastName"):
            if ln not in best:
                continue
            cr = eid.crop_xyxy(img, best[ln][0], PAD)
            tbest, grid = _tesseract_grid(cr, min_side=MIN_SIDE_NAME, max_side=MAX_SIDE_NAME)
            if tbest:
                tess_parts.append(tbest)
            for k, v in grid.items():
                if v:
                    tess_grid[f"{ln}_{k}"] = v
        tess_best = " ".join(tess_parts).strip()
    else:
        lab = "address"
        bgr = eid.crop_xyxy(img, best[lab][0], PAD)
        box = list(best[lab][0])
        batch_text, batch_differs = _batch_address_text(img, best, reader, PAD)
        eo_joined, eo_boxes = _easyocr_detail(bgr, reader, min_side=MIN_SIDE_ADDR, max_side=MAX_SIDE_ADDR)
        tess_best, tess_grid = _tesseract_grid(bgr, min_side=MIN_SIDE_ADDR, max_side=MAX_SIDE_ADDR)

    up = _upscale_for_field(bgr, min_side=min_side, max_side=max_side)
    crop_dir = OUT_DIR / "crops" / Path(img_path_stem).stem
    crop_dir.mkdir(parents=True, exist_ok=True)
    suffix = field
    cv2.imwrite(str(crop_dir / f"{suffix}_raw.png"), bgr)
    cv2.imwrite(str(crop_dir / f"{suffix}_upscaled.png"), up)
    if field == "name":
        for ln in ("firstName", "lastName"):
            if ln in best:
                cr = eid.crop_xyxy(img, best[ln][0], PAD)
                up_ln = _upscale_for_field(cr, min_side=MIN_SIDE_NAME, max_side=MAX_SIDE_NAME)
                cv2.imwrite(str(crop_dir / f"{ln}_upscaled.png"), up_ln)

    fd = FieldDiag(
        field=field,
        expected=expected,
        pipeline_actual=pipeline_actual,
        easyocr_joined=eo_joined,
        easyocr_boxes=eo_boxes,
        tess_best=tess_best,
        tess_by_psm={k: v for k, v in list(tess_grid.items())[:12]},
        easyocr_cer=cer(expected, eo_joined),
        tess_cer=cer(expected, tess_best),
        batch_text=batch_text,
        batch_differs=batch_differs,
        crop_raw_shape=(bgr.shape[1], bgr.shape[0]) if bgr.size else (0, 0),
        crop_upscaled_shape=(up.shape[1], up.shape[0]) if up.size else (0, 0),
        box_xyxy=box,
        visual_notes=_assess_visual(up),
        boundary_notes=_boundary_check(img, box, PAD, field) if box else "",
    )
    v, r = _classify_verdict(fd, field=field)
    fd.verdict = v
    fd.verdict_reason = r
    return fd


img_path_stem = ""


def main() -> int:
    global img_path_stem
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "0"
    cfg = ExtractConfig(image=ROOT / "x.jpg", quiet=True, fast_mode=True, engine="easyocr")
    engine, tess_langs, reader = _init_ocr(cfg, device)
    field_yolo = eid.get_yolo(DEFAULT_FIELD_WEIGHTS)

    held_out = [
        c
        for c in discover_test_cases(ROOT / "test_data" / "id_cards")
        if str(c["ground_truth"].get("source") or "") in HELD_OUT_SOURCES
    ]

    failures: list[SampleDiag] = []
    for case in held_out:
        front = case["front"]
        gt = case["ground_truth"]
        exp_name = _expected_name(gt)
        exp_addr = (gt.get("address") or "").strip()

        row = extract_front(
            front,
            cfg,
            device=device,
            engine=engine,
            tess_langs=tess_langs,
            easyocr_reader=reader,
            dw=None,
        )
        act_name = row.get("full_name") or ""
        act_addr = row.get("address") or ""
        name_c = cer(exp_name, act_name) if exp_name else 0.0
        addr_c = cer(exp_addr, act_addr) if exp_addr else 0.0
        name_fail = bool(exp_name) and not (
            exact_match(exp_name, act_name, field="name") or name_c <= DEFAULT_CER_THRESHOLD
        )
        addr_fail = bool(exp_addr) and not (
            exact_match(exp_addr, act_addr, field="address") or addr_c <= DEFAULT_CER_THRESHOLD
        )
        if not name_fail and not addr_fail:
            continue

        img = cv2.imread(str(front))
        if img is None:
            continue
        img = eid.resize_for_speed(img, max_side=880)
        r = field_yolo.predict(source=img, conf=0.25, device=device, imgsz=480, verbose=False)[0]
        id_to_name = eid.load_class_names()
        best = eid.best_boxes_by_label(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int),
            r.boxes.conf.cpu().numpy(),
            id_to_name,
        )

        img_path_stem = front.name
        sd = SampleDiag(
            image=front.name,
            name_fail=name_fail,
            address_fail=addr_fail,
            code_path="fast_mode=True → firstName/lastName/address individual EasyOCR; dob/serial batched strip only",
        )
        if name_fail:
            sd.fields.append(
                _diag_field(img, best, reader, "name", exp_name, act_name, min_side=MIN_SIDE_NAME, max_side=MAX_SIDE_NAME)
            )
        if addr_fail:
            sd.fields.append(
                _diag_field(
                    img, best, reader, "address", exp_addr, act_addr, min_side=MIN_SIDE_ADDR, max_side=MAX_SIDE_ADDR
                )
            )
        failures.append(sd)

    # JSON dump
    def _json_safe(obj):
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    payload = []
    for sd in failures:
        payload.append(
            _json_safe(
                {
                    **{k: v for k, v in asdict(sd).items() if k != "fields"},
                    "fields": [asdict(f) for f in sd.fields],
                }
            )
        )
    json_path = OUT_DIR / "diagnosis.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown report
    lines = [
        "# Held-out failure diagnosis",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "Baseline: `report_20260707_143537` — name 69.2% (4/13 fail), address 84.6% (2/13 fail).",
        "",
        f"Failures analyzed: **{len(failures)}** images, **{sum(1 for s in failures for f in s.fields)}** fields.",
        "",
        "## Code-path check",
        "",
        "- `extract_id_all.py` fast_mode: `firstName`, `lastName`, `address` use **individual** `ocr_text_field_easyocr`.",
        "- Only `dob` + `serial` go through `ocr_fields_batch_easyocr` (name/address crops used as blank spacers only).",
        "- Re-ran legacy batch OCR on failing crops to confirm batch ≠ pipeline output.",
        "",
    ]

    counts = {"a": 0, "b": 0, "c": 0}
    for sd in failures:
        lines.append(f"## `{sd.image}`")
        lines.append("")
        lines.append(f"- Code path: {sd.code_path}")
        lines.append(f"- Name fail: {sd.name_fail} | Address fail: {sd.address_fail}")
        lines.append("")
        for fd in sd.fields:
            counts[fd.verdict] = counts.get(fd.verdict, 0) + 1
            crop_rel = f"crops/{Path(sd.image).stem}/{fd.field}_upscaled.png"
            lines.append(f"### {fd.field.upper()}")
            lines.append("")
            lines.append(f"- **Verdict ({fd.verdict})**: {fd.verdict_reason}")
            lines.append(f"- **Expected**: `{fd.expected}`")
            lines.append(f"- **Pipeline actual**: `{fd.pipeline_actual}`")
            lines.append(f"- **EasyOCR joined**: `{fd.easyocr_joined}` (CER={fd.easyocr_cer:.3f})")
            lines.append(f"- **Tesseract best**: `{fd.tess_best}` (CER={fd.tess_cer:.3f})")
            if fd.batch_text is not None:
                lines.append(f"- **Legacy batch OCR**: `{fd.batch_text}` | differs from individual: **{fd.batch_differs}**")
            lines.append(f"- **Crop (upscaled)**: `{crop_rel}` raw={fd.crop_raw_shape} upscaled={fd.crop_upscaled_shape}")
            lines.append(f"- **Visual**: {fd.visual_notes}")
            lines.append(f"- **Boundary**: {fd.boundary_notes}")
            lines.append("")
            lines.append("**EasyOCR boxes (detail=1):**")
            lines.append("")
            if fd.easyocr_boxes:
                lines.append("| text | conf | bbox |")
                lines.append("|------|------|------|")
                for b in fd.easyocr_boxes:
                    lines.append(
                        f"| `{b.text}` | {b.conf:.3f} | ({b.x1:.0f},{b.y1:.0f})-({b.x2:.0f},{b.y2:.0f}) |"
                    )
            else:
                lines.append("_no detections_")
            lines.append("")
            best_tess = sorted(
                ((k, v) for k, v in fd.tess_by_psm.items() if v),
                key=lambda kv: cer(fd.expected, kv[1]),
            )[:5]
            if best_tess:
                lines.append("**Top Tesseract variants (by CER):**")
                lines.append("")
                for k, v in best_tess:
                    lines.append(f"- `{k}`: `{v}` (CER={cer(fd.expected, v):.3f})")
                lines.append("")

    lines.extend(
        [
            "## Verdict summary",
            "",
            f"| Category | Count | Meaning |",
            f"|----------|-------|---------|",
            f"| **(a)** fixable bug | {counts.get('a', 0)} | padding / code path |",
            f"| **(b)** engine merge | {counts.get('b', 0)} | Tesseract beats EasyOCR on this crop |",
            f"| **(c)** ceiling | {counts.get('c', 0)} | resolution / legibility / both engines fail |",
            "",
            f"JSON: `{json_path.relative_to(ROOT)}`",
        ]
    )
    report_path = OUT_DIR / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"Counts: a={counts.get('a',0)} b={counts.get('b',0)} c={counts.get('c',0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
