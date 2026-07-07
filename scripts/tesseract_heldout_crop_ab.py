#!/usr/bin/env python
"""Tesseract vs EasyOCR on saved held-out failure crops (apples-to-apples)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Local tessdata (ara from tessdata_fast; eng copied to match install)
os.environ["TESSDATA_PREFIX"] = str(ROOT / "tessdata")

import pytesseract

import export_id_to_excel as eid
from tests.id_metrics import cer

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CROP_ROOT = ROOT / "runs" / "diagnose_heldout_failures" / "crops"
LANGS = ("ara", "ara+eng")
PSMS = (6, 7, 11)

# EasyOCR CER from report_20260707_143537 / diagnose_heldout_failures
EASYOCR_CER = {
    "real_20_jpg.rf.55557c5cc16a33f20de60a82abf3af00|name": 0.174,
    "real_20_jpg.rf.55557c5cc16a33f20de60a82abf3af00|address": 0.191,
    "real_20220817_140950_jpg.rf.43e87d53935d991e2759c835756b0e05|name": 0.320,
    "real_Omar-Khaled-ID-2_jpeg_jpg.rf.82c142350288a6f6cd03a14c37ca78a6|name": 0.304,
    "real_IMG20220809112613_jpg.rf.59b708e7aa082a84c38d180707e633ad|name": 0.167,
    "real_Front_jpg.rf.4ff273115771ae7e6199f7753ddacb6a|address": 0.264,
}

SAMPLES = [
    {
        "dir": "real_20_jpg.rf.55557c5cc16a33f20de60a82abf3af00",
        "field": "name",
        "expected": "محمد عبدالعزيز احمد عون",
        "crops": ["firstName_upscaled.png", "lastName_upscaled.png"],
    },
    {
        "dir": "real_20_jpg.rf.55557c5cc16a33f20de60a82abf3af00",
        "field": "address",
        "expected": "شارع الحاجة ست - عزبة فرج واصل\nكفر الزيات - الغربية",
        "crops": ["address_upscaled.png"],
    },
    {
        "dir": "real_20220817_140950_jpg.rf.43e87d53935d991e2759c835756b0e05",
        "field": "name",
        "expected": "محمد احمد محمد عبدالمقصود",
        "crops": ["firstName_upscaled.png", "lastName_upscaled.png"],
    },
    {
        "dir": "real_Omar-Khaled-ID-2_jpeg_jpg.rf.82c142350288a6f6cd03a14c37ca78a6",
        "field": "name",
        "expected": "عمر خالد حمدى عبدالمجيد",
        "crops": ["firstName_upscaled.png", "lastName_upscaled.png"],
    },
    {
        "dir": "real_IMG20220809112613_jpg.rf.59b708e7aa082a84c38d180707e633ad",
        "field": "name",
        "expected": "ايمن احمد محمد سيد",
        "crops": ["firstName_upscaled.png", "lastName_upscaled.png"],
    },
    {
        "dir": "real_Front_jpg.rf.4ff273115771ae7e6199f7753ddacb6a",
        "field": "address",
        "expected": "٣ ش عيد وهبه - عزبة احمد سليم\nشبرا الخيمه ثان - القليوبية",
        "crops": ["address_upscaled.png"],
    },
]


def _tess_read(gray, lang: str, psm: int) -> str:
    cfg = f"--oem 3 --psm {psm}"
    try:
        t = pytesseract.image_to_string(gray, lang=lang, config=cfg)
    except Exception:
        return ""
    return " ".join(t.split()).strip()


def _ocr_crop_paths(crop_paths: list[Path], lang: str, psm: int) -> str:
    parts: list[str] = []
    for p in crop_paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        t = _tess_read(gray, lang, psm)
        if t:
            parts.append(t)
    return " ".join(parts).strip()


def main() -> None:
    out_lines = [
        "# Tesseract vs EasyOCR — held-out failure crops",
        "",
        f"TESSDATA_PREFIX: `{os.environ['TESSDATA_PREFIX']}`",
        "ara.traineddata: tessdata_fast (eng install matched tessdata_fast ~3.9MB)",
        f"PSM tried: {PSMS} | langs: {LANGS}",
        "",
        "| Sample | Field | EasyOCR CER | Best Tess CER | Best config | Tess beats? |",
        "|--------|-------|-------------|---------------|-------------|-------------|",
    ]

    any_beat = 0
    results = []

    for s in SAMPLES:
        crop_paths = [CROP_ROOT / s["dir"] / c for c in s["crops"]]
        exp = s["expected"]
        key = f"{s['dir']}|{s['field']}"
        eo_cer = EASYOCR_CER[key]

        best_cer = 1.0
        best_text = ""
        best_cfg = ""
        grid: list[tuple[str, str, float]] = []

        for lang in LANGS:
            for psm in PSMS:
                hyp = _ocr_crop_paths(crop_paths, lang, psm)
                c = cer(exp, hyp)
                grid.append((f"{lang}_psm{psm}", hyp, c))
                if c < best_cer:
                    best_cer = c
                    best_text = hyp
                    best_cfg = f"{lang} psm={psm}"

        beats = best_cer + 0.001 < eo_cer  # strict: tess must be lower
        if beats:
            any_beat += 1
        delta = eo_cer - best_cer
        out_lines.append(
            f"| `{s['dir'][:24]}…` | {s['field']} | {eo_cer:.3f} | {best_cer:.3f} | {best_cfg} | "
            f"{'**yes** Δ=' + f'{delta:.3f}' if beats else 'no'} |"
        )
        results.append({**s, "easyocr_cer": eo_cer, "best_tess_cer": best_cer, "best_cfg": best_cfg, "best_text": best_text, "grid": grid, "beats": beats})

    out_lines.extend(["", "## Per-sample detail", ""])
    for r in results:
        out_lines.append(f"### {r['dir']} — {r['field']}")
        out_lines.append(f"- Expected: `{r['expected']}`")
        out_lines.append(f"- EasyOCR CER: **{r['easyocr_cer']:.3f}**")
        out_lines.append(f"- Best Tesseract ({r['best_cfg']}): CER **{r['best_tess_cer']:.3f}** → `{r['best_text']}`")
        out_lines.append("")
        out_lines.append("| config | CER | output |")
        out_lines.append("|--------|-----|--------|")
        for cfg, text, c in sorted(r["grid"], key=lambda x: x[2]):
            out_lines.append(f"| {cfg} | {c:.3f} | `{text}` |")
        out_lines.append("")

    out_lines.extend(
        [
            "## Conclusion",
            "",
            f"- Tesseract beats EasyOCR on **{any_beat}/6** samples (lower CER on same upscaled crops).",
            "",
        ]
    )

    report = ROOT / "runs" / "diagnose_heldout_failures" / "tesseract_ab_report.md"
    report.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {report}")
    for r in results:
        mark = "BEATS" if r["beats"] else "no"
        print(
            f"{r['dir'][:30]:30} {r['field']:7} EO={r['easyocr_cer']:.3f} "
            f"Tess={r['best_tess_cer']:.3f} ({r['best_cfg']}) {mark}"
        )


if __name__ == "__main__":
    main()
