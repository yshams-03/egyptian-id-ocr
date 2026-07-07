#!/usr/bin/env python
"""Quick Tesseract vs EasyOCR on held-out name/address failures."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import export_id_to_excel as eid
import extract_name_address as ena
from extract_id_all import NAME_OCR_MAX_SIDE, NAME_OCR_MIN_SIDE, ExtractConfig, _init_ocr
from tests.ground_truth import discover_test_cases
from tests.id_metrics import cer
from tests.labeling.sources import HELD_OUT_SOURCES

FAILS = {
    "real_20_jpg.rf.55557c5cc16a33f20de60a82abf3af00.jpg": ["name", "address"],
    "real_20220817_140950_jpg.rf.43e87d53935d991e2759c835756b0e05.jpg": ["name"],
    "real_Omar-Khaled-ID-2_jpeg_jpg.rf.82c142350288a6f6cd03a14c37ca78a6.jpg": ["name"],
    "real_IMG20220809112613_jpg.rf.59b708e7aa082a84c38d180707e633ad.jpg": ["name"],
    "real_Front_jpg.rf.4ff273115771ae7e6199f7753ddacb6a.jpg": ["address"],
}


def main() -> None:
    eid.setup_tesseract()
    cfg = ExtractConfig(image=ROOT / "x.jpg", quiet=True, fast_mode=True, engine="easyocr")
    _, _, reader = _init_ocr(cfg, "0")
    field_yolo = eid.get_yolo(ROOT / "runs/train_id_detectr_hyper/weights/best.pt")

    for case in discover_test_cases(ROOT / "test_data" / "id_cards"):
        gt = case["ground_truth"]
        if str(gt.get("source") or "") not in HELD_OUT_SOURCES:
            continue
        img_name = case["front"].name
        if img_name not in FAILS:
            continue
        img = cv2.imread(str(case["front"]))
        img = eid.resize_for_speed(img, max_side=880)
        r = field_yolo.predict(source=img, conf=0.25, device="0", imgsz=480, verbose=False)[0]
        best = eid.best_boxes_by_label(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int),
            r.boxes.conf.cpu().numpy(),
            eid.load_class_names(),
        )
        print(f"=== {img_name}")
        for field in FAILS[img_name]:
            if field == "name":
                exp = (
                    (gt.get("full_name") or "").strip()
                    or (gt.get("name") or "").strip()
                    or f"{gt.get('first_name', '')} {gt.get('last_name', '')}".strip()
                )
                eo_parts, te_parts = [], []
                for ln in ("firstName", "lastName"):
                    if ln not in best:
                        continue
                    cr = eid.crop_xyxy(img, best[ln][0], 6)
                    eo_parts.append(
                        ena.ocr_text_field_easyocr(
                            cr, reader, min_side=NAME_OCR_MIN_SIDE, max_side=NAME_OCR_MAX_SIDE
                        )
                    )
                    te_parts.append(
                        ena.ocr_text_field_tesseract(
                            cr, min_side=NAME_OCR_MIN_SIDE, langs=["ara", "ara+eng"], expect_arabic=True
                        )
                    )
                eo = " ".join(p for p in eo_parts if p).strip()
                te = " ".join(p for p in te_parts if p).strip()
            else:
                exp = (gt.get("address") or "").strip()
                cr = eid.crop_xyxy(img, best["address"][0], 6)
                eo = eid.clean_address_text(
                    ena.ocr_text_field_easyocr(cr, reader, min_side=120, max_side=520)
                )
                te = eid.clean_address_text(
                    ena.ocr_text_field_tesseract(cr, min_side=120, langs=["ara", "ara+eng"], expect_arabic=True)
                )
            eo_c = cer(exp, eo)
            te_c = cer(exp, te)
            winner = "Tesseract" if te_c + 0.05 < eo_c else ("EasyOCR" if eo_c + 0.05 < te_c else "tie")
            print(f"  {field}: expected={exp!r}")
            print(f"    EasyOCR CER={eo_c:.3f}  {eo!r}")
            print(f"    Tess    CER={te_c:.3f}  {te!r}")
            print(f"    winner: {winner}")


if __name__ == "__main__":
    main()
