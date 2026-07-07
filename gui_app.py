"""
Local web GUI for testing extract_id_all.py.

  py -m pip install fastapi uvicorn python-multipart
  py gui_app.py

Open http://127.0.0.1:8000/ — serves index.html and POST /process.
"""
from __future__ import annotations

import asyncio
import base64
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from extract_id_all import ExtractConfig, extract_all

app = FastAPI(title="Egyptian ID OCR Test GUI")

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "runs" / "gui_temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

FIELD_WEIGHTS = BASE_DIR / "runs" / "train_id_detectr_hyper" / "weights" / "best.pt"
DIGIT_WEIGHTS = BASE_DIR / "runs" / "train_arabic_numbers_v2" / "weights" / "best.pt"

_executor = ThreadPoolExecutor(max_workers=1)
_easyocr_reader: object | None = None
_field_yolo: object | None = None
_digit_yolo: object | None = None
_models_ready = False


def _form_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "on")


def _warmup_models() -> None:
    global _easyocr_reader, _field_yolo, _digit_yolo, _models_ready
    import export_id_to_excel as eid

    use_gpu = False
    try:
        import torch

        use_gpu = torch.cuda.is_available()
    except Exception:
        pass
    print(f"[warmup] Device: {'CUDA' if use_gpu else 'CPU'}", flush=True)
    if FIELD_WEIGHTS.is_file():
        print("[warmup] Loading field YOLO (cached for all requests)…", flush=True)
        _field_yolo = eid.get_yolo(FIELD_WEIGHTS)
    else:
        print(f"[warmup] Warning: field weights missing: {FIELD_WEIGHTS}", flush=True)
    if DIGIT_WEIGHTS.is_file():
        print("[warmup] Loading digit YOLO…", flush=True)
        _digit_yolo = eid.get_yolo(DIGIT_WEIGHTS)
    print("[warmup] Loading EasyOCR (one-time, ~1–2 min)…", flush=True)
    try:
        import easyocr

        _easyocr_reader = easyocr.Reader(["ar", "en"], gpu=use_gpu, verbose=False)
        print("[warmup] EasyOCR ready.", flush=True)
    except Exception as ex:
        print(f"[warmup] EasyOCR skipped: {ex}", flush=True)
        _easyocr_reader = None
    _models_ready = True
    print("[warmup] Server ready — target /process: under 30s (front+back).", flush=True)


def _crop_thumbnail_b64(path: Path, max_width: int = 360) -> str | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _run_extract(cfg: ExtractConfig) -> dict[str, str]:
    t0 = time.perf_counter()
    print(f"[process] start {cfg.image.name} fast={cfg.fast_mode}", flush=True)
    row = extract_all(cfg)
    print(f"[process] done in {time.perf_counter() - t0:.1f}s", flush=True)
    return row


@app.on_event("startup")
async def startup_warmup() -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _warmup_models)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "models_ready": _models_ready,
        "easyocr_cached": _easyocr_reader is not None,
        "field_weights": FIELD_WEIGHTS.is_file(),
        "digit_weights": DIGIT_WEIGHTS.is_file(),
        "yolo_cached": _field_yolo is not None,
    }


@app.get("/")
async def get_index():
    index_file = BASE_DIR / "index.html"
    if not index_file.is_file():
        return JSONResponse({"error": "index.html not found"}, status_code=404)
    return FileResponse(str(index_file))


@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    back_file: UploadFile | None = File(None),
    conf: float = Form(0.25),
    engine: str = Form("mixed"),
    lang_mode: str = Form("both"),
    decode_nid: str = Form("true"),
    auto_card_crop: str = Form("false"),
    strip_address_digits: str = Form("false"),
    auto_detect_side: str = Form("false"),
    back_only: str = Form("false"),
    fast_mode: str = Form("true"),
):
    if not _models_ready:
        return JSONResponse(
            {
                "success": False,
                "error": "Models still loading. Wait for “Server ready” in the terminal, then retry.",
            },
            status_code=503,
        )

    req_id = str(uuid.uuid4())
    req_dir = TEMP_DIR / req_id
    req_dir.mkdir(parents=True, exist_ok=True)

    file_ext = Path(file.filename or "upload.jpg").suffix or ".jpg"
    input_path = req_dir / f"uploaded{file_ext}"

    try:
        content = await file.read()
        input_path.write_bytes(content)
    except OSError as e:
        shutil.rmtree(req_dir, ignore_errors=True)
        return JSONResponse({"success": False, "error": f"Failed to save image: {e}"})

    back_path: Path | None = None
    if back_file is not None and back_file.filename:
        back_path = req_dir / f"back{Path(back_file.filename).suffix or '.jpg'}"
        back_path.write_bytes(await back_file.read())

    crop_dir = req_dir / "crops"
    use_fast = _form_bool(fast_mode, True)
    cfg = ExtractConfig(
        image=input_path,
        back_image=back_path,
        save_crops=None if use_fast else crop_dir,
        conf=conf,
        engine="easyocr" if use_fast else engine,
        lang_mode=lang_mode,
        decode_nid=_form_bool(decode_nid, True),
        auto_card_crop=_form_bool(auto_card_crop, False),
        strip_address_digits=_form_bool(strip_address_digits, False),
        auto_detect_side=_form_bool(auto_detect_side, False),
        force_back=_form_bool(back_only, False),
        fast_mode=use_fast,
        device="0",
        raise_on_empty=False,
        quiet=True,
        easyocr_reader=_easyocr_reader,
        field_yolo=_field_yolo,
        digit_yolo=_digit_yolo,
    )

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(_executor, partial(_run_extract, cfg))
    except SystemExit as se:
        shutil.rmtree(req_dir, ignore_errors=True)
        msg = str(se) if str(se) else "Extraction failed."
        return JSONResponse({"success": False, "error": msg})
    except Exception as e:
        shutil.rmtree(req_dir, ignore_errors=True)
        return JSONResponse({"success": False, "error": f"Pipeline error: {e}"})

    warnings: list[str] = []
    if result.get("nid_decode_error") == "no detections":
        warnings.append(
            "No ID fields detected. Use a clear front-of-card photo (not orientation-only train tiles)."
        )
    elif result.get("nid_decode_error"):
        warnings.append(str(result["nid_decode_error"]))
    if result.get("nid_mismatch_warning") == "true":
        warnings.append("Front national_id and back_nid differ (stored both; see Excel).")

    crops: dict[str, str] = {}
    if crop_dir.is_dir():
        for f in sorted(crop_dir.iterdir()):
            if f.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                continue
            thumb = _crop_thumbnail_b64(f)
            if thumb:
                crops[f.stem] = thumb

    shutil.rmtree(req_dir, ignore_errors=True)

    return JSONResponse(
        {
            "success": True,
            "data": result,
            "crops": crops,
            "warnings": warnings,
        }
    )


if __name__ == "__main__":
    import uvicorn

    print("Egyptian ID OCR GUI: http://127.0.0.1:8000/")
    print("Wait for [warmup] Server ready… before processing (first start loads EasyOCR).")
    print("Press Ctrl+C to stop.")
    uvicorn.run("gui_app:app", host="127.0.0.1", port=8000, reload=False)
