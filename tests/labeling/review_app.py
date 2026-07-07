"""
Part 2: FastAPI review UI for draft front-side labels.

  py -m tests.labeling.review_app
  Open http://127.0.0.1:8765/
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from tests.ground_truth import prefill_from_national_id
from tests.labeling.promote import (
    DEFAULT_VALID_RATIO,
    _find_image,
    list_pending_drafts,
    load_draft_for_review,
    save_reviewed,
)
from tests.labeling.yolo_boxes import FRONT_FIELD_NAMES
import export_id_to_excel as eid

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "test_data" / "id_cards"

app = FastAPI(title="ID front-side label review")


class SavePayload(BaseModel):
    ground_truth: dict
    boxes: list[dict]
    promote_to_dataset: bool = True
    valid_ratio: float = DEFAULT_VALID_RATIO


class DecodeNidPayload(BaseModel):
    national_id: str


@app.post("/api/decode-nid")
def api_decode_nid(payload: DecodeNidPayload):
    pref = prefill_from_national_id(payload.national_id)
    digits = pref.get("national_id", "")
    ok = bool(pref.get("decoded_birth_date"))
    checksum_valid = pref.get("checksum_valid") == "true"
    errors: list[str] = []
    if len(digits) != 14:
        errors.append(f"national_id length {len(digits)} != 14")
    elif not ok:
        errors.append("could not decode national_id structure")
    elif not checksum_valid:
        errors.append(
            f"invalid check digit: expected {pref.get('expected_check_digit', '?')}, "
            f"got {pref.get('decoded_check_digit', '?')}"
        )
    return {
        "ok": ok and len(errors) == 0,
        "structurally_valid": ok,
        "checksum_valid": checksum_valid,
        "errors": errors,
        "fields": pref,
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "review.html").read_text(encoding="utf-8")


@app.get("/api/classes")
def api_classes():
    names = eid.load_class_names()
    name_to_id = {v: k for k, v in names.items()}
    front = sorted(FRONT_FIELD_NAMES, key=lambda n: name_to_id.get(n, 99))
    return [{"id": name_to_id[n], "name": n} for n in front if n in name_to_id]


@app.get("/api/pending")
def api_pending(sort: str = "fast_confirm"):
    return list_pending_drafts(DATA_DIR, sort_mode=sort)


@app.get("/api/draft/{stem}")
def api_draft(stem: str):
    try:
        return load_draft_for_review(DATA_DIR, stem)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


@app.get("/api/image/{stem}")
def api_image(stem: str):
    p = _find_image(DATA_DIR, stem)
    if p:
        return FileResponse(p)
    raise HTTPException(404, "image not found")


@app.post("/api/save/{stem}")
def api_save(stem: str, payload: SavePayload):
    try:
        paths = save_reviewed(
            DATA_DIR,
            stem,
            payload.ground_truth,
            payload.boxes,
            valid_ratio=payload.valid_ratio,
            promote_to_dataset=payload.promote_to_dataset,
        )
        return {"ok": True, "paths": paths}
    except Exception as e:
        raise HTTPException(400, str(e)) from e


def main() -> None:
    import uvicorn

    print("Label review: http://127.0.0.1:8765/")
    print(f"Data dir: {DATA_DIR.resolve()}")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")


if __name__ == "__main__":
    main()
