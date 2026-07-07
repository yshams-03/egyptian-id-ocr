"""
Ground-truth schema, template generation, and front/back image pairing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from egypt_nid_decode import NidDecodeError, decode_egyptian_nid, western_digits_only

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Full schema for test_data/id_cards/<stem>.json
GROUND_TRUTH_KEYS: tuple[str, ...] = (
    "first_name",
    "last_name",
    "full_name",
    "address",
    "national_id",
    "dob",
    "serial",
    "job",
    "religion",
    "gender",
    "marital_status",
    "expiry_date",
    "issue_date",
    "back_nid",
    "decoded_birth_date",
    "decoded_governorate",
    "decoded_gender",
    "decoded_century",
    "decoded_sequence",
    "decoded_check_digit",
    "back_image",
    "source",
    "tags",
    "notes",
)


def empty_ground_truth() -> dict[str, Any]:
    return {k: "" for k in GROUND_TRUTH_KEYS if k not in ("tags",)} | {"tags": [], "notes": ""}


def prefill_from_national_id(national_id: str) -> dict[str, str]:
    """Fill decoded_* fields from a manually entered 14-digit NID."""
    out: dict[str, str] = {}
    digits = western_digits_only(national_id)
    out["national_id"] = digits
    if len(digits) != 14:
        return out
    try:
        d = decode_egyptian_nid(digits, verify_checksum=False)
        dec = d.as_export_dict()
        out["dob"] = dec["Birth Date"]
        out["decoded_birth_date"] = dec["Birth Date"]
        out["decoded_governorate"] = dec["Governorate"]
        out["decoded_gender"] = dec["Gender"]
        out["decoded_century"] = dec["Century"]
        out["decoded_sequence"] = dec.get("Sequence", "")
        out["decoded_check_digit"] = dec.get("Check Digit", "")
        out["checksum_valid"] = "true" if d.checksum_valid else "false"
        out["expected_check_digit"] = str(d.expected_check_digit)
    except NidDecodeError:
        pass
    return out


def generate_template_json(
    image_path: Path,
    out_path: Path | None = None,
    *,
    national_id: str = "",
    existing: dict[str, Any] | None = None,
) -> Path:
    """
    Write a hand-fillable ground-truth JSON for one image.
    If national_id is provided, decoded fields are pre-filled from egypt_nid_decode.py.
    """
    image_path = image_path.expanduser().resolve()
    out_path = out_path or image_path.with_suffix(".json")
    payload = empty_ground_truth()
    if existing:
        payload.update({k: existing.get(k, payload.get(k, "")) for k in GROUND_TRUTH_KEYS})
    nid = national_id or str(existing.get("national_id", "") if existing else "")
    if nid:
        payload.update(prefill_from_national_id(nid))
    payload["notes"] = (
        payload.get("notes")
        or "Fill Arabic fields by hand. Enter national_id once — decoded_* auto-filled."
    )
    payload["tags"] = list(existing.get("tags", [])) if existing else []
    # Suggest back image filename if present
    back = image_path.parent / f"{image_path.stem}_back{image_path.suffix}"
    if back.is_file():
        payload["back_image"] = back.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def resolve_back_image(front_image: Path, ground_truth: dict[str, Any]) -> Path | None:
    """Optional back image: ground_truth['back_image'] or <stem>_back.<ext>."""
    front_image = front_image.expanduser().resolve()
    rel = (ground_truth.get("back_image") or "").strip()
    if rel:
        for base in (front_image.parent, front_image.parent.parent):
            p = (base / rel).resolve()
            if p.is_file():
                return p
    for ext in IMAGE_EXTS:
        p = front_image.parent / f"{front_image.stem}_back{ext}"
        if p.is_file():
            return p
    return None


def load_ground_truth(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_ground_truth_path(image_path: Path, data_dir: Path) -> Path | None:
    """`<stem>.json` beside image, or `ground_truth/<stem>.json` under data_dir."""
    stem = image_path.stem
    for candidate in (
        image_path.with_suffix(".json"),
        data_dir / f"{stem}.json",
        data_dir / "ground_truth" / f"{stem}.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def discover_test_cases(data_dir: Path) -> list[dict[str, Any]]:
    """
    Return test cases: {front, back?, ground_truth_path, ground_truth}.
    Skips *_back images as primaries.
    """
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        return []

    cases: list[dict[str, Any]] = []
    for img in sorted(data_dir.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        if img.stem.endswith("_back"):
            continue
        gt_path = resolve_ground_truth_path(img, data_dir)
        gt = load_ground_truth(gt_path) if gt_path else empty_ground_truth()
        back = resolve_back_image(img, gt)
        cases.append(
            {
                "front": img,
                "back": back,
                "ground_truth_path": gt_path,
                "ground_truth": gt,
            }
        )
    return cases
