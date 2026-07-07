"""
National ID structure validation and DOB cross-check.

Uses egypt_nid_decode.py (Eslam2014 format).
"""
from __future__ import annotations

from dataclasses import dataclass

from egypt_nid_decode import NidDecodeError, decode_egyptian_nid, western_digits_only

from tests.id_metrics import DobNidCrossCheck, normalize_dob


@dataclass
class NidValidation:
    ok: bool
    errors: list[str]
    decoded_birth_date: str = ""
    decoded_governorate: str = ""
    decoded_gender: str = ""


def validate_extracted_nid(national_id: str) -> NidValidation:
    """Validate structure of extracted 14-digit NID."""
    errors: list[str] = []
    digits = western_digits_only(national_id)

    if len(digits) != 14:
        errors.append(f"national_id length {len(digits)} != 14")
        return NidValidation(ok=False, errors=errors)

    century = int(digits[0])
    if century not in (2, 3):
        errors.append(f"invalid century digit: {century} (expected 2 or 3)")

    try:
        decoded = decode_egyptian_nid(digits)
    except NidDecodeError as e:
        errors.append(str(e))
        return NidValidation(ok=False, errors=errors)

    if not decoded.checksum_valid:
        errors.append(
            f"invalid check digit: expected {decoded.expected_check_digit}, got {decoded.check_digit}"
        )

    return NidValidation(
        ok=len(errors) == 0,
        errors=errors,
        decoded_birth_date=decoded.birth_date,
        decoded_governorate=decoded.governorate,
        decoded_gender=decoded.gender,
    )


def dob_matches_nid(extracted_dob: str, national_id: str) -> tuple[bool, str]:
    """Cross-check printed/derived DOB against NID embedded birth date."""
    val = validate_extracted_nid(national_id)
    if not val.ok or not val.decoded_birth_date:
        return False, "cannot decode NID for DOB cross-check"
    exp = normalize_dob(val.decoded_birth_date)
    got = normalize_dob(extracted_dob)
    if not got:
        return False, "extracted DOB empty"
    if exp != got:
        return False, f"DOB mismatch: printed_ocr={got} nid_decode={exp}"
    return True, ""


def build_dob_nid_cross_check(actual: dict[str, str]) -> DobNidCrossCheck:
    """Structured DOB vs NID cross-validation (flags bad NID read vs bad DOB OCR)."""
    nid = actual.get("national_id", "")
    printed = actual.get("dob") or actual.get("decoded_birth_date", "")
    ok, msg = dob_matches_nid(printed, nid)
    decoded = ""
    if len(western_digits_only(nid)) == 14:
        val = validate_extracted_nid(nid)
        decoded = val.decoded_birth_date
    return DobNidCrossCheck(
        ok=ok,
        printed_dob=printed,
        decoded_from_nid=decoded,
        national_id=nid,
        message=msg,
    )
