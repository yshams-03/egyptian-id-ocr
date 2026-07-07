"""
Generates FAKE, WATERMARKED test cards. Output must never be usable as a real identity document.

Build structurally valid 14-digit NIDs with the reserved synthetic governorate code.
"""
from __future__ import annotations

import random
from datetime import date

from egypt_nid_decode import NidDecodeError, compute_nid_check_digit, decode_egyptian_nid

from tests.synthetic.constants import SYNTHETIC_GOVERNORATE_CODE


def _check_digit_for(partial13: str) -> int:
    """Modulo-11 check digit used by real Egyptian NIDs."""
    return compute_nid_check_digit(partial13)


def build_synthetic_nid(
    birth: date,
    *,
    gender: str,
    rng: random.Random | None = None,
) -> str:
    """
    Return a 14-digit ID that passes decode_egyptian_nid with governorate code 99.
    DOB and gender are encoded in the number; always decode to cross-check.
    """
    rng = rng or random.Random()
    century_digit = 3 if birth.year >= 2000 else 2
    yy = birth.year % 100
    gender_digit = rng.choice((1, 3, 5, 7, 9)) if gender == "Male" else rng.choice((0, 2, 4, 6, 8))
    seq_prefix = rng.randint(0, 999)
    sequence = f"{seq_prefix:03d}{gender_digit}"
    partial = (
        f"{century_digit}"
        f"{yy:02d}{birth.month:02d}{birth.day:02d}"
        f"{SYNTHETIC_GOVERNORATE_CODE}"
        f"{sequence}"
    )
    if len(partial) != 13:
        raise ValueError(f"expected 13 digits before check, got {len(partial)}")
    nid = partial + str(_check_digit_for(partial))
    decoded = decode_egyptian_nid(nid)
    if decoded.governorate_code != SYNTHETIC_GOVERNORATE_CODE:
        raise NidDecodeError("synthetic NID governorate mismatch")
    if decoded.birth_date != birth.isoformat():
        raise NidDecodeError("synthetic NID birth date mismatch")
    if decoded.gender != gender:
        raise NidDecodeError("synthetic NID gender mismatch")
    return nid


def random_birth_date(rng: random.Random) -> date:
    year = rng.randint(1975, 2005)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return date(year, month, day)
