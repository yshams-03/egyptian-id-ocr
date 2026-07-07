"""
Generates FAKE, WATERMARKED test cards. Output must never be usable as a real identity document.

Randomized fictional Arabic content from fixed placeholder pools only.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass
from datetime import date, timedelta

from tests.synthetic.constants import (
    PLACEHOLDER_DISTRICTS,
    PLACEHOLDER_FAMILY_NAMES,
    PLACEHOLDER_FATHER_NAMES,
    PLACEHOLDER_FIRST_NAMES,
    PLACEHOLDER_GOVERNORATES_AR,
    PLACEHOLDER_GRANDFATHER_NAMES,
    PLACEHOLDER_JOBS,
    PLACEHOLDER_RELIGIONS,
    PLACEHOLDER_STREETS,
    SERIAL_PREFIX,
)
from tests.synthetic.nid import build_synthetic_nid, random_birth_date


@dataclass(frozen=True)
class SyntheticCardContent:
    first_name: str
    last_name: str
    full_name: str
    address: str
    national_id: str
    dob: str
    serial: str
    job: str
    religion: str
    gender: str
    marital_status: str
    expiry_date: str
    issue_date: str
    back_nid: str
    decoded_birth_date: str
    decoded_governorate: str
    decoded_gender: str
    decoded_century: str
    decoded_sequence: str
    decoded_check_digit: str
    is_compound_name: bool
    is_multiline_address: bool


def _pick_name(rng: random.Random) -> tuple[str, str, str, bool]:
    first = rng.choice(PLACEHOLDER_FIRST_NAMES)
    father = rng.choice(PLACEHOLDER_FATHER_NAMES)
    grandfather = rng.choice(PLACEHOLDER_GRANDFATHER_NAMES)
    family = rng.choice(PLACEHOLDER_FAMILY_NAMES)
    compound = " " in first.strip() or first in ("عبد الرحمن", "أبو بكر", "عبد الله")
    last = f"{father} {grandfather} {family}".strip()
    full = f"{first} {last}".strip()
    return first, last, full, compound


def _pick_address(rng: random.Random, multiline: bool) -> str:
    street = rng.choice(PLACEHOLDER_STREETS)
    district = rng.choice(PLACEHOLDER_DISTRICTS)
    gov = rng.choice(PLACEHOLDER_GOVERNORATES_AR)
    if multiline:
        return f"{street}\n{district}\n{gov}"
    return f"{street} - {district} - {gov}"


def _fake_serial(rng: random.Random) -> str:
    suffix = "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    return f"{SERIAL_PREFIX}{suffix}"


def generate_content(
    rng: random.Random | None = None,
    *,
    multiline_address: bool | None = None,
    gender: str | None = None,
    birth: date | None = None,
) -> SyntheticCardContent:
    rng = rng or random.Random()
    multiline = rng.choice((True, False)) if multiline_address is None else multiline_address
    gender = gender or rng.choice(("Male", "Female"))
    birth = birth or random_birth_date(rng)

    first, last, full, compound = _pick_name(rng)
    address = _pick_address(rng, multiline)
    nid = build_synthetic_nid(birth, gender=gender, rng=rng)

    from egypt_nid_decode import decode_egyptian_nid

    dec = decode_egyptian_nid(nid)
    export = dec.as_export_dict()

    issue = birth + timedelta(days=rng.randint(7000, 9000))
    expiry = issue + timedelta(days=rng.randint(3000, 4000))

    return SyntheticCardContent(
        first_name=first,
        last_name=last,
        full_name=full,
        address=address,
        national_id=nid,
        dob=export["Birth Date"],
        serial=_fake_serial(rng),
        job=rng.choice(PLACEHOLDER_JOBS),
        religion=rng.choice(PLACEHOLDER_RELIGIONS),
        gender="ذكر" if gender == "Male" else "أنثى",
        marital_status=rng.choice(("أعزب", "متزوج", "أرمل")),
        expiry_date=expiry.strftime("%Y-%m-%d"),
        issue_date=issue.strftime("%Y-%m-%d"),
        back_nid=nid,
        decoded_birth_date=export["Birth Date"],
        decoded_governorate=export["Governorate"],
        decoded_gender=export["Gender"],
        decoded_century=export["Century"],
        decoded_sequence=export.get("Sequence", ""),
        decoded_check_digit=export.get("Check Digit", ""),
        is_compound_name=compound,
        is_multiline_address=multiline,
    )


def is_placeholder_first_name(name: str) -> bool:
    """Guardrail: first token must come from the fixed pool."""
    token = name.strip().split()[0] if name.strip() else ""
    if token in PLACEHOLDER_FIRST_NAMES:
        return True
    for p in PLACEHOLDER_FIRST_NAMES:
        if name.strip().startswith(p):
            return True
    return False
