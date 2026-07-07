"""
Decode 14-digit Egyptian national ID numbers.

Format (reference: https://github.com/Eslam2014/extract-information-from-eg-national-id):

    x - yymmdd - ss - iiig - z

    3 031224 02 0185 9
    |  |      |  |    check digit (Ministry validation)
    |  |      |  sequence on this birth day (digits 10–12; gender in digit 13)
    |  |      governorate code (07–08)
    |  birth date yy/mm/dd with century digit x
    century code: 2 → 1900s, 3 → 2000s, …

Gender: digit 13 (0-based index 12) — odd = Male, even = Female.

Check digit (digit 14): modulo-11 weighted checksum over the first 13 digits
(multipliers 2,7,6,5,4,3,2,7,6,5,4,3,2 — reference: egypt-natid / egyid).

CLI:
    py egypt_nid_decode.py 30312240201859
"""
from __future__ import annotations

import argparse
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime

# Governorate codes — aligned with Eslam2014/extract-information-from-eg-national-id
EGYPT_NID_GOVERNORATES: dict[str, str] = {
    "01": "Cairo",
    "02": "Alexandria",
    "03": "Port Said",
    "04": "Suez",
    "11": "Damietta",
    "12": "Dakahlia",
    "13": "Ash Sharqia",
    "14": "Kaliobeya",
    "15": "Kafr El - Sheikh",
    "16": "Gharbia",
    "17": "Monoufia",
    "18": "El Beheira",
    "19": "Ismailia",
    "21": "Giza",
    "22": "Beni Suef",
    "23": "Fayoum",
    "24": "El Menia",
    "25": "Assiut",
    "26": "Sohag",
    "27": "Qena",
    "28": "Aswan",
    "29": "Luxor",
    "31": "Red Sea",
    "32": "New Valley",
    "33": "Matrouh",
    "34": "North Sinai",
    "35": "South Sinai",
    "88": "Foreign",
    # Reserved for tests/synthetic/ fixtures only — not a real civil-registry code.
    "99": "Synthetic Test",
}

# Weighted modulo-11 checksum over the first 13 digits (public reverse-engineered rule).
NID_CHECK_MULTIPLIERS: tuple[int, ...] = (2, 7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2)


class NidDecodeError(ValueError):
    """Invalid or non-conforming national ID."""


@dataclass(frozen=True)
class DecodedNid:
    century_digit: int
    birth_century: int  # e.g. 20 → 2000–2099 style math (see decode)
    century_label: str  # e.g. "2000-2099"
    birth_date: str  # ISO YYYY-MM-DD
    governorate_code: str
    governorate: str
    sequence: str  # iiig (digits 10–12)
    gender_digit: int
    gender: str
    check_digit: int
    expected_check_digit: int
    checksum_valid: bool
    raw: str

    def as_export_dict(self) -> dict[str, str]:
        """Keys used by export_id_to_excel / GUI."""
        return {
            "Birth Date": self.birth_date,
            "Century": self.century_label,
            "Birth Century": str(self.birth_century),
            "Governorate": self.governorate,
            "Governorate Code": self.governorate_code,
            "Gender": self.gender,
            "Sequence": self.sequence,
            "Check Digit": str(self.check_digit),
        }


def western_digits_only(s: str) -> str:
    arabic = str.maketrans("٠١٢٣٤٥ٶ٧٨٩", "0123456789")
    return re.sub(r"[^\d]", "", str(s).translate(arabic))


def compute_nid_check_digit(partial13: str) -> int:
    """Return the expected 14th check digit for a 13-digit NID prefix."""
    digits = western_digits_only(partial13)
    if len(digits) != 13:
        raise NidDecodeError("Check digit input must be exactly 13 digits")
    total = sum(int(digits[i]) * NID_CHECK_MULTIPLIERS[i] for i in range(13))
    remainder = total % 11
    return abs(11 - remainder) % 10


def verify_nid_checksum(national_id: str) -> bool:
    """Return True when digit 14 matches the modulo-11 checksum."""
    digits = western_digits_only(national_id)
    if len(digits) != 14:
        return False
    return int(digits[13]) == compute_nid_check_digit(digits[:13])


def _birth_century_from_code(century_digit: int) -> tuple[int, str]:
    """Map first digit to full century number and label (Eslam: code + 18)."""
    birth_century = century_digit + 18
    if birth_century < 19:
        raise NidDecodeError(f"Invalid century digit: {century_digit}")
    if birth_century == 20:
        label = "1900-1999"
    elif birth_century == 21:
        label = "2000-2099"
    else:
        label = f"{(birth_century - 1) * 100}-{(birth_century * 100) - 1}"
    return birth_century, label


def _full_year(century_digit: int, yy: int) -> int:
    birth_century, _ = _birth_century_from_code(century_digit)
    return (birth_century * 100) - 100 + yy


def decode_egyptian_nid(national_id: str, *, verify_checksum: bool = True) -> DecodedNid:
    """
    Parse and validate a 14-digit Egyptian national ID.
    Raises NidDecodeError on invalid input.
    Set verify_checksum=False to decode structurally when the check digit may be wrong (e.g. OCR).
    """
    digits = western_digits_only(national_id)
    if len(digits) != 14:
        raise NidDecodeError("National ID must be exactly 14 digits")

    century_digit = int(digits[0])
    yy = int(digits[1:3])
    month = int(digits[3:5])
    day = int(digits[5:7])
    governorate_code = digits[7:9]
    sequence = digits[9:13]
    gender_digit = int(digits[12])
    check_digit = int(digits[13])

    birth_century, century_label = _birth_century_from_code(century_digit)
    if governorate_code not in EGYPT_NID_GOVERNORATES:
        raise NidDecodeError(f"Unknown governorate code: {governorate_code}")

    full_year = _full_year(century_digit, yy)
    try:
        birth_dt = datetime(full_year, month, day)
    except ValueError as e:
        raise NidDecodeError(f"Invalid birth date in ID: {e}") from e

    now = datetime.now()
    if birth_dt > now:
        raise NidDecodeError("Birth date is in the future")
    if birth_dt.year < 1900:
        raise NidDecodeError("Birth date before 1900")

    gender = "Male" if gender_digit % 2 == 1 else "Female"
    expected_check_digit = compute_nid_check_digit(digits[:13])
    checksum_valid = check_digit == expected_check_digit
    if verify_checksum and not checksum_valid:
        raise NidDecodeError(
            f"Invalid check digit: expected {expected_check_digit}, got {check_digit}"
        )

    return DecodedNid(
        century_digit=century_digit,
        birth_century=birth_century,
        century_label=century_label,
        birth_date=birth_dt.strftime("%Y-%m-%d"),
        governorate_code=governorate_code,
        governorate=EGYPT_NID_GOVERNORATES[governorate_code],
        sequence=sequence,
        gender_digit=gender_digit,
        gender=gender,
        check_digit=check_digit,
        expected_check_digit=expected_check_digit,
        checksum_valid=checksum_valid,
        raw=digits,
    )


def decode_egyptian_id(id_number: str) -> dict[str, str]:
    """Backward-compatible dict API for export_id_to_excel."""
    return decode_egyptian_nid(id_number).as_export_dict()


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Decode Egyptian 14-digit national ID (Eslam2014 format).

            Example:
              py egypt_nid_decode.py 30312240201859
            """
        ),
    )
    parser.add_argument("national_id", help="14-digit national ID")
    args = parser.parse_args()
    try:
        d = decode_egyptian_nid(args.national_id)
    except NidDecodeError as e:
        raise SystemExit(str(e)) from e
    print(f"Raw:              {d.raw}")
    print(f"Century digit:    {d.century_digit} ({d.century_label})")
    print(f"Birth date:       {d.birth_date}")
    print(f"Governorate:      {d.governorate} ({d.governorate_code})")
    print(f"Sequence (iiig):  {d.sequence}")
    print(f"Gender:           {d.gender} (digit {d.gender_digit})")
    print(f"Check digit (z):  {d.check_digit} (expected {d.expected_check_digit})")
    print(f"Checksum valid:   {d.checksum_valid}")


if __name__ == "__main__":
    main()
