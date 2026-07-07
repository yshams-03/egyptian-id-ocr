"""
Generates FAKE, WATERMARKED test cards. Output must never be usable as a real identity document.

Map SyntheticCardContent → tests/ground_truth.py JSON schema.
"""
from __future__ import annotations

from typing import Any

from tests.ground_truth import GROUND_TRUTH_KEYS, empty_ground_truth
from tests.labeling.sources import SOURCE_SYNTHETIC
from tests.synthetic.content import SyntheticCardContent


def content_to_ground_truth(
    content: SyntheticCardContent,
    *,
    extra_tags: list[str] | None = None,
    back_image: str = "",
) -> dict[str, Any]:
    gt = empty_ground_truth()
    gt.update(
        {
            "first_name": content.first_name,
            "last_name": content.last_name,
            "full_name": content.full_name,
            "address": content.address,
            "national_id": content.national_id,
            "dob": content.dob,
            "serial": content.serial,
            "job": content.job,
            "religion": content.religion,
            "gender": content.gender,
            "marital_status": content.marital_status,
            "expiry_date": content.expiry_date,
            "issue_date": content.issue_date,
            "back_nid": content.back_nid,
            "decoded_birth_date": content.decoded_birth_date,
            "decoded_governorate": content.decoded_governorate,
            "decoded_gender": content.decoded_gender,
            "decoded_century": content.decoded_century,
            "decoded_sequence": content.decoded_sequence,
            "decoded_check_digit": content.decoded_check_digit,
            "back_image": back_image,
            "source": SOURCE_SYNTHETIC,
            "notes": "AUTO-GENERATED SYNTHETIC FIXTURE — not a real identity document.",
        }
    )
    tags = ["synthetic"]
    if content.is_compound_name:
        tags.append("compound_name")
    if content.is_multiline_address:
        tags.append("multiline_address")
    if extra_tags:
        tags.extend(t for t in extra_tags if t not in tags)
    gt["tags"] = tags
    # Drop keys not in schema
    return {k: gt[k] for k in GROUND_TRUTH_KEYS if k in gt} | {"tags": gt["tags"], "notes": gt["notes"]}
