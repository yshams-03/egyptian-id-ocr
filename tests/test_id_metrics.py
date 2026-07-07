"""Unit tests for metrics, NID validation, ground-truth templates."""
from __future__ import annotations

import pytest

from tests.ground_truth import prefill_from_national_id
from tests.id_metrics import cer, exact_match, normalize_serial, score_fields
from tests.nid_validate import build_dob_nid_cross_check, dob_matches_nid, validate_extracted_nid
from egypt_nid_decode import compute_nid_check_digit, verify_nid_checksum


class TestNormalizeArabic:
    def test_alef_variants(self):
        from tests.id_metrics import normalize_arabic_text

        assert normalize_arabic_text("أحمد") == normalize_arabic_text("احمد")

    def test_cer_identical(self):
        assert cer("مصطفى", "مصطفى") == 0.0


class TestExactMatch:
    def test_nid_digits_only(self):
        assert exact_match("29611091301456", "٢٩٦١١٠٩١٣٠١٤٥٦", field="national_id")

    def test_serial_normalize(self):
        assert normalize_serial("GC9412479") == "GC9412479"
        assert exact_match("GC9412479", "GC9412479", field="serial")
        assert exact_match("ID1949712", "101949712", field="serial")
        assert exact_match("GG6848691", "666848691", field="serial")
        assert exact_match("HE3221885", "#E3221885", field="serial")
        assert exact_match("JP2261375", "/P2261375", field="serial")
        assert exact_match("GC9412479", "6(9412479", field="serial")

    def test_dob_formats(self):
        assert exact_match("1996-11-09", "1996/11/09", field="dob")


class TestNidValidator:
    def test_valid_nid(self):
        v = validate_extracted_nid("29611091301456")
        assert v.ok
        assert v.decoded_birth_date == "1996-11-09"

    def test_checksum(self):
        assert verify_nid_checksum("29611091301456")
        assert compute_nid_check_digit("2961109130145") == 6
        assert not verify_nid_checksum("29611091301450")

    def test_invalid_checksum(self):
        v = validate_extracted_nid("29611091301450")
        assert not v.ok
        assert any("check digit" in e for e in v.errors)

    def test_dob_cross_check(self):
        ok, _ = dob_matches_nid("1996-11-09", "29611091301456")
        assert ok

    def test_dob_cross_check_structured(self):
        cc = build_dob_nid_cross_check(
            {"national_id": "29611091301456", "dob": "1996-11-09"}
        )
        assert cc.ok


class TestGroundTruthPrefill:
    def test_prefill_from_nid(self):
        p = prefill_from_national_id("29611091301456")
        assert p["dob"] == "1996-11-09"
        assert p["decoded_governorate"] == "Ash Sharqia"
        assert p["checksum_valid"] == "true"

    def test_prefill_bad_checksum_still_decodes(self):
        p = prefill_from_national_id("29611091301450")
        assert p["dob"] == "1996-11-09"
        assert p["checksum_valid"] == "false"


class TestScoreFields:
    def test_skip_empty_expected(self):
        scores = score_fields({"national_id": "29611091301456"}, {"national_id": "29611091301456"})
        serial = next(s for s in scores if s.field == "serial")
        assert serial.skipped and serial.passed
