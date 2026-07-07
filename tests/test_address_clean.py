"""Unit tests for address post-OCR cleanup."""
from __future__ import annotations

import export_id_to_excel as eid


def test_clean_address_preserves_short_arabic_digit_tokens():
    assert eid.clean_address_text("ق ٩٤ اللوتس ١١") == "ق ٩٤ اللوتس ١١"


def test_clean_address_preserves_western_short_digit_tokens():
    assert eid.clean_address_text("3 ش عيد وهبه") == "3 ش عيد وهبه"


def test_clean_address_preserves_compound_building_number():
    assert eid.clean_address_text("٣٠ ش ابراهيم بديوى") == "٣٠ ش ابراهيم بديوى"


def test_clean_address_strip_digits_removes_numeric_runs():
    assert eid.clean_address_text("ق ٩٤ اللوتس ١١", strip_digits=True) == "ق  اللوتس"


def test_clean_address_collapses_whitespace_only():
    assert eid.clean_address_text("  التجمع   الاول  ") == "التجمع الاول"
