"""
Generates FAKE, WATERMARKED test cards. Output must never be usable as a real identity document.

Fixed placeholder pools — intentionally not a real-name generator.
"""
from __future__ import annotations

# Not in Egypt's civil-registry table except our explicit decode entry "Synthetic Test".
SYNTHETIC_GOVERNORATE_CODE = "99"

SERIAL_PREFIX = "TEST-"

# Fictional placeholder given names (single-token or compound units).
PLACEHOLDER_FIRST_NAMES: tuple[str, ...] = (
    "فحص",
    "عينة",
    "تجريبي",
    "اختبار",
    "نموذج",
    "عبد الرحمن",
    "أبو بكر",
    "عبد الله",
)

PLACEHOLDER_FATHER_NAMES: tuple[str, ...] = (
    "بيانات",
    "صوري",
    "وهمي",
    "تجريب",
    "اختبار",
    "محمد",
    "أحمد",
)

PLACEHOLDER_GRANDFATHER_NAMES: tuple[str, ...] = (
    "نظام",
    "آلي",
    "برنامج",
    "فحص",
    "تجربة",
)

PLACEHOLDER_FAMILY_NAMES: tuple[str, ...] = (
    "البطاقة",
    "الصورة",
    "الاختبار",
    "التجريبي",
    "الوهمي",
    "النموذج",
)

PLACEHOLDER_STREETS: tuple[str, ...] = (
    "شارع الفحص",
    "شارع العينة",
    "شارع الاختبار",
    "شارع النموذج",
    "شارع التجريب",
)

PLACEHOLDER_DISTRICTS: tuple[str, ...] = (
    "حي الاختبار",
    "حي النموذج",
    "حي الفحص",
    "مركز تجريبي",
)

PLACEHOLDER_GOVERNORATES_AR: tuple[str, ...] = (
    "محافظة وهمية",
    "منطقة اختبار",
    "إقليم تجريبي",
)

PLACEHOLDER_JOBS: tuple[str, ...] = (
    "طالب",
    "موظف تجريبي",
    "فني اختبار",
    "مبرمج",
)

PLACEHOLDER_RELIGIONS: tuple[str, ...] = (
    "مسلم",
    "مسيحي",
)

# Distinctive watermark tint (RGBA) used for guardrail pixel checks.
WATERMARK_TINT_RGBA = (255, 69, 0, 110)  # orange-red, semi-transparent
WATERMARK_CORNER_TEXT_EN = "SPECIMEN"
WATERMARK_CORNER_TEXT_AR = "عينة اختبار"
WATERMARK_DIAGONAL_TEXT = "SPECIMEN / عينة اختبار"

CARD_WIDTH = 1000
CARD_HEIGHT = 630
