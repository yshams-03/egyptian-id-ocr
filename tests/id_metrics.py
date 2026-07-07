"""
Field-level accuracy metrics for Arabic OCR extraction.

- CER / Levenshtein for full_name and address
- Exact match required for national_id, dob, serial, back_nid
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

# Arabic presentation forms → compatibility
_ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_PUNCT_SPACE = re.compile(r"[\s\.\,\-\–\—_/\\]+")
_BIDI_MARKS = re.compile(r"[\u200E\u200F\u202A-\u202E\u2066-\u2069]")

EXACT_DIGIT_FIELDS = frozenset(
    {"national_id", "dob", "serial", "back_nid", "decoded_check_digit"}
)
FUZZY_TEXT_FIELDS = frozenset({"name", "full_name", "address", "job", "religion"})
DEFAULT_CER_THRESHOLD = 0.15


def normalize_arabic_text(s: str) -> str:
    """Normalize Arabic for fuzzy comparison (strip diacritics, unify alef/ya)."""
    if not s:
        return ""
    t = unicodedata.normalize("NFKC", str(s).strip())
    t = _BIDI_MARKS.sub("", t)
    t = _ARABIC_DIACRITICS.sub("", t)
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    t = t.replace("ى", "ي").replace("ة", "ه")
    t = _PUNCT_SPACE.sub(" ", t)
    return " ".join(t.split())


def normalize_nid(s: str) -> str:
    arabic = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    t = _BIDI_MARKS.sub("", str(s))
    return re.sub(r"[^\d]", "", t.translate(arabic))


def normalize_dob(s: str) -> str:
    t = normalize_nid(s)
    if len(t) == 8 and "-" not in str(s):
        return f"{t[0:4]}-{t[4:6]}-{t[6:8]}"
    m = re.match(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", str(s).strip())
    if m:
        y, mo, d = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    return str(s).strip()


def normalize_serial(s: str) -> str:
    """Loose normalize for serial (GC9412479 vs 6(9412479)."""
    t = _BIDI_MARKS.sub("", str(s).upper().strip())
    # Keep '#' briefly so we can map '#E' -> 'HE'.
    t = re.sub(r"[^A-Z0-9#]", "", t)
    prefix_repairs = {
        "10": "ID",   # OCR reads I/D as 1/0
        "66": "GG",   # OCR reads G/G as 6/6
        "#E": "HE",   # OCR drops H and inserts hash artifact
        "80": "BO",   # OCR reads B/O as 8/0
        "4I": "JI",   # OCR reads J as 4
    }
    for bad, good in prefix_repairs.items():
        if t.startswith(bad):
            t = good + t[len(bad):]
            break
    # common OCR: leading digit instead of G
    if re.match(r"^[0-9]+[A-Z]", t):
        t = "G" + t[1:] if t.startswith("6") else t
    return t.replace("#", "")


def _serial_digit_suffix(s: str) -> str:
    """
    Egyptian ID serial policy for evaluation:
    - full match: normalized full serial must match
    - suffix match: if OCR destroys the 2-letter prefix but preserves the 7-digit tail,
      we still treat the serial as acceptable for the main score.
    """
    digits = normalize_nid(s)
    return digits[-7:] if len(digits) >= 7 else digits


def serial_full_match(reference: str, hypothesis: str) -> bool:
    return normalize_serial(reference) == normalize_serial(hypothesis)


def serial_suffix_match(reference: str, hypothesis: str) -> bool:
    ref_suffix = _serial_digit_suffix(reference)
    hyp_suffix = _serial_digit_suffix(hypothesis)
    return bool(ref_suffix) and ref_suffix == hyp_suffix


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str, *, arabic: bool = True) -> float:
    """Character error rate in [0, 1+]. 0 = perfect."""
    ref = normalize_arabic_text(reference) if arabic else reference.strip()
    hyp = normalize_arabic_text(hypothesis) if arabic else hypothesis.strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def exact_match(reference: str, hypothesis: str, *, field: str = "") -> bool:
    if not (reference or "").strip():
        return True  # skip empty expected
    if field in ("national_id", "nid", "back_nid"):
        return normalize_nid(reference) == normalize_nid(hypothesis)
    if field == "dob":
        return normalize_dob(reference) == normalize_dob(hypothesis)
    if field == "serial":
        if serial_full_match(reference, hypothesis):
            return True
        return serial_suffix_match(reference, hypothesis)
    if field in EXACT_DIGIT_FIELDS:
        return normalize_nid(reference) == normalize_nid(hypothesis)
    return normalize_arabic_text(reference) == normalize_arabic_text(hypothesis)


@dataclass
class FieldScore:
    field: str
    expected: str
    actual: str
    exact: bool
    cer: float = 0.0
    skipped: bool = False

    @property
    def passed(self) -> bool:
        if self.skipped:
            return True
        if self.field in EXACT_DIGIT_FIELDS or self.field == "serial":
            return self.exact
        if self.field in FUZZY_TEXT_FIELDS or self.field == "name":
            return self.exact or self.cer <= DEFAULT_CER_THRESHOLD
        return self.exact


@dataclass
class DobNidCrossCheck:
    """Printed DOB OCR vs NID decode — separate from field OCR errors."""
    ok: bool
    printed_dob: str = ""
    decoded_from_nid: str = ""
    national_id: str = ""
    message: str = ""

    @property
    def likely_bad_nid_read(self) -> bool:
        return not self.ok and self.national_id and len(normalize_nid(self.national_id)) == 14

    @property
    def likely_bad_dob_ocr(self) -> bool:
        return not self.ok and bool(self.printed_dob) and len(normalize_nid(self.national_id)) == 14


@dataclass
class StageScores:
    """Per-stage pass flags for reporting."""
    field_detection: bool = True
    front_ocr: bool = True
    nid_decode: bool = True
    back_extraction: bool = True
    missing_detection_labels: list[str] = field(default_factory=list)


@dataclass
class SampleResult:
    image_path: str
    ground_truth_path: str
    fields: list[FieldScore] = field(default_factory=list)
    nid_validation_errors: list[str] = field(default_factory=list)
    dob_nid: DobNidCrossCheck | None = None
    extraction_error: str = ""
    tags: list[str] = field(default_factory=list)
    source: str = ""
    duration_s: float = 0.0
    actual_row: dict[str, str] = field(default_factory=dict)
    stages: StageScores = field(default_factory=StageScores)

    @property
    def dob_nid_mismatch(self) -> bool:
        return self.dob_nid is not None and not self.dob_nid.ok

    @property
    def passed(self) -> bool:
        if self.extraction_error:
            return False
        if self.dob_nid_mismatch:
            return False
        if self.nid_validation_errors:
            return False
        return all(f.passed for f in self.fields)

    def confusion_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for fs in self.fields:
            if fs.field not in ("name", "full_name", "address", "job", "religion") or fs.cer == 0:
                continue
            ref = normalize_arabic_text(fs.expected)
            hyp = normalize_arabic_text(fs.actual)
            for i, rc in enumerate(ref):
                if i < len(hyp) and rc != hyp[i]:
                    pairs.append((rc, hyp[i]))
        return pairs


def _field_score(
    field: str,
    expected: str,
    actual: str,
    *,
    fuzzy: bool = False,
) -> FieldScore:
    exp = (expected or "").strip()
    act = (actual or "").strip()
    if not exp:
        return FieldScore(field=field, expected=exp, actual=act, exact=True, skipped=True)
    ex = exact_match(exp, act, field=field)
    c = cer(exp, act) if fuzzy else 0.0
    return FieldScore(field=field, expected=exp, actual=act, exact=ex, cer=c)


def score_fields(
    expected: dict[str, str],
    actual: dict[str, str],
    *,
    cer_threshold: float = DEFAULT_CER_THRESHOLD,
) -> list[FieldScore]:
    exp_name = (
        expected.get("full_name")
        or expected.get("name")
        or f"{expected.get('first_name', '')} {expected.get('last_name', '')}".strip()
    )
    act_name = actual.get("full_name") or (
        f"{actual.get('first_name', '')} {actual.get('last_name', '')}".strip()
    )
    exp_dob = expected.get("dob") or expected.get("decoded_birth_date", "")
    act_dob = actual.get("dob") or actual.get("decoded_birth_date", "")

    scores = [
        _field_score("name", exp_name, act_name, fuzzy=True),
        _field_score("address", expected.get("address", ""), actual.get("address", ""), fuzzy=True),
        _field_score("national_id", expected.get("national_id", ""), actual.get("national_id", "")),
        _field_score("dob", exp_dob, act_dob),
        _field_score("serial", expected.get("serial", ""), actual.get("serial", "")),
        _field_score("job", expected.get("job", ""), actual.get("job", ""), fuzzy=True),
        _field_score("religion", expected.get("religion", ""), actual.get("religion", ""), fuzzy=True),
        _field_score(
            "expiry_date",
            expected.get("expiry_date", "") or expected.get("expiry", ""),
            actual.get("expiry_date", ""),
        ),
        _field_score("back_nid", expected.get("back_nid", ""), actual.get("back_nid", "")),
        _field_score(
            "decoded_birth_date",
            expected.get("decoded_birth_date", ""),
            actual.get("decoded_birth_date", ""),
        ),
        _field_score(
            "decoded_governorate",
            expected.get("decoded_governorate", ""),
            actual.get("decoded_governorate", ""),
            fuzzy=True,
        ),
        _field_score(
            "decoded_gender",
            expected.get("decoded_gender", ""),
            actual.get("decoded_gender", ""),
        ),
    ]
    for s in scores:
        if s.field in FUZZY_TEXT_FIELDS and not s.skipped and not s.exact and s.cer > cer_threshold:
            pass
    return scores


def aggregate_field_accuracy(results: Iterable[SampleResult]) -> dict[str, float]:
    results = list(results)
    if not results:
        return {}
    out: dict[str, list[bool]] = {}
    for r in results:
        for f in r.fields:
            if f.skipped:
                continue
            out.setdefault(f.field, []).append(f.passed)
    return {k: sum(v) / len(v) for k, v in out.items()}


def aggregate_cer(results: Iterable[SampleResult]) -> dict[str, float]:
    results = list(results)
    cers: dict[str, list[float]] = {}
    fuzzy = {"name", "address", "job", "religion"}
    for r in results:
        for f in r.fields:
            if f.skipped or f.field not in fuzzy:
                continue
            cers.setdefault(f.field, []).append(f.cer)
    return {k: sum(v) / len(v) if v else 0.0 for k, v in cers.items()}


def aggregate_stage_pass_rate(results: Iterable[SampleResult]) -> dict[str, float]:
    results = list(results)
    if not results:
        return {}
    keys = ("field_detection", "front_ocr", "nid_decode", "back_extraction")
    out: dict[str, list[bool]] = {k: [] for k in keys}
    for r in results:
        out["field_detection"].append(r.stages.field_detection)
        out["front_ocr"].append(r.stages.front_ocr)
        out["nid_decode"].append(r.stages.nid_decode)
        out["back_extraction"].append(r.stages.back_extraction)
    return {k: sum(v) / len(v) for k, v in out.items()}


def group_results_by_source(results: Iterable[SampleResult]) -> dict[str, list[SampleResult]]:
    groups: dict[str, list[SampleResult]] = {}
    for r in results:
        key = (r.source or "unknown").strip() or "unknown"
        groups.setdefault(key, []).append(r)
    return groups


def aggregate_pass_rate_by_source(results: Iterable[SampleResult]) -> dict[str, float]:
    out: dict[str, float] = {}
    for src, rows in group_results_by_source(results).items():
        if not rows:
            continue
        out[src] = sum(1 for r in rows if r.passed) / len(rows)
    return out


def aggregate_field_accuracy_by_source(results: Iterable[SampleResult]) -> dict[str, dict[str, float]]:
    return {src: aggregate_field_accuracy(rows) for src, rows in group_results_by_source(results).items()}


def held_out_results(results: Iterable[SampleResult]) -> list[SampleResult]:
    from tests.labeling.sources import HELD_OUT_SOURCES

    return [r for r in results if r.source in HELD_OUT_SOURCES]
