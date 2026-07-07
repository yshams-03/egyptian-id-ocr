"""
Local heuristic scorer to pick EasyOCR vs Tesseract(ara) for name/address fields.

No ground truth at inference — uses lexicon derived from train-source GT only.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

import export_id_to_excel as eid
import extract_name_address as ena
from tests.id_metrics import levenshtein, normalize_arabic_text

BASE = Path(__file__).resolve().parent
LEXICON_PATH = BASE / "scripts" / "lexicon" / "egyptian_lexicon.json"
TESSDATA_PREFIX = BASE / "tessdata"

_ARABIC = re.compile(r"[\u0600-\u06FF]")
_LATIN = re.compile(r"[A-Za-z]")
_DIGIT = re.compile(r"[\d٠-٩]")


@dataclass
class EngineSelectStats:
    selections: list[dict] = field(default_factory=list)

    def reset(self) -> None:
        self.selections.clear()

    def record(
        self,
        *,
        image: str,
        field_label: str,
        easyocr_text: str,
        tess_text: str,
        chosen: str,
        easy_score: float,
        tess_score: float,
        expected: str = "",
    ) -> None:
        self.selections.append(
            {
                "image": image,
                "field": field_label,
                "easyocr": easyocr_text,
                "tesseract": tess_text,
                "chosen": chosen,
                "easy_score": round(easy_score, 2),
                "tess_score": round(tess_score, 2),
                "expected": expected,
            }
        )


# Selection audit log (reset per suite run)
_stats = EngineSelectStats()


def get_engine_select_stats() -> EngineSelectStats:
    return _stats


def reset_engine_select_stats() -> None:
    _stats.reset()


@lru_cache(maxsize=1)
def load_lexicon() -> dict:
    if not LEXICON_PATH.is_file():
        raise FileNotFoundError(
            f"Lexicon missing at {LEXICON_PATH}. Run: py scripts/lexicon/build_lexicon.py"
        )
    return json.loads(LEXICON_PATH.read_text(encoding="utf-8"))


def _ensure_tesseract() -> None:
    if TESSDATA_PREFIX.is_dir():
        os.environ.setdefault("TESSDATA_PREFIX", str(TESSDATA_PREFIX))
    try:
        eid.setup_tesseract()
    except SystemExit:
        pass


def _tokens(text: str) -> list[str]:
    t = normalize_arabic_text(text)
    if not t:
        return []
    return [p for p in t.split() if p]


def _fuzzy_in_lexicon(token: str, vocab: set[str], max_dist: int) -> bool:
    if not token:
        return False
    if token in vocab:
        return True
    lim = 1 if len(token) <= 4 else max_dist
    for v in vocab:
        if abs(len(v) - len(token)) > lim:
            continue
        if levenshtein(token, v) <= lim:
            return True
    return False


def _arabic_ratio(text: str) -> float:
    if not text:
        return 0.0
    ar = sum(1 for c in text if _ARABIC.match(c))
    return ar / max(len(text.replace(" ", "")), 1)


def _garbage_penalty(text: str) -> float:
    """Penalize digit/Latin soup and very short junk tokens like 4م٧."""
    pen = 0.0
    for tok in _tokens(text):
        if _DIGIT.search(tok) and _ARABIC.search(tok) and len(tok) <= 4:
            pen += 80.0
        if _LATIN.search(tok):
            pen += 40.0
        if len(tok) == 1 and not _ARABIC.match(tok):
            pen += 30.0
        # low Arabic ratio per token
        ar = sum(1 for c in tok if _ARABIC.match(c))
        if len(tok) >= 2 and ar / len(tok) < 0.5:
            pen += 25.0
    return pen


def score_name_text(text: str, lexicon: dict | None = None) -> float:
    if not (text or "").strip():
        return -1e9
    lex = lexicon or load_lexicon()
    vocab = set(lex.get("name_tokens", [])) | set(lex.get("name_compounds", []))
    toks = _tokens(text)
    if not toks:
        return -1e9

    score = 0.0
    score += _arabic_ratio(text) * 120.0
    score -= _garbage_penalty(text)

    # token count: Egyptian names usually 2-5 parts when full name; single-field crops 1-3
    n = len(toks)
    if n == 1:
        score += 15.0
    elif 2 <= n <= 4:
        score += 35.0
    elif n > 6:
        score -= 20.0 * (n - 6)

    dict_hits = sum(1 for t in toks if _fuzzy_in_lexicon(t, vocab, max_dist=2))
    score += (dict_hits / max(n, 1)) * 180.0

    # reward common compound patterns
    for t in toks:
        if t.startswith("عبد") and len(t) >= 5:
            score += 15.0
        if t.startswith("ابو") or t.startswith("ابن"):
            score += 10.0

    score += min(len(normalize_arabic_text(text)), 40) * 0.5
    return score


def score_address_text(text: str, lexicon: dict | None = None) -> float:
    if not (text or "").strip():
        return -1e9
    lex = lexicon or load_lexicon()
    vocab = (
        set(lex.get("address_tokens", []))
        | set(lex.get("governorates", []))
        | set(lex.get("connectors", []))
    )
    toks = _tokens(text)
    if not toks:
        return -1e9

    score = 0.0
    score += _arabic_ratio(text) * 100.0
    score -= _garbage_penalty(text) * 0.5  # digits OK in addresses

    n = len(toks)
    if 3 <= n <= 12:
        score += 30.0
    elif n < 2:
        score -= 20.0
    elif n > 15:
        score -= 15.0

    dict_hits = sum(1 for t in toks if _fuzzy_in_lexicon(t, vocab, max_dist=2))
    score += (dict_hits / max(n, 1)) * 150.0

    conn = set(lex.get("connectors", []))
    if any(t in conn for t in toks):
        score += 25.0

    score += min(len(normalize_arabic_text(text)), 80) * 0.3
    return score


def _tesseract_read(bgr: np.ndarray, *, min_side: int, psm: int) -> str:
    import pytesseract

    _ensure_tesseract()
    big = eid.upscale_crop(bgr, min_side=max(min_side, 120))
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    cfg = f"--oem 3 --psm {psm}"
    try:
        t = pytesseract.image_to_string(gray, lang="ara", config=cfg)
    except Exception:
        return ""
    return " ".join(t.split()).strip()


def _best_tesseract(bgr: np.ndarray, *, min_side: int, field_kind: str) -> tuple[str, float]:
    scorer = score_name_text if field_kind == "name" else score_address_text
    best_t, best_s = "", -1e9
    for psm in (6, 11):
        t = _tesseract_read(bgr, min_side=min_side, psm=psm)
        if not t:
            continue
        s = scorer(t)
        if s > best_s:
            best_s = s
            best_t = t
    return best_t, best_s


def select_field_text(
    bgr: np.ndarray,
    *,
    reader,
    field_kind: str,
    field_label: str,
    min_side: int,
    max_side: int = 520,
    image_path: str = "",
) -> str:
    """
    Run EasyOCR + Tesseract(ara psm 6/11), score both, return higher-scoring text.
    Empty Tesseract never wins.
    """
    easy = ena.ocr_text_field_easyocr(bgr, reader, min_side=min_side, max_side=max_side)
    tess, tess_score = _best_tesseract(bgr, min_side=min_side, field_kind=field_kind)
    scorer = score_name_text if field_kind == "name" else score_address_text
    easy_score = scorer(easy)

    if tess and tess_score > easy_score:
        chosen, out = "tesseract", tess
    else:
        chosen, out = "easyocr", easy

    _stats.record(
        image=image_path,
        field_label=field_label,
        easyocr_text=easy,
        tess_text=tess,
        chosen=chosen,
        easy_score=easy_score,
        tess_score=tess_score if tess else -1e9,
    )
    return out
