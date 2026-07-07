#!/usr/bin/env python
"""Build name/address lexicon from train-source ground truth only (no held-out leakage)."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.ground_truth import discover_test_cases
from tests.id_metrics import normalize_arabic_text
from tests.labeling.sources import (
    HELD_OUT_SOURCES,
    SOURCE_MANUAL,
    SOURCE_ROBOFLOW_TRAIN,
)

OUT = Path(__file__).resolve().parent / "egyptian_lexicon.json"
_ARABIC_WORD = re.compile(r"[\u0600-\u06FF]+")
_CONNECTORS = {
    "شارع",
    "ش",
    "حي",
    "حى",
    "مركز",
    "قرية",
    "عزبة",
    "عزبه",
    "كفر",
    "مدينة",
    "مدينه",
    "التجمع",
    "الاول",
    "الأول",
    "ثان",
    "ثانى",
    "ثاني",
    "منطقة",
    "منطقه",
    "ميدان",
    "برج",
    "عمارة",
    "عماره",
    "دور",
    "شقة",
    "شقه",
    "بجوار",
    "امام",
    "أمام",
    "خلف",
    "بين",
    "طريق",
}
_COMPOUND_PREFIXES = ("عبد", "ابو", "أبو", "ام", "ام", "بن", "ابن")


def _tokens_from_text(text: str) -> list[str]:
    t = normalize_arabic_text(text)
    if not t:
        return []
    parts = re.split(r"[\s\.\,\-\–\—/\\]+", t)
    return [p for p in parts if p and _ARABIC_WORD.search(p)]


def _eligible_sources() -> frozenset[str]:
    return frozenset({SOURCE_ROBOFLOW_TRAIN, SOURCE_MANUAL})


def build_lexicon(data_dir: Path) -> dict:
    name_tokens: Counter[str] = Counter()
    address_tokens: Counter[str] = Counter()
    compounds: Counter[str] = Counter()
    governorates: Counter[str] = Counter()
    n_samples = 0

    for case in discover_test_cases(data_dir):
        gt = case["ground_truth"]
        source = str(gt.get("source") or "").strip()
        if source in HELD_OUT_SOURCES:
            continue
        if source not in _eligible_sources():
            continue

        name = (
            (gt.get("full_name") or "").strip()
            or (gt.get("name") or "").strip()
            or f"{gt.get('first_name', '')} {gt.get('last_name', '')}".strip()
        )
        addr = (gt.get("address") or "").strip()
        if not name and not addr:
            continue
        n_samples += 1

        for tok in _tokens_from_text(name):
            name_tokens[tok] += 1
            for pref in _COMPOUND_PREFIXES:
                if tok.startswith(pref) and len(tok) > len(pref) + 1:
                    compounds[tok] += 1

        for tok in _tokens_from_text(addr):
            address_tokens[tok] += 1
            if tok in _CONNECTORS:
                address_tokens[tok] += 2
            # governorate often after last '-' on second line
        for line in re.split(r"[\n\r]+", addr):
            if "-" in line:
                tail = line.split("-")[-1].strip()
                for tok in _tokens_from_text(tail):
                    if len(tok) >= 4:
                        governorates[tok] += 1

        address_tokens.update(_CONNECTORS)

    return {
        "meta": {
            "n_samples": n_samples,
            "sources": sorted(_eligible_sources()),
            "excluded": sorted(HELD_OUT_SOURCES),
        },
        "name_tokens": [t for t, _ in name_tokens.most_common()],
        "name_compounds": [t for t, _ in compounds.most_common()],
        "address_tokens": [t for t, _ in address_tokens.most_common()],
        "governorates": [t for t, _ in governorates.most_common()],
        "connectors": sorted(_CONNECTORS),
    }


def main() -> int:
    data_dir = ROOT / "test_data" / "id_cards"
    lex = build_lexicon(data_dir)
    OUT.write_text(json.dumps(lex, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(
        f"  samples={lex['meta']['n_samples']} "
        f"names={len(lex['name_tokens'])} compounds={len(lex['name_compounds'])} "
        f"addr={len(lex['address_tokens'])} gov={len(lex['governorates'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
