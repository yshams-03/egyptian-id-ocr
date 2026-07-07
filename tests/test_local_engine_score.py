"""Unit tests for local engine selection scorer."""
from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import local_engine_select as les

LEX = json.loads((Path(__file__).resolve().parents[1] / "scripts" / "lexicon" / "egyptian_lexicon.json").read_text(encoding="utf-8"))


def test_module_imports_and_stats_order():
    """EngineSelectStats must be defined before module-level _stats = EngineSelectStats()."""
    src = Path(les.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    class_line = stats_line = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "EngineSelectStats":
            class_line = node.lineno
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_stats":
                    stats_line = node.lineno
    assert class_line is not None and stats_line is not None
    assert class_line < stats_line, f"EngineSelectStats (L{class_line}) must precede _stats (L{stats_line})"
    importlib.reload(les)
    assert les.get_engine_select_stats() is not None


def test_garbage_scores_lower_than_real_name():
    bad = les.score_name_text("4م٧", LEX)
    good = les.score_name_text("عبدالعزيز", LEX)
    assert good > bad + 50


def test_compound_name_scores_high():
    s = les.score_name_text("محمد عبدالعزيز احمد", LEX)
    assert s > 100


def test_empty_scores_very_low():
    assert les.score_name_text("", LEX) < -1e8


def test_address_connector_bonus():
    with_conn = les.score_address_text("شارع الحاجه ست عزبة فرج واصل", LEX)
    without = les.score_address_text("xyz gibberish token soup", LEX)
    assert with_conn > without
