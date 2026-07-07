"""CLI wrapper for tests.labeling.inventory."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.labeling.inventory import build_inventory, print_inventory_report
from tests.labeling.import_roboflow import DEFAULT_OUT

DATA_DIR = ROOT / "test_data" / "id_cards"


def main() -> int:
    report = build_inventory(DATA_DIR)
    print_inventory_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
