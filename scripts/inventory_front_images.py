"""CLI wrapper for tests.labeling.inventory."""
from tests.labeling.inventory import build_inventory, print_inventory_report
from tests.labeling.import_roboflow import DEFAULT_OUT
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "test_data" / "id_cards"


def main() -> int:
    report = build_inventory(DATA_DIR)
    print_inventory_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
