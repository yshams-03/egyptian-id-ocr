"""Fast tests for labeling utilities (no YOLO weights)."""
from pathlib import Path

from tests.labeling.yolo_boxes import YoloBox, read_draft_label_file, write_draft_label_file


def test_yolo_label_roundtrip(tmp_path: Path):
    boxes = [
        YoloBox(class_id=26, class_name="nid", cx=0.5, cy=0.5, w=0.3, h=0.1, conf=0.9),
        YoloBox(class_id=0, class_name="address", cx=0.7, cy=0.6, w=0.2, h=0.15, conf=0.8),
    ]
    p = tmp_path / "sample.txt"
    write_draft_label_file(boxes, p)
    loaded = read_draft_label_file(p, {0: "address", 26: "nid"})
    assert len(loaded) == 2
    assert abs(loaded[0].cx - 0.5) < 1e-5
