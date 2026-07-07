"""Split guard and source tagging tests."""
from tests.labeling.split_guard import assign_yolo_split, check_promotion_allowed
from tests.labeling.sources import (
    SOURCE_ROBOFLOW_TEST,
    SOURCE_ROBOFLOW_TRAIN,
    SOURCE_ROBOFLOW_VALID,
)


def test_held_out_never_assigned_train():
    for stem in ("a", "b", "real_foo"):
        assert assign_yolo_split(SOURCE_ROBOFLOW_VALID, stem) == "valid"
        assert assign_yolo_split(SOURCE_ROBOFLOW_TEST, stem) == "valid"
        check_promotion_allowed(SOURCE_ROBOFLOW_TEST, stem)


def test_train_source_can_hash_to_train():
    # find a stem that hashes to train for roboflow_train
    found = False
    for i in range(500):
        stem = f"stem_{i}"
        if assign_yolo_split(SOURCE_ROBOFLOW_TRAIN, stem) == "train":
            found = True
            break
    assert found
