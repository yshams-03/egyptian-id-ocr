"""Guardrail tests — synthetic generator must stay visibly fake and use reserved NIDs."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from PIL import Image

from egypt_nid_decode import decode_egyptian_nid
from tests.synthetic.constants import (
    SERIAL_PREFIX,
    SYNTHETIC_GOVERNORATE_CODE,
    WATERMARK_CORNER_TEXT_AR,
    WATERMARK_CORNER_TEXT_EN,
)
from tests.synthetic.content import generate_content, is_placeholder_first_name
from tests.synthetic.generate import generate_one
from tests.synthetic.layout import LAYOUT_STANDARD
from tests.synthetic.render import render_front
from tests.synthetic.watermark import watermark_pixel_score


def test_synthetic_nid_uses_reserved_governorate_code():
    content = generate_content(rng=__import__("random").Random(42))
    assert content.national_id[7:9] == SYNTHETIC_GOVERNORATE_CODE
    dec = decode_egyptian_nid(content.national_id)
    assert dec.governorate_code == SYNTHETIC_GOVERNORATE_CODE
    assert dec.governorate == "Synthetic Test"
    assert dec.birth_date == content.dob


def test_synthetic_serial_has_test_prefix():
    content = generate_content(rng=__import__("random").Random(7))
    assert content.serial.startswith(SERIAL_PREFIX)


def test_synthetic_name_from_placeholder_pool():
    rng = __import__("random").Random(99)
    for _ in range(20):
        content = generate_content(rng=rng)
        assert is_placeholder_first_name(content.first_name), content.first_name


def test_generated_image_has_watermark_signal():
    content = generate_content(rng=__import__("random").Random(1))
    img = render_front(content, LAYOUT_STANDARD)
    photo_box = LAYOUT_STANDARD.photo.to_pixels(img.width, img.height)
    score = watermark_pixel_score(img, photo_box)
    assert score > 0.02, "expected watermark tint in photo region"


def test_generate_one_writes_gitignored_paths_and_valid_json():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        front, back, gt = generate_one(out, tags=["synthetic", "blurry"], rng=__import__("random").Random(0))
        assert front.is_file()
        assert back and back.is_file()
        assert front.suffix == ".jpg"
        jpath = front.with_suffix(".json")
        assert jpath.is_file()
        loaded = json.loads(jpath.read_text(encoding="utf-8"))
        assert loaded["national_id"] == gt["national_id"]
        assert "synthetic" in loaded["tags"]
        assert loaded["national_id"][7:9] == SYNTHETIC_GOVERNORATE_CODE
        assert is_placeholder_first_name(loaded["first_name"])

        # corner text present in raster (rough check via getbbox on crop not needed — read pixels)
        rgb = Image.open(front).convert("RGB")
        w, h = rgb.size
        corner = rgb.crop((0, 0, min(200, w), min(80, h)))
        # watermark uses orange tint; corner should not be uniform background
        pixels = list(corner.get_flattened_data())
        assert max(p[0] for p in pixels) > 150


def test_governorate_88_is_real_foreign_not_synthetic_reserved():
    """Document design choice: 88 = Foreign in real table; synthetic uses 99."""
    from egypt_nid_decode import EGYPT_NID_GOVERNORATES

    assert EGYPT_NID_GOVERNORATES["88"] == "Foreign"
    assert SYNTHETIC_GOVERNORATE_CODE == "99"
    assert SYNTHETIC_GOVERNORATE_CODE not in EGYPT_NID_GOVERNORATES or (
        EGYPT_NID_GOVERNORATES[SYNTHETIC_GOVERNORATE_CODE] == "Synthetic Test"
    )
