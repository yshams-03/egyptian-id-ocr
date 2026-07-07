"""
Generates FAKE, WATERMARKED test cards. Output must never be usable as a real identity document.

Field regions derived from Egyptian-ID-Detectr-3 YOLO label averages (train split).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldBox:
    """Normalized YOLO-style box: center x/y and width/height in 0–1."""

    cx: float
    cy: float
    w: float
    h: float

    def to_pixels(self, card_w: int, card_h: int) -> tuple[int, int, int, int]:
        """Return (left, top, right, bottom) pixel box."""
        pw = self.w * card_w
        ph = self.h * card_h
        px = self.cx * card_w
        py = self.cy * card_h
        left = int(px - pw / 2)
        top = int(py - ph / 2)
        right = int(px + pw / 2)
        bottom = int(py + ph / 2)
        return left, top, right, bottom


@dataclass(frozen=True)
class CardLayout:
    name: str
    photo: FieldBox
    front_logo: FieldBox
    first_name: FieldBox
    last_name: FieldBox
    address: FieldBox
    dob: FieldBox
    nid: FieldBox
    serial: FieldBox
    # back
    job: FieldBox
    demo: FieldBox
    expiry: FieldBox
    issue: FieldBox
    nid_back: FieldBox
    watermark_tut: FieldBox


# Averaged from egyptian_id_detectr/.../train/labels (31-class detector).
LAYOUT_STANDARD = CardLayout(
    name="standard",
    photo=FieldBox(0.188, 0.376, 0.213, 0.428),
    front_logo=FieldBox(0.615, 0.202, 0.341, 0.164),
    first_name=FieldBox(0.831, 0.361, 0.079, 0.066),
    last_name=FieldBox(0.718, 0.437, 0.303, 0.073),
    address=FieldBox(0.727, 0.563, 0.294, 0.139),
    dob=FieldBox(0.243, 0.735, 0.292, 0.161),
    nid=FieldBox(0.667, 0.769, 0.438, 0.076),
    serial=FieldBox(0.243, 0.854, 0.209, 0.065),
    job=FieldBox(0.627, 0.234, 0.294, 0.060),
    demo=FieldBox(0.570, 0.375, 0.403, 0.071),
    expiry=FieldBox(0.561, 0.518, 0.427, 0.070),
    issue=FieldBox(0.368, 0.168, 0.132, 0.054),
    nid_back=FieldBox(0.628, 0.168, 0.287, 0.054),
    watermark_tut=FieldBox(0.165, 0.292, 0.184, 0.316),
)

# Slightly shifted typography — simulates newer card print variant.
LAYOUT_NEW = CardLayout(
    name="new",
    photo=FieldBox(0.195, 0.370, 0.210, 0.420),
    front_logo=FieldBox(0.620, 0.195, 0.335, 0.160),
    first_name=FieldBox(0.820, 0.350, 0.090, 0.070),
    last_name=FieldBox(0.710, 0.425, 0.310, 0.075),
    address=FieldBox(0.720, 0.555, 0.300, 0.145),
    dob=FieldBox(0.250, 0.725, 0.285, 0.155),
    nid=FieldBox(0.660, 0.760, 0.445, 0.080),
    serial=FieldBox(0.250, 0.845, 0.215, 0.068),
    job=FieldBox(0.635, 0.240, 0.288, 0.062),
    demo=FieldBox(0.575, 0.380, 0.398, 0.068),
    expiry=FieldBox(0.565, 0.525, 0.420, 0.072),
    issue=FieldBox(0.375, 0.172, 0.128, 0.052),
    nid_back=FieldBox(0.632, 0.172, 0.282, 0.052),
    watermark_tut=FieldBox(0.170, 0.295, 0.178, 0.310),
)

LAYOUTS = {"standard": LAYOUT_STANDARD, "old": LAYOUT_STANDARD, "new": LAYOUT_NEW}
