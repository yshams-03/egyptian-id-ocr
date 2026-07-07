"""
Generates FAKE, WATERMARKED test cards. Output must never be usable as a real identity document.

Procedural front/back card rendering — no real photos or copyrighted assets.
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from tests.synthetic.constants import CARD_HEIGHT, CARD_WIDTH
from tests.synthetic.content import SyntheticCardContent
from tests.synthetic.layout import CardLayout, LAYOUT_STANDARD
from tests.synthetic.watermark import apply_watermarks


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        ("C:/Windows/Fonts/tahomabd.ttf", "C:/Windows/Fonts/arialbd.ttf")
        if bold
        else ("C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/arial.ttf")
    )
    for path in names:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_background(draw: ImageDraw.ImageDraw, w: int, h: int, variant: str) -> None:
    base = (235, 228, 210) if variant == "standard" else (228, 236, 245)
    draw.rectangle((0, 0, w, h), fill=base)
    # procedural pyramid / sphinx hint (simple triangles, not photographic)
    draw.polygon([(w * 0.05, h * 0.55), (w * 0.35, h * 0.15), (w * 0.55, h * 0.55)], fill=(210, 200, 175))
    draw.ellipse((w * 0.58, h * 0.62, w * 0.95, h * 0.92), fill=(200, 215, 195))


def _draw_text_in_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (20, 20, 20),
    align_right: bool = True,
) -> None:
    left, top, right, bottom = box
    lines = text.split("\n") if text else [""]
    y = top + 4
    for line in lines:
        if not line:
            y += font.size + 2
            continue
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        line_h = getattr(font, "size", 14) + 2
        if align_right:
            x = right - tw - 6
        else:
            x = left + 6
        draw.text((x, y), line, fill=fill, font=font)
        y += th + 4 if th else line_h
        if y > bottom:
            break


def _draw_photo_placeholder(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, fill=(180, 190, 200), outline=(80, 80, 80), width=2)
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    r = min(right - left, bottom - top) // 5
    draw.ellipse((cx - r, cy - r * 2, cx + r, cy), fill=(140, 150, 165))
    draw.ellipse((cx - r * 2, cy - r, cx + r * 2, cy + r * 3), fill=(140, 150, 165))


def _format_nid_display(nid: str) -> str:
    d = nid
    if len(d) == 14:
        return f"{d[0:7]} {d[7:14]}"
    return d


def _format_dob_display(iso: str) -> str:
    parts = iso.split("-")
    if len(parts) == 3:
        y, m, d = parts
        return f"{d}/{m}/{y}"
    return iso


def render_front(
    content: SyntheticCardContent,
    layout: CardLayout = LAYOUT_STANDARD,
    *,
    width: int = CARD_WIDTH,
    height: int = CARD_HEIGHT,
) -> Image.Image:
    img = Image.new("RGB", (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    _draw_background(draw, width, height, layout.name)

    photo_box = layout.photo.to_pixels(width, height)
    logo_box = layout.front_logo.to_pixels(width, height)
    draw.rectangle(logo_box, fill=(200, 30, 30), outline=(120, 0, 0))
    draw.text(
        (logo_box[0] + 8, logo_box[1] + 8),
        "جمهورية مصر العربية\nبطاقة تحقيق الشخصية",
        fill=(255, 255, 255),
        font=_load_font(max(14, width // 55)),
    )

    _draw_photo_placeholder(draw, photo_box)

    name_font = _load_font(max(16, width // 42))
    field_font = _load_font(max(14, width // 50))
    small_font = _load_font(max(12, width // 58))

    _draw_text_in_box(draw, content.first_name, layout.first_name.to_pixels(width, height), font=name_font)
    _draw_text_in_box(draw, content.last_name, layout.last_name.to_pixels(width, height), font=name_font)
    _draw_text_in_box(draw, content.address, layout.address.to_pixels(width, height), font=field_font)
    _draw_text_in_box(
        draw,
        _format_dob_display(content.dob),
        layout.dob.to_pixels(width, height),
        font=field_font,
        align_right=False,
    )
    _draw_text_in_box(
        draw,
        _format_nid_display(content.national_id),
        layout.nid.to_pixels(width, height),
        font=field_font,
    )
    _draw_text_in_box(
        draw,
        content.serial,
        layout.serial.to_pixels(width, height),
        font=small_font,
        align_right=False,
    )

    return apply_watermarks(img, photo_box=photo_box)


def render_back(
    content: SyntheticCardContent,
    layout: CardLayout = LAYOUT_STANDARD,
    *,
    width: int = CARD_WIDTH,
    height: int = CARD_HEIGHT,
) -> Image.Image:
    img = Image.new("RGB", (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    _draw_background(draw, width, height, layout.name)

    field_font = _load_font(max(14, width // 50))
    small_font = _load_font(max(12, width // 58))

    _draw_text_in_box(draw, content.back_nid, layout.nid_back.to_pixels(width, height), font=small_font)
    _draw_text_in_box(draw, content.job, layout.job.to_pixels(width, height), font=field_font)
    demo_text = f"{content.gender}  {content.marital_status}  {content.religion}"
    _draw_text_in_box(draw, demo_text, layout.demo.to_pixels(width, height), font=field_font)
    _draw_text_in_box(
        draw,
        _format_dob_display(content.expiry_date),
        layout.expiry.to_pixels(width, height),
        font=field_font,
    )
    _draw_text_in_box(
        draw,
        _format_dob_display(content.issue_date),
        layout.issue.to_pixels(width, height),
        font=small_font,
        align_right=False,
    )

    tut_box = layout.watermark_tut.to_pixels(width, height)
    draw.rectangle(tut_box, outline=(150, 150, 150), width=1)
    draw.text((tut_box[0] + 4, tut_box[1] + 4), "TUT", fill=(100, 100, 100), font=small_font)

    return apply_watermarks(img, photo_box=tut_box)
