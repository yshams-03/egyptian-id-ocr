"""
Generates FAKE, WATERMARKED test cards. Output must never be usable as a real identity document.

SPECIMEN / عينة اختبار watermarks — corner + diagonal band across photo area.
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from tests.synthetic.constants import (
    WATERMARK_CORNER_TEXT_AR,
    WATERMARK_CORNER_TEXT_EN,
    WATERMARK_DIAGONAL_TEXT,
    WATERMARK_TINT_RGBA,
)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def apply_watermarks(
    img: Image.Image,
    *,
    photo_box: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    """Overlay unmistakable test-sample marks; returns RGB image."""
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = base.size

    corner_font = _load_font(max(18, w // 28))
    band_font = _load_font(max(28, w // 14))

    # Corner marks
    draw.text((12, 8), WATERMARK_CORNER_TEXT_EN, fill=WATERMARK_TINT_RGBA, font=corner_font)
    draw.text((12, h - 36), WATERMARK_CORNER_TEXT_AR, fill=WATERMARK_TINT_RGBA, font=corner_font)

    # Diagonal band across photo (or full card if no box)
    if photo_box:
        left, top, right, bottom = photo_box
        cx = (left + right) // 2
        cy = (top + bottom) // 2
        band_w = right - left
        band_h = bottom - top
        # Corner stamp inside photo (always detectable for guardrail tests)
        stamp = Image.new("RGBA", (band_w, band_h), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(stamp)
        sdraw.text((6, 6), WATERMARK_CORNER_TEXT_EN, fill=WATERMARK_TINT_RGBA, font=corner_font)
        sdraw.text((6, band_h - 28), WATERMARK_CORNER_TEXT_AR, fill=WATERMARK_TINT_RGBA, font=corner_font)
        overlay.paste(stamp, (left, top), stamp)
    else:
        cx, cy = w // 2, h // 2
        band_w, band_h = w, h

    diag = Image.new("RGBA", (band_w, band_h), (0, 0, 0, 0))
    ddraw = ImageDraw.Draw(diag)
    ddraw.text(
        (band_w // 10, band_h // 3),
        WATERMARK_DIAGONAL_TEXT,
        fill=WATERMARK_TINT_RGBA,
        font=band_font,
    )
    diag = diag.rotate(35, expand=True, resample=Image.Resampling.BICUBIC)
    paste_x = cx - diag.width // 2
    paste_y = cy - diag.height // 2
    overlay.paste(diag, (paste_x, paste_y), diag)

    # faint full-card diagonal (extra safeguard)
    for i in range(-2, 3):
        y0 = int(h * 0.15 * i)
        draw.line([(0, y0), (w, y0 + w // 2)], fill=(255, 69, 0, 45), width=max(2, w // 200))

    out = Image.alpha_composite(base, overlay).convert("RGB")
    if photo_box:
        # Burn-in photo stamp (guardrail-detectable; still clearly a test mark)
        left, top, _, _ = photo_box
        draw_rgb = ImageDraw.Draw(out)
        draw_rgb.text(
            (left + 6, top + 6),
            WATERMARK_CORNER_TEXT_EN,
            fill=(220, 90, 40),
            font=corner_font,
        )
    return out


def watermark_pixel_score(img: Image.Image, photo_box: tuple[int, int, int, int]) -> float:
    """
    Fraction of photo-area pixels with elevated orange watermark tint.
    Used by guardrail tests only.
    """
    rgb = img.convert("RGB")
    left, top, right, bottom = photo_box
    left = max(0, left)
    top = max(0, top)
    hits = 0
    total = 0
    for y in range(top, min(bottom, rgb.height), 4):
        for x in range(left, min(right, rgb.width), 4):
            r, g, b = rgb.getpixel((x, y))
            if r > 150 and (r - g) > 40 and b < 120:
                hits += 1
            total += 1
    return hits / total if total else 0.0
