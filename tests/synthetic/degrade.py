"""
Generates FAKE, WATERMARKED test cards. Output must never be usable as a real identity document.

Composable image degradations matching harness edge-case tags.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
from PIL import Image


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def apply_rotation(img: Image.Image, degrees: float) -> Image.Image:
    bgr = _pil_to_bgr(img)
    h, w = bgr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    out = cv2.warpAffine(
        bgr,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return _bgr_to_pil(out)


def apply_blur(img: Image.Image, ksize: int = 7) -> Image.Image:
    bgr = _pil_to_bgr(img)
    k = ksize if ksize % 2 == 1 else ksize + 1
    out = cv2.GaussianBlur(bgr, (k, k), 0)
    return _bgr_to_pil(out)


def apply_glare(
    img: Image.Image,
    *,
    center: tuple[float, float] = (0.55, 0.35),
    radius_frac: float = 0.22,
    intensity: float = 0.75,
) -> Image.Image:
    bgr = _pil_to_bgr(img).astype(np.float32)
    h, w = bgr.shape[:2]
    cx, cy = int(center[0] * w), int(center[1] * h)
    radius = int(radius_frac * min(w, h))
    yy, xx = np.ogrid[:h, :w]
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2
    bgr[mask] = bgr[mask] * (1 - intensity) + 255 * intensity
    return _bgr_to_pil(np.clip(bgr, 0, 255).astype(np.uint8))


def apply_occlusion(
    img: Image.Image,
    *,
    box_frac: tuple[float, float, float, float] = (0.6, 0.5, 0.25, 0.12),
    color: tuple[int, int, int] = (40, 40, 40),
) -> Image.Image:
    bgr = _pil_to_bgr(img)
    h, w = bgr.shape[:2]
    cx, cy, bw, bh = box_frac
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    cv2.rectangle(bgr, (x1, y1), (x2, y2), color, thickness=-1)
    return _bgr_to_pil(bgr)


def apply_resolution(img: Image.Image, scale: float = 0.45) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
    return small.resize((w, h), Image.Resampling.BILINEAR)


DEGRADE_BY_TAG: dict[str, Callable[[Image.Image], Image.Image]] = {
    "rotated": lambda im: apply_rotation(im, 8.0),
    "skewed": lambda im: apply_rotation(im, 14.0),
    "blurry": lambda im: apply_blur(im, 9),
    "low_resolution": lambda im: apply_resolution(im, 0.4),
    "glare": lambda im: apply_glare(im),
    "partial_occlusion": lambda im: apply_occlusion(im),
}


def apply_degradations(img: Image.Image, tags: set[str]) -> Image.Image:
    out = img
    for tag in ("low_resolution", "blurry", "glare", "partial_occlusion", "rotated", "skewed"):
        if tag in tags and tag in DEGRADE_BY_TAG:
            out = DEGRADE_BY_TAG[tag](out)
    return out
