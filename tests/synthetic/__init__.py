"""
Generates FAKE, WATERMARKED test cards. Output must never be usable as a real identity document.

Synthetic Egyptian ID fixtures for the extraction test pipeline only.
"""

from tests.synthetic.generate import generate_batch, generate_one

__all__ = ["generate_one", "generate_batch"]
