"""Read raw text out of PDF, plain text, and markdown files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

from ..text_utils import normalize_text

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text", ""}

PathLike = Union[str, Path]


class UnsupportedFileError(ValueError):
    pass


def extract_pdf_text(path: PathLike) -> str:
    import pdfplumber

    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            # A small x_tolerance keeps words from being glued together in tight layouts.
            text = page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""
            pages.append(text)
    text = "\n".join(pages)
    # Fonts without a unicode map leave "(cid:127)" style codes; at line start these are bullet glyphs.
    text = re.sub(r"(?m)^(\s*)\(cid:\d+\)\s*", "\\1- ", text)
    return re.sub(r"\(cid:\d+\)", "", text)


def extract_text(path: PathLike) -> str:
    """Return normalized text for a resume or job description file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw = extract_pdf_text(path)
    elif suffix in TEXT_SUFFIXES:
        raw = path.read_text(encoding="utf-8", errors="replace")
    else:
        raise UnsupportedFileError(f"Unsupported file type '{suffix}'. Use .pdf, .txt, or .md.")
    text = normalize_text(raw)
    if not text:
        raise ValueError(f"No text could be extracted from {path}. Scanned PDFs need OCR first.")
    return text
