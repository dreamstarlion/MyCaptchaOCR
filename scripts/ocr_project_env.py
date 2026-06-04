#!/usr/bin/env python3
"""Project-local environment setup for OCR scripts."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OCR_CACHE_ROOT = PROJECT_ROOT / ".ocr-cache"


def configure_ocr_environment() -> dict[str, Path]:
    paths = {
        "cache_root": OCR_CACHE_ROOT,
        "ddddocr": OCR_CACHE_ROOT / "ddddocr",
        "xdg": OCR_CACHE_ROOT / "xdg",
        "tmp": OCR_CACHE_ROOT / "tmp",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("XDG_CACHE_HOME", str(paths["xdg"]))
    os.environ.setdefault("TMPDIR", str(paths["tmp"]))
    os.environ.setdefault("TEMP", str(paths["tmp"]))
    os.environ.setdefault("TMP", str(paths["tmp"]))
    return paths
