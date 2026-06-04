#!/usr/bin/env python3
"""Validate the project-local OCR runtime."""

from __future__ import annotations

import sys
from importlib.metadata import version

from ocr_project_env import configure_ocr_environment


def main() -> None:
    paths = configure_ocr_environment()

    import cv2
    import ddddocr
    import numpy
    import pandas
    import PIL

    print(f"python={sys.version.split()[0]}")
    print(f"ddddocr={version('ddddocr')}")
    print(f"opencv={cv2.__version__}")
    print(f"numpy={numpy.__version__}")
    print(f"pandas={pandas.__version__}")
    print(f"pillow={PIL.__version__}")
    for name, path in paths.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
