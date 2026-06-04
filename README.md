# cn-captcha-ocr

OpenCV + ddddocr pipeline for noisy Chinese captcha images.

This project focuses on image preprocessing before OCR. It generates multiple
candidate views of each captcha, reads them with ddddocr, and reranks the OCR
outputs with character-position evidence. PaddleOCR was removed because its
detector and recognizer were unreliable on this captcha style.

## Features

- Crops the captcha body from screenshots.
- Enlarges before denoising to preserve Chinese glyph strokes.
- Builds conservative, color-priority, line-inpaint, dark-line-suppression, and crop variants.
- Runs ddddocr default, beta, and old recognizers.
- Reranks candidates while reducing false consensus from repeated near-duplicate preprocessing.
- Writes reproducible CSV, Markdown, and visual sheet outputs.

## Project Layout

```text
data/raw/                 sample input images
data/processed/           generated candidate images, ignored by Git
docs/                     design notes and sample results
reports/                  generated CSV/Markdown/PNG reports, ignored by Git
scripts/                  runnable pipeline scripts
requirements.txt          runtime dependencies
```

## Setup

Use a local virtual environment. Do not install dependencies globally.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
```

Check the runtime:

```bash
.venv/bin/python scripts/check_ocr_env.py
```

## Run

Process the sample images:

```bash
.venv/bin/python scripts/adaptive_ocr_pipeline.py
.venv/bin/python scripts/adaptive_ocr_rerank.py --top-n 12
```

Use a different input directory:

```bash
.venv/bin/python scripts/adaptive_ocr_pipeline.py --input-dir /path/to/images --pattern "*.png"
.venv/bin/python scripts/adaptive_ocr_rerank.py --top-n 12
```

Main outputs:

```text
reports/adaptive_ocr_rows.csv
reports/adaptive_ocr_scores.csv
reports/adaptive_ocr_summary.md
reports/adaptive_ocr_v2_combinations.csv
reports/adaptive_ocr_v2_summary.md
reports/adaptive_ocr_v2_top_sheet.png
```

## Current Decision

PaddleOCR is intentionally not part of this project. On these distorted captcha
images, full PaddleOCR tends to detect interference lines and fragments as text,
while recognition-only also produced unstable Chinese predictions. The current
working path is OpenCV candidate generation plus ddddocr reranking.

See [docs/APPROACH.md](docs/APPROACH.md) for the image-processing strategy and
[docs/RESULTS.md](docs/RESULTS.md) for the current sample outcomes.
