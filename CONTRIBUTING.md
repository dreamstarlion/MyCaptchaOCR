# Contributing

MyCaptchaOCR is a small OCR research pipeline. Keep changes reproducible and
avoid filename-specific or label-specific rules.

## Development

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/check_ocr_env.py
```

Run the sample pipeline before submitting behavior changes:

```bash
.venv/bin/python scripts/adaptive_ocr_pipeline.py
.venv/bin/python scripts/adaptive_ocr_rerank.py --top-n 12
```

## Guidelines

- Prefer preprocessing and scoring changes that generalize across samples.
- Do not hardcode expected labels or sample filenames into OCR logic.
- Keep generated outputs under `data/processed/` and `reports/`; both are
  ignored by Git.
- Add or update docs when defaults, dependencies, or output formats change.
