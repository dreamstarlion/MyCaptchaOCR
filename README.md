# MyCaptchaOCR

OpenCV + ddddocr pipeline for noisy Chinese captcha images.

MyCaptchaOCR focuses on preprocessing and evidence-based reranking before OCR.
It generates multiple image variants, reads them with ddddocr, and reranks OCR
outputs by consensus, character-position evidence, and preprocessing diversity.

PaddleOCR is intentionally not included. On the bundled distorted captcha
samples, PaddleOCR detection treated interference lines and glyph fragments as
text boxes, while recognition-only mode produced unstable Chinese predictions.

## Features

- Crops the captcha body from source images.
- Upscales before denoising to preserve Chinese glyph strokes.
- Builds conservative, color-priority, line-inpaint, dark-line-suppression, and
  crop variants.
- Runs ddddocr default, beta, and old recognizers.
- Uses an adaptive default profile: stable images run a small candidate set,
  while low-confidence images fall back to the full candidate set.
- Reranks candidates while reducing false consensus from near-duplicate
  preprocessing variants.
- Writes reproducible CSV, Markdown, and visual sheet outputs.

## Requirements

- Python 3.11 through 3.14.
- Current pinned runtime dependencies:
  - `ddddocr==1.6.1`
  - `onnxruntime==1.26.0`
  - `opencv-python==4.13.0.92`
  - `pillow==12.2.0`
  - `numpy==2.4.6`
  - `PySide6==6.11.1`

`opencv-contrib-python` and `pandas` are not required.

## Setup

Use a local virtual environment. Do not install dependencies globally.

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
```

Check the runtime:

```bash
.venv/bin/python scripts/check_ocr_env.py
```

On Windows PowerShell, use a Windows virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\check_ocr_env.py
```

The check prints Python, ddddocr, ONNX Runtime, OpenCV, NumPy, Pillow, PySide6,
and the project-local OCR cache paths.

## Usage

Run the default Chinese sample set:

```bash
.venv/bin/python scripts/adaptive_ocr_pipeline.py
.venv/bin/python scripts/adaptive_ocr_rerank.py --top-n 12
```

Run the mini desktop UI:

```bash
.venv/bin/python scripts/ocr_desktop_app.py
```

On Windows:

```powershell
.\.venv\Scripts\python.exe scripts\ocr_desktop_app.py
```

Click `选取区域`, drag over the screen region to recognize, then click `识别`.
The UI keeps OCR engines warm after the first recognition so later runs avoid
model reload time. On macOS, grant Screen Recording permission if the selection
capture is blank or blocked.

See [docs/DESKTOP_UI.md](docs/DESKTOP_UI.md) for platform notes and packaging
commands.

The default input pattern is `sample-[0-9]*.png`, which covers the five bundled
Chinese captcha samples in `data/raw/`. To process every raw PNG, including the
alphanumeric smoke-test image, use:

```bash
.venv/bin/python scripts/adaptive_ocr_pipeline.py --input-dir data/raw --pattern "*.png"
.venv/bin/python scripts/adaptive_ocr_rerank.py --top-n 12
```

Use a different input directory:

```bash
.venv/bin/python scripts/adaptive_ocr_pipeline.py --input-dir /path/to/images --pattern "*.png"
.venv/bin/python scripts/adaptive_ocr_rerank.py --top-n 12
```

### Profiles

- `--profile adaptive` is the default. It first runs 75 no-crop candidates per
  image, then expands to a capped fallback set only when the early result is low
  confidence. The default fallback cap is 455 candidates; use
  `--adaptive-full-limit` to tune the speed/accuracy tradeoff.
- `--profile fast` always uses the 75-candidate set.
- `--profile full` always uses the full generated candidate set.

Each image prints `mode`, candidate count, OCR row count, top candidate, and
preprocessing/OCR/total seconds. `adaptive-fast` means the early result was
accepted; `adaptive-balanced` means the image expanded to the capped fallback
candidate set.

## Outputs

Generated files are ignored by Git:

```text
data/processed/adaptive_ocr/       generated candidate images
reports/adaptive_ocr_rows.csv      OCR rows
reports/adaptive_ocr_scores.csv    first-stage text scores
reports/adaptive_ocr_summary.md    first-stage summary
reports/adaptive_ocr_top_sheet.png visual first-stage sheet
reports/adaptive_ocr_v2_combinations.csv
reports/adaptive_ocr_v2_summary.md
reports/adaptive_ocr_v2_top_sheet.png
```

## Sample Results

See [docs/RESULTS.md](docs/RESULTS.md) for current sample outputs. The most
ambiguous sample is `sample-034918.png`; the reranker keeps `狱己擦九` and
`狱己擦力` close because the final character has nearly tied evidence.

Recent local timing on Python 3.14.5 with the default adaptive profile:

| sample set | average total time | notes |
| --- | ---: | --- |
| 5 Chinese samples | 5.64s/image | stable samples complete in about 1.5-1.8s |
| full profile baseline | 18.58s/image | every image runs all 650 candidates |

## Project Layout

```text
data/raw/                 tracked sample input images
data/processed/           generated candidate images, ignored by Git
docs/                     design notes and sample results
reports/                  generated CSV/Markdown/PNG reports, ignored by Git
scripts/                  runnable pipeline scripts
requirements.txt          pinned runtime dependencies
pyproject.toml            project metadata
```

## Limitations

- This is an OCR research pipeline, not a CAPTCHA-bypass service.
- Use it only for images you own or are authorized to analyze.
- It assumes the target text is Chinese and four characters long by default;
  override `--expected-len` for other lengths.
- The adaptive profile is tuned on the bundled samples. Use `--profile full`
  when evaluating new captcha styles.

## License

MIT. See [LICENSE](LICENSE).
