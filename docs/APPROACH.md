# MyCaptchaOCR Approach

The project treats OCR as the last step. The important work is to preserve
Chinese glyph topology while reducing background noise, dots, and line
interference.

## Pipeline

1. Crop the captcha body from the source image using brightness projection.
2. Upscale the cropped captcha before destructive operations.
3. Generate multiple preprocessing families:
   - original ROI at 1x to 5x
   - light background flattening
   - tiny-dot inpaint
   - low-saturation dark-line suppression
   - horizontal dark-line inpaint
   - adaptive dominant-color extraction
   - small right/bottom crops
4. Run ddddocr default, beta, and old recognizers. The default adaptive profile
   starts with 75 no-crop variants and only expands to a capped fallback set
   when the early OCR consensus is low confidence. The default cap is 455
   candidates so ambiguous images get more evidence without paying full-profile
   cost. The desktop app ships with `OCR_FORCE_BALANCED = True`, which skips the
   early exit and always runs the 455-candidate balanced path; candidate
   inference is parallelized across CPU cores, so a balanced run is about 9s on a
   6-core CPU instead of about 37s when run serially.
5. Normalize OCR text to Chinese characters.
6. Rerank by:
   - expected length
   - character-position evidence
   - OCR engine diversity
   - preprocessing-family diversity
   - direct candidate hits

## Scoring Notes

Many generated images are near duplicates. A single OCR model can repeat the
same wrong result across dozens of similar variants, so direct hits from the
same engine are capped more aggressively than cross-engine agreement.

For ambiguous samples, the pipeline should expose top-N candidates instead of
blindly trusting top-1. This is especially important when the last character is
changed by a trailing line, such as `九` being misread as `力`.

## Sample Naming

Tracked sample files use ASCII names under `data/raw/`. Chinese captcha samples
use `sample-<six-digit-key>.png`, which lets the scripts derive a stable sample
key without depending on localized screenshot filenames.

## Why PaddleOCR Was Removed

PaddleOCR is strong for ordinary documents and scene text, but these images are
captcha-like: distorted, line-covered, dotted, and low-resolution. In local
tests, PaddleOCR's detector split interference lines and glyph fragments into
bad text boxes. Recognition-only also produced unstable results on processed
captcha candidates. Keeping PaddleOCR made the project heavier without
improving the final decision.
