# MyCaptchaOCR Approach

The project treats OCR as the last step. The important work is to preserve
Chinese glyph topology while reducing background noise, dots, and line
interference.

## Pipeline

1. Crop the captcha body from the screenshot using brightness projection.
2. Upscale the cropped captcha before destructive operations.
3. Generate multiple preprocessing families:
   - original ROI at 1x to 5x
   - light background flattening
   - tiny-dot inpaint
   - low-saturation dark-line suppression
   - horizontal dark-line inpaint
   - adaptive dominant-color extraction
   - small right/bottom crops
4. Run ddddocr default, beta, and old recognizers on all candidates.
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

## Why PaddleOCR Was Removed

PaddleOCR is strong for ordinary documents and scene text, but these images are
captcha-like: distorted, line-covered, dotted, and low-resolution. In local
tests, PaddleOCR's detector split interference lines and glyph fragments into
bad text boxes. Recognition-only also produced unstable results on processed
captcha candidates. Keeping PaddleOCR made the project heavier without
improving the final decision.
