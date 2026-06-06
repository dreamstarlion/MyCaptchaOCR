#!/usr/bin/env python3
"""Single-image OCR service used by the desktop UI."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from adaptive_ocr_pipeline import create_ocr_engines, run_image
from adaptive_ocr_rerank import score_combinations


@dataclass(frozen=True)
class RecognitionCandidate:
    text: str
    score: float


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    score: float | None
    mode: str
    candidates: tuple[RecognitionCandidate, ...]
    elapsed_s: float


class CaptchaOcrService:
    """Reuse OCR engines and expose a small single-image API."""

    def __init__(
        self,
        *,
        profile: str = "adaptive",
        expected_len: int = 4,
        confidence_margin: float = 5.0,
        confidence_min_engines: int = 2,
        confidence_min_families: int = 2,
        adaptive_full_limit: int | None = 455,
    ) -> None:
        self.profile = profile
        self.expected_len = expected_len
        self.confidence_margin = confidence_margin
        self.confidence_min_engines = confidence_min_engines
        self.confidence_min_families = confidence_min_families
        self.adaptive_full_limit = adaptive_full_limit
        self._engines = None

    def _engines_once(self):
        if self._engines is None:
            self._engines = create_ocr_engines()
        return self._engines

    def recognize(self, image_path: Path) -> RecognitionResult:
        if not image_path.exists():
            raise FileNotFoundError(f"image does not exist: {image_path}")

        start = time.perf_counter()
        run = run_image(
            image_path,
            self._engines_once(),
            self.profile,
            self.expected_len,
            self.confidence_margin,
            self.confidence_min_engines,
            self.confidence_min_families,
            self.adaptive_full_limit,
        )

        row_dicts = [row.__dict__ for row in run.rows]
        reranked = score_combinations(row_dicts, self.expected_len)
        candidates = tuple(RecognitionCandidate(row.text, row.score) for row in reranked[:5])

        if candidates:
            text = candidates[0].text
            score: float | None = candidates[0].score
        else:
            text = run.top_text
            score = run.top_score

        return RecognitionResult(
            text=text,
            score=score,
            mode=run.mode,
            candidates=candidates,
            elapsed_s=time.perf_counter() - start,
        )


def recognize_image(image_path: Path) -> RecognitionResult:
    return CaptchaOcrService().recognize(image_path)
