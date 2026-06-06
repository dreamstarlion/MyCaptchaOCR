#!/usr/bin/env python3
"""Adaptive OCR candidate generation without per-image ground truth.

This script is intentionally not keyed by filename or known labels. It creates
conservative and color-priority variants for each image, runs ddddocr engines,
and ranks text candidates by consensus and shape constraints.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import ddddocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_project_env import PROJECT_ROOT, configure_ocr_environment


configure_ocr_environment()


RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "adaptive_ocr"
REPORTS_DIR = PROJECT_ROOT / "reports"
RISKY_CONFUSABLE_PAIRS = {
    frozenset(pair)
    for pair in [
        ("授", "受"),
        ("国", "田"),
        ("言", "旨"),
        ("九", "力"),
        ("三", "参"),
        ("三", "多"),
        ("多", "参"),
    ]
}
FAMILY_PRIORITY = {
    "roi": 0,
    "clahe_sharp": 1,
    "line_inpaint_sharp": 2,
    "stroke_preserve": 3,
    "stroke_preserve_wide": 4,
    "dominant_color": 5,
    "dominant_color_mid": 6,
    "dominant_color_loose": 7,
    "hline": 8,
    "line_inpaint": 9,
    "dark_suppress": 10,
    "dark_suppress_strong": 11,
    "light": 12,
    "depoint": 13,
    "dominant_color_strict": 14,
}
PREFERRED_CROP_WINDOWS = {
    (0, 4),
    (0, 8),
    (0, 12),
    (6, 4),
    (6, 8),
    (12, 4),
    (20, 0),
    (20, 4),
}
PREFERRED_CROP_WINDOWS_ORDER = {window: idx for idx, window in enumerate(sorted(PREFERRED_CROP_WINDOWS))}


@dataclass
class Candidate:
    image_key: str
    variant: str
    family: str
    path: Path


@dataclass
class OCRRow:
    image_key: str
    variant: str
    family: str
    engine: str
    raw_text: str
    text: str
    path: str


@dataclass
class TextScore:
    image_key: str
    text: str
    score: float
    total_hits: int
    engine_count: int
    family_count: int
    variant_count: int
    length_penalty: float
    non_chinese_penalty: float
    best_examples: str


@dataclass
class ImageRun:
    candidates: list[Candidate]
    rows: list[OCRRow]
    mode: str
    top_text: str
    top_score: float | None
    preprocess_s: float
    ocr_s: float
    total_s: float


def encode_png(bgr: np.ndarray) -> bytes:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def normalize_text(text: str) -> str:
    text = (text or "").translate(str.maketrans({"叁": "三", "參": "参"}))
    return "".join(re.findall(r"[\u4e00-\u9fff]", text))


def variant_scale(variant: str) -> int:
    match = re.search(r"roi_(\d)x", variant)
    return int(match.group(1)) if match else 9


def crop_window(variant: str) -> tuple[int, int] | None:
    match = re.search(r"_crop_r(\d+)_b(\d+)", variant)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def root_family(family: str) -> str:
    if family.startswith("char_"):
        return family
    return family.removesuffix("_crop")


def candidate_priority(item: tuple[str, str, np.ndarray]) -> tuple[int, int, int, int, str]:
    family, variant, _image = item
    scale = variant_scale(variant)
    window = crop_window(variant)
    family_root = root_family(family)
    family_rank = FAMILY_PRIORITY.get(family_root, 99)
    if family.startswith("char_"):
        pos_match = re.search(r"_p([1-9]\d*)", family)
        pos = int(pos_match.group(1)) if pos_match else 9
        margin_match = re.search(r"_m(\d+)_p", variant)
        margin = int(margin_match.group(1)) if margin_match else 99
        return (1, scale, pos, margin, variant)
    if window is not None:
        if window not in PREFERRED_CROP_WINDOWS:
            return (4, scale, family_rank, 999, variant)
        return (2, scale, family_rank, PREFERRED_CROP_WINDOWS_ORDER[window], variant)
    return (0, scale, family_rank, 0, variant)


def read_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def write_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise ValueError(f"cannot encode image: {path}")
    buf.tofile(str(path))


def image_key(path: Path) -> str:
    match = re.search(r"(\d{6})", path.stem)
    return match.group(1) if match else path.stem


def find_inner_roi(bgr: np.ndarray) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bright_rows = (gray > 80).mean(axis=1)
    bright_cols = (gray > 80).mean(axis=0)
    rows = np.where(bright_rows > 0.45)[0]
    cols = np.where(bright_cols > 0.45)[0]
    if len(rows) and len(cols):
        x1, x2 = int(cols[0]), int(cols[-1]) + 1
        y1, y2 = int(rows[0]), int(rows[-1]) + 1
        if (x2 - x1) * (y2 - y1) >= gray.size * 0.25:
            return x1, y1, x2 - x1, y2 - y1
    h, w = gray.shape
    return 0, 0, w, h


def crop_and_upscale(bgr: np.ndarray, scale: int) -> np.ndarray:
    x, y, w, h = find_inner_roi(bgr)
    roi = bgr[y : y + h, x : x + w]
    roi = whiten_edge_borders(roi)
    return cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def whiten_edge_borders(bgr: np.ndarray, max_edge_ratio: float = 0.07) -> np.ndarray:
    """Remove screenshot border remnants without touching interior strokes."""
    out = bgr.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    max_x = max(1, int(w * max_edge_ratio))
    max_y = max(1, int(h * max_edge_ratio))

    def dark_col(idx: int) -> bool:
        col = gray[:, idx]
        return float((col < 95).mean()) > 0.35 or float((col < 45).mean()) > 0.08

    def dark_row(idx: int) -> bool:
        row = gray[idx, :]
        return float((row < 95).mean()) > 0.35 or float((row < 45).mean()) > 0.08

    left = 0
    while left < max_x and dark_col(left):
        left += 1
    right = 0
    while right < max_x and dark_col(w - 1 - right):
        right += 1
    top = 0
    while top < max_y and dark_row(top):
        top += 1
    bottom = 0
    while bottom < max_y and dark_row(h - 1 - bottom):
        bottom += 1

    if left:
        out[:, :left] = 255
    if right:
        out[:, w - right :] = 255
    if top:
        out[:top, :] = 255
    if bottom:
        out[h - bottom :, :] = 255
    return out


def lighten_background(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    bg = cv2.GaussianBlur(l, (0, 0), sigmaX=11, sigmaY=11)
    flat = cv2.divide(l, np.maximum(bg, 1), scale=245)
    flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX)
    out = cv2.cvtColor(cv2.merge([flat, a, b]), cv2.COLOR_LAB2BGR)
    return cv2.addWeighted(bgr, 0.68, out, 0.32, 0)


def clahe_sharpen(bgr: np.ndarray, clip_limit: float = 2.2) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    tile = (max(2, min(8, bgr.shape[1] // 48)), max(2, min(8, bgr.shape[0] // 24)))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile)
    l2 = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0, sigmaY=1.0)
    return cv2.addWeighted(enhanced, 1.55, blur, -0.55, 0)


def suppress_tiny_dots(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark = (gray < 165).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    dot_mask = np.zeros_like(dark)
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if area <= 80 and max(w, h) <= 14:
            dot_mask[labels == idx] = 255
    return cv2.inpaint(bgr, dot_mask, 3, cv2.INPAINT_TELEA)


def crop_margins(bgr: np.ndarray, right: int = 0, bottom: int = 0) -> np.ndarray:
    h, w = bgr.shape[:2]
    y2 = max(1, h - bottom)
    x2 = max(1, w - right)
    return bgr[:y2, :x2].copy()


def hue_distance(h: np.ndarray, center: int) -> np.ndarray:
    d = np.abs(h.astype(np.int16) - int(center))
    return np.minimum(d, 180 - d)


def dominant_hue(bgr: np.ndarray) -> int | None:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    mask = (s > 35) & (gray < 245)
    if int(mask.sum()) < 80:
        return None
    hist = np.bincount(h[mask].reshape(-1), weights=s[mask].reshape(-1), minlength=180)
    return int(np.argmax(hist))


def dominant_color_priority(bgr: np.ndarray, band: int = 18, sat_min: int = 25, close: int = 3) -> np.ndarray:
    center = dominant_hue(bgr)
    if center is None:
        return bgr.copy()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    mask = (hue_distance(h, center) <= band) & (s >= sat_min) & (gray < 248)
    mask = mask.astype(np.uint8) * 255
    if close:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    white = np.full_like(bgr, 255)
    return np.where(mask[:, :, None] > 0, bgr, white)


def dominant_color_stroke_preserve(bgr: np.ndarray, band: int = 24, sat_min: int = 18) -> np.ndarray:
    center = dominant_hue(bgr)
    if center is None:
        return clahe_sharpen(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    core = (hue_distance(h, center) <= band) & (s >= sat_min) & (gray < 250)
    dark_colored = (hue_distance(h, center) <= band + 8) & (gray < 175) & (s >= max(8, sat_min - 10))
    mask = (core | dark_colored).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    edge_band = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    keep = (mask > 0) | ((edge_band > 0) & (gray < 235))
    out = np.full_like(bgr, 255)
    out[keep] = bgr[keep]
    return clahe_sharpen(out, clip_limit=1.8)


def white_low_saturation_dark(bgr: np.ndarray, gray_thresh: int = 110, sat_thresh: int = 115) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray < gray_thresh) & (hsv[:, :, 1] < sat_thresh)
    out = bgr.copy()
    out[mask] = (255, 255, 255)
    return out


def inpaint_dark_horizontal(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    candidate = ((gray < 110) & (hsv[:, :, 1] < 130)).astype(np.uint8) * 255
    edges = cv2.Canny(candidate, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=70, maxLineGap=12)
    mask = np.zeros_like(gray)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            if angle < 12:
                cv2.line(mask, (x1, y1), (x2, y2), 255, 5)
    mask = cv2.bitwise_and(mask, candidate)
    return cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)


def inpaint_long_dark_lines(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    candidate = (((gray < 115) & (hsv[:, :, 1] < 145)) | (gray < 55)).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2)))
    edges = cv2.Canny(candidate, 40, 130)
    min_len = max(18, int(w * 0.28))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=18, minLineLength=min_len, maxLineGap=max(6, int(w * 0.06)))
    mask = np.zeros_like(gray)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            length = math.hypot(float(x2 - x1), float(y2 - y1))
            if length < min_len:
                continue
            thickness = max(2, int(round(min(h, w) * 0.018)))
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
    mask = cv2.bitwise_and(cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))), candidate)
    if int(mask.sum()) == 0:
        return bgr.copy()
    return cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)


def foreground_mask_for_windows(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = (((hsv[:, :, 1] > 22) & (gray < 248)) | (gray < 165)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mask


def character_crops(bgr: np.ndarray, expected_len: int, margin_ratio: float = 0.06) -> list[np.ndarray]:
    mask = foreground_mask_for_windows(bgr)
    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        x1, x2, y1, y2 = 0, w, 0, h
    else:
        x1, x2 = max(0, int(xs.min()) - 2), min(w, int(xs.max()) + 3)
        y1, y2 = max(0, int(ys.min()) - 2), min(h, int(ys.max()) + 3)
    width = max(1, x2 - x1)
    crops = []
    for idx in range(expected_len):
        a = int(x1 + width * idx / expected_len)
        b = int(x1 + width * (idx + 1) / expected_len)
        margin = max(2, int(width * margin_ratio))
        aa = max(0, a - margin)
        bb = min(w, b + margin)
        crop = bgr[y1:y2, aa:bb].copy()
        crop = cv2.copyMakeBorder(crop, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        crops.append(crop)
    return crops


def build_variants(raw: np.ndarray, include_crops: bool = True, expected_len: int = 4) -> list[tuple[str, str, np.ndarray]]:
    variants: list[tuple[str, str, np.ndarray]] = []
    for scale in [1, 2, 3, 4, 5]:
        roi = crop_and_upscale(raw, scale)
        clean_lines = inpaint_long_dark_lines(roi)
        sharp = clahe_sharpen(roi)
        base = [
            ("roi", f"roi_{scale}x", roi),
            ("light", f"roi_{scale}x_light", lighten_background(roi)),
            ("clahe_sharp", f"roi_{scale}x_clahe_sharp", sharp),
            ("depoint", f"roi_{scale}x_depoint", suppress_tiny_dots(roi)),
            ("hline", f"roi_{scale}x_hline", inpaint_dark_horizontal(roi)),
            ("line_inpaint", f"roi_{scale}x_line_inpaint", clean_lines),
            ("line_inpaint_sharp", f"roi_{scale}x_line_inpaint_sharp", clahe_sharpen(clean_lines)),
            ("dark_suppress", f"roi_{scale}x_dark_suppress", white_low_saturation_dark(roi)),
            ("dark_suppress_strong", f"roi_{scale}x_dark_suppress_strong", white_low_saturation_dark(roi, gray_thresh=125, sat_thresh=75)),
            ("dominant_color", f"roi_{scale}x_dominant_color", dominant_color_priority(roi, band=18, sat_min=25)),
            ("dominant_color_mid", f"roi_{scale}x_dominant_color_mid", dominant_color_priority(roi, band=24, sat_min=25)),
            ("dominant_color_loose", f"roi_{scale}x_dominant_color_loose", dominant_color_priority(roi, band=28, sat_min=18)),
            ("dominant_color_strict", f"roi_{scale}x_dominant_color_strict", dominant_color_priority(roi, band=12, sat_min=45)),
            ("stroke_preserve", f"roi_{scale}x_stroke_preserve", dominant_color_stroke_preserve(roi, band=24, sat_min=18)),
            ("stroke_preserve_wide", f"roi_{scale}x_stroke_preserve_wide", dominant_color_stroke_preserve(roi, band=32, sat_min=12)),
        ]
        variants.extend(base)
        for family, name, image in base:
            if include_crops and family in {"roi", "dark_suppress", "dark_suppress_strong", "dominant_color", "dominant_color_mid", "dominant_color_loose", "dominant_color_strict", "hline", "line_inpaint", "stroke_preserve", "stroke_preserve_wide", "clahe_sharp"}:
                for right in [0, 6, 12, 20]:
                    for bottom in [0, 4, 8, 12]:
                        if right == 0 and bottom == 0:
                            continue
                        variants.append((f"{family}_crop", f"{name}_crop_r{right}_b{bottom}", crop_margins(image, right, bottom)))
        if include_crops and scale in {3, 4}:
            for family, name, image in [
                ("char_roi", f"roi_{scale}x_char_roi", roi),
                ("char_dominant", f"roi_{scale}x_char_dominant", dominant_color_priority(roi, band=24, sat_min=20)),
                ("char_strict", f"roi_{scale}x_char_strict", dominant_color_priority(roi, band=12, sat_min=45)),
                ("char_stroke", f"roi_{scale}x_char_stroke", dominant_color_stroke_preserve(roi, band=28, sat_min=14)),
                ("char_sharp", f"roi_{scale}x_char_sharp", sharp),
            ]:
                for margin in [0.02, 0.06, 0.12, 0.18]:
                    margin_label = int(round(margin * 100))
                    for idx, crop in enumerate(character_crops(image, expected_len, margin_ratio=margin), 1):
                        variants.append((f"{family}_p{idx}", f"{name}_m{margin_label}_p{idx}", crop))
    return variants


def select_variants(
    variants: list[tuple[str, str, np.ndarray]],
    max_candidates: int | None,
) -> list[tuple[str, str, np.ndarray]]:
    if max_candidates is None or len(variants) <= max_candidates:
        return variants
    indexed = list(enumerate(variants))
    selected = sorted(indexed, key=lambda item: candidate_priority(item[1]))[:max_candidates]
    return [item for _idx, item in sorted(selected, key=lambda item: item[0])]


def save_candidates(
    path: Path,
    include_crops: bool = True,
    expected_len: int = 4,
    max_candidates: int | None = None,
) -> list[Candidate]:
    key = image_key(path)
    out_dir = OUT_DIR / key
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = read_bgr(path)
    candidates = []
    seen = set()
    variants = build_variants(raw, include_crops=include_crops, expected_len=expected_len)
    for family, variant, image in select_variants(variants, max_candidates=max_candidates):
        if variant in seen:
            continue
        seen.add(variant)
        out = out_dir / f"{variant}.png"
        write_image(out, image)
        candidates.append(Candidate(key, variant, family, out))
    return candidates


def image_adjusted_limit(path: Path, limit: int | None) -> int | None:
    if limit is None:
        return None
    with Image.open(path) as img:
        w, h = img.size
    if w >= 220 or h >= 85 or w * h >= 22_000:
        return min(limit, 455)
    return limit


def create_ocr_engines() -> dict[str, ddddocr.DdddOcr]:
    return {
        "ddddocr_default": ddddocr.DdddOcr(show_ad=False),
        "ddddocr_beta": ddddocr.DdddOcr(show_ad=False, beta=True),
        "ddddocr_old": ddddocr.DdddOcr(show_ad=False, old=True),
    }


def run_ocr_with_engines(candidates: list[Candidate], engines: dict[str, ddddocr.DdddOcr]) -> list[OCRRow]:
    rows = []
    for candidate in candidates:
        data = candidate.path.read_bytes()
        for engine_name, engine in engines.items():
            raw_text = str(engine.classification(data) or "")
            rows.append(
                OCRRow(
                    image_key=candidate.image_key,
                    variant=candidate.variant,
                    family=candidate.family,
                    engine=engine_name,
                    raw_text=raw_text,
                    text=normalize_text(raw_text),
                    path=candidate.path.relative_to(PROJECT_ROOT).as_posix(),
                )
            )
    return rows


def run_ocr(candidates: list[Candidate]) -> list[OCRRow]:
    return run_ocr_with_engines(candidates, create_ocr_engines())


def score_texts(rows: list[OCRRow], expected_len: int) -> list[TextScore]:
    by_image_text: dict[tuple[str, str], list[OCRRow]] = defaultdict(list)
    for row in rows:
        if row.text:
            by_image_text[(row.image_key, row.text)].append(row)

    scores = []
    for (key, text), group in by_image_text.items():
        engines = {row.engine for row in group}
        families = {row.family for row in group}
        variants = {row.variant for row in group}
        length_penalty = abs(len(text) - expected_len) * 7.0
        non_chinese_penalty = 0.0
        exact_bonus = 8.0 if len(text) == expected_len else 0.0
        # Avoid allowing hundreds of near-duplicate crop variants to dominate.
        hit_score = min(len(group), 12) * 0.35
        score = (
            exact_bonus
            + hit_score
            + len(engines) * 2.5
            + len(families) * 1.2
            + min(len(variants), 10) * 0.45
            - length_penalty
            - non_chinese_penalty
        )
        examples = []
        for row in group[:5]:
            examples.append(f"{row.engine}:{row.variant}")
        scores.append(
            TextScore(
                image_key=key,
                text=text,
                score=round(score, 3),
                total_hits=len(group),
                engine_count=len(engines),
                family_count=len(families),
                variant_count=len(variants),
                length_penalty=length_penalty,
                non_chinese_penalty=non_chinese_penalty,
                best_examples="; ".join(examples),
            )
        )
    return sorted(scores, key=lambda item: (item.image_key, -item.score, item.text))


def differs_by_risky_confusable(left: str, right: str) -> bool:
    if len(left) != len(right):
        return False
    diffs = [(a, b) for a, b in zip(left, right) if a != b]
    return len(diffs) == 1 and frozenset(diffs[0]) in RISKY_CONFUSABLE_PAIRS


def has_confusable_conflict(scores: list[TextScore], expected_len: int, max_gap: float = 28.0) -> bool:
    if not scores:
        return False
    top = scores[0]
    if len(top.text) != expected_len:
        return True
    for score in scores[1:20]:
        if len(score.text) != expected_len:
            continue
        if top.score - score.score > max_gap:
            break
        if differs_by_risky_confusable(top.text, score.text):
            return True
    return False


def is_confident(scores: list[TextScore], expected_len: int, min_margin: float, min_engines: int, min_families: int) -> bool:
    if not scores:
        return False
    top = scores[0]
    if len(top.text) != expected_len:
        return False
    if has_confusable_conflict(scores, expected_len):
        return False
    second_score = scores[1].score if len(scores) > 1 else float("-inf")
    margin = top.score - second_score
    if margin < min_margin:
        return False
    if top.engine_count >= min_engines and top.family_count >= min_families and margin >= min_margin:
        return True
    if top.score >= 32.0 and top.engine_count >= 3 and top.family_count >= 6 and top.total_hits >= 35:
        return True
    if top.score >= 30.0 and top.engine_count >= 3 and top.family_count >= 5:
        return True
    if top.score >= 29.5 and top.family_count >= 8 and top.total_hits >= 30:
        return True
    return False


def run_image(
    path: Path,
    engines: dict[str, ddddocr.DdddOcr],
    profile: str,
    expected_len: int,
    confidence_margin: float,
    confidence_min_engines: int,
    confidence_min_families: int,
    adaptive_full_limit: int | None,
) -> ImageRun:
    start = time.perf_counter()
    preprocess_s = 0.0
    ocr_s = 0.0

    if profile == "full":
        preprocess_start = time.perf_counter()
        candidates = save_candidates(path, include_crops=True, expected_len=expected_len)
        preprocess_s += time.perf_counter() - preprocess_start

        ocr_start = time.perf_counter()
        rows = run_ocr_with_engines(candidates, engines)
        ocr_s += time.perf_counter() - ocr_start
        scores = score_texts(rows, expected_len)
        top = scores[0] if scores else None
        return ImageRun(candidates, rows, "full", top.text if top else "", top.score if top else None, preprocess_s, ocr_s, time.perf_counter() - start)

    preprocess_start = time.perf_counter()
    fast_candidates = save_candidates(path, include_crops=False, expected_len=expected_len)
    preprocess_s += time.perf_counter() - preprocess_start

    ocr_start = time.perf_counter()
    fast_rows = run_ocr_with_engines(fast_candidates, engines)
    ocr_s += time.perf_counter() - ocr_start
    fast_scores = score_texts(fast_rows, expected_len)

    if profile == "fast" or is_confident(fast_scores, expected_len, confidence_margin, confidence_min_engines, confidence_min_families):
        top = fast_scores[0] if fast_scores else None
        mode = "fast" if profile == "fast" else "adaptive-fast"
        return ImageRun(fast_candidates, fast_rows, mode, top.text if top else "", top.score if top else None, preprocess_s, ocr_s, time.perf_counter() - start)

    preprocess_start = time.perf_counter()
    adjusted_full_limit = image_adjusted_limit(path, adaptive_full_limit)
    full_candidates = save_candidates(
        path,
        include_crops=True,
        expected_len=expected_len,
        max_candidates=adjusted_full_limit,
    )
    preprocess_s += time.perf_counter() - preprocess_start
    seen_variants = {candidate.variant for candidate in fast_candidates}
    extra_candidates = [candidate for candidate in full_candidates if candidate.variant not in seen_variants]

    ocr_start = time.perf_counter()
    extra_rows = run_ocr_with_engines(extra_candidates, engines)
    ocr_s += time.perf_counter() - ocr_start

    rows = fast_rows + extra_rows
    scores = score_texts(rows, expected_len)
    top = scores[0] if scores else None
    return ImageRun(full_candidates, rows, "adaptive-balanced", top.text if top else "", top.score if top else None, preprocess_s, ocr_s, time.perf_counter() - start)


def write_csv(rows: list[object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].__dataclass_fields__)  # type: ignore[attr-defined]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def render_image(path: Path, width: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    ratio = width / img.width
    return img.resize((width, max(1, int(img.height * ratio))), Image.Resampling.BICUBIC)


def build_top_sheet(scores: list[TextScore], rows: list[OCRRow], output: Path, top_n: int = 5) -> None:
    best_row_for = {}
    for row in rows:
        key = (row.image_key, row.text)
        if row.text and key not in best_row_for:
            best_row_for[key] = row

    selected = []
    by_key = defaultdict(list)
    for score in scores:
        by_key[score.image_key].append(score)
    for key, key_scores in sorted(by_key.items()):
        for score in key_scores[:top_n]:
            row = best_row_for.get((score.image_key, score.text))
            if row:
                selected.append((score, row))
    if not selected:
        return

    font = ImageFont.load_default()
    cell_w = 360
    label_h = 44
    gap = 18
    rendered = []
    for score, row in selected:
        img = render_image(PROJECT_ROOT / row.path, cell_w)
        rendered.append((score, row, img))
    sheet = Image.new("RGB", (cell_w + 32, sum(img.height + label_h + gap for _score, _row, img in rendered) + 20), "white")
    draw = ImageDraw.Draw(sheet)
    y = 10
    for score, row, img in rendered:
        draw.text((16, y), f"{score.image_key} {score.text} score={score.score}", fill=(0, 0, 0), font=font)
        draw.text((16, y + 16), f"{row.engine} {row.variant}"[:100], fill=(0, 0, 0), font=font)
        sheet.paste(img, (16, y + label_h))
        y += img.height + label_h + gap
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def write_markdown(scores: list[TextScore], output: Path) -> None:
    by_key = defaultdict(list)
    for score in scores:
        by_key[score.image_key].append(score)
    lines = [
        "# Adaptive OCR Pipeline",
        "",
        "No filename-specific rules or known labels are used. Scores come from OCR consensus, expected length, and diversity across preprocessing families.",
        "",
        "| image | top candidates |",
        "| --- | --- |",
    ]
    for key, key_scores in sorted(by_key.items()):
        top = ", ".join(f"`{s.text}` ({s.score})" for s in key_scores[:5])
        lines.append(f"| {key} | {top} |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--pattern", default="sample-[0-9]*.png")
    parser.add_argument("--expected-len", type=int, default=4)
    parser.add_argument("--profile", choices=["adaptive", "fast", "full"], default="adaptive")
    parser.add_argument("--confidence-margin", type=float, default=5.0)
    parser.add_argument("--confidence-min-engines", type=int, default=2)
    parser.add_argument("--confidence-min-families", type=int, default=2)
    parser.add_argument(
        "--adaptive-full-limit",
        type=int,
        default=455,
        help="Maximum candidates used by adaptive fallback. Use 0 for unrestricted full fallback.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob(args.pattern))
    if not paths:
        raise SystemExit(f"no images matched {args.input_dir / args.pattern}")

    engine_start = time.perf_counter()
    engines = create_ocr_engines()
    engine_init_s = time.perf_counter() - engine_start
    adaptive_full_limit = None if args.adaptive_full_limit <= 0 else args.adaptive_full_limit

    candidates = []
    rows = []
    runs = []
    for path in paths:
        run = run_image(
            path,
            engines,
            args.profile,
            args.expected_len,
            args.confidence_margin,
            args.confidence_min_engines,
            args.confidence_min_families,
            adaptive_full_limit,
        )
        runs.append(run)
        candidates.extend(run.candidates)
        rows.extend(run.rows)
        top = f" top={run.top_text!r} score={run.top_score}" if run.top_text else ""
        print(
            f"{image_key(path)} mode={run.mode} candidates={len(run.candidates)} ocr_rows={len(run.rows)}"
            f"{top} preprocess_s={run.preprocess_s:.3f} ocr_s={run.ocr_s:.3f} total_s={run.total_s:.3f}"
        )
    scores = score_texts(rows, args.expected_len)

    write_csv(rows, REPORTS_DIR / "adaptive_ocr_rows.csv")
    write_csv(scores, REPORTS_DIR / "adaptive_ocr_scores.csv")
    write_markdown(scores, REPORTS_DIR / "adaptive_ocr_summary.md")
    build_top_sheet(scores, rows, REPORTS_DIR / "adaptive_ocr_top_sheet.png")
    print(f"images: {len(paths)}")
    print(f"profile: {args.profile}")
    print(f"engine_init_s: {engine_init_s:.3f}")
    print(f"candidates: {len(candidates)}")
    print(f"ocr rows: {len(rows)}")
    print(f"avg_total_s_per_image: {sum(run.total_s for run in runs) / len(runs):.3f}")
    print(f"avg_ocr_s_per_image: {sum(run.ocr_s for run in runs) / len(runs):.3f}")
    print(f"wrote {REPORTS_DIR / 'adaptive_ocr_summary.md'}")


if __name__ == "__main__":
    main()
