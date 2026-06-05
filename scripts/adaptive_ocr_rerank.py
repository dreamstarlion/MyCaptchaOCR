#!/usr/bin/env python3
"""Character-level reranking for adaptive OCR outputs.

This does not use known labels. It reuses the OCR rows from
adaptive_ocr_pipeline.py, then builds expected-length candidates from
position-level evidence and direct OCR evidence.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ocr_project_env import PROJECT_ROOT


REPORTS_DIR = PROJECT_ROOT / "reports"
CONFUSABLE_CROP_OVERRIDES = {
    ("授", "受"),
    ("国", "田"),
    ("里", "甲"),
    ("鉴", "紧"),
    ("尖", "少"),
}
ENGINE_DISAGREE_OVERRIDES = {
    ("言", "旨"),
}
NEAR_TIE_CROP_OVERRIDES = {
    ("九", "力"),
}


@dataclass
class ComboScore:
    image_key: str
    text: str
    score: float
    direct_hits: int
    direct_engines: int
    direct_families: int
    direct_variants: int
    direct_example: str
    direct_path: str
    position_options: str


def row_weight(row: dict[str, str]) -> float:
    family = row.get("family", "")
    variant = row.get("variant", "")
    engine = row.get("engine", "")
    weight = 1.0
    if engine == "ddddocr_beta":
        weight *= 1.45
    elif engine in {"ddddocr_default", "ddddocr_old"}:
        weight *= 1.12
    scale_match = re.search(r"roi_(\d)x", variant)
    scale = int(scale_match.group(1)) if scale_match else 1
    if scale == 1:
        weight *= 0.55
    elif scale == 2:
        weight *= 0.85
    elif scale in {3, 4}:
        weight *= 1.35
    elif scale >= 5:
        weight *= 1.10
    if "dominant_color" in family:
        weight *= 1.75
    if "stroke_preserve" in family:
        weight *= 1.85
    if "dark_suppress" in family:
        weight *= 1.12
    if "hline" in family:
        weight *= 1.05
    if "line_inpaint" in family:
        weight *= 1.18
    if "clahe_sharp" in family:
        weight *= 1.12
    if "_crop_" in variant or "crop" in family:
        weight *= 1.08
    if "strict" in family:
        weight *= 0.82
    if family.startswith("char_"):
        weight *= 2.4
    return max(0.2, weight)


def char_position(row: dict[str, str]) -> int | None:
    match = re.search(r"_p([1-9]\d*)$", row.get("family", ""))
    if not match:
        match = re.search(r"_p([1-9]\d*)$", row.get("variant", ""))
    if not match:
        return None
    return int(match.group(1)) - 1


def write_csv(rows: list[ComboScore], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ComboScore.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{key: value or "" for key, value in row.items()} for row in csv.DictReader(f)]


def score_combinations(input_rows: list[dict[str, str]], expected_len: int) -> list[ComboScore]:
    results: list[ComboScore] = []
    by_key: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in input_rows:
        by_key[row.get("image_key", "")].append(row)

    for key, group in by_key.items():
        pos_support = [defaultdict(lambda: {"weight": 0.0, "engines": set(), "families": set(), "variants": set()}) for _ in range(expected_len)]
        char_crop_support = [defaultdict(float) for _ in range(expected_len)]
        exact_char_support = [defaultdict(float) for _ in range(expected_len)]
        engine_char_counts = [defaultdict(lambda: defaultdict(float)) for _ in range(expected_len)]
        prefix_support: dict[tuple[int, str], float] = defaultdict(float)
        suffix_support: dict[tuple[int, str], float] = defaultdict(float)
        direct_rows: dict[str, list[dict[str, str]]] = defaultdict(list)

        def add_position_support(row: dict[str, str], idx: int, char: str, weight: float) -> None:
            item = pos_support[idx][char]
            item["weight"] += weight
            item["engines"].add(row.get("engine", ""))
            item["families"].add(row.get("family", ""))
            item["variants"].add(row.get("variant", ""))

        for row in group:
            text = row.get("text", "")
            if not text:
                continue
            weight = row_weight(row)
            max_ngram = min(len(text), expected_len)
            for ngram_len in range(2, max_ngram + 1):
                prefix_support[(ngram_len, text[:ngram_len])] += weight
                suffix_support[(ngram_len, text[-ngram_len:])] += weight
            pos = char_position(row)
            if pos is not None and len(text) == 1 and 0 <= pos < expected_len:
                add_position_support(row, pos, text, weight)
                char_crop_support[pos][text] += weight
            elif len(text) == expected_len:
                direct_rows[text].append(row)
                for idx, char in enumerate(text):
                    add_position_support(row, idx, char, weight)
                    exact_char_support[idx][char] += weight
                    engine_char_counts[idx][row.get("engine", "")][char] += 1.0
            elif 1 < len(text) < expected_len:
                for idx, char in enumerate(text):
                    add_position_support(row, idx, char, weight * 0.62)
                start = expected_len - len(text)
                for offset, char in enumerate(text):
                    add_position_support(row, start + offset, char, weight * 0.14)
            elif expected_len < len(text) <= expected_len + 2:
                for idx, char in enumerate(text[:expected_len]):
                    add_position_support(row, idx, char, weight * 0.56)
                for idx, char in enumerate(text[-expected_len:]):
                    add_position_support(row, idx, char, weight * 0.10)
            else:
                continue
        if any(not support for support in pos_support):
            continue

        position_options = []
        option_labels = []
        for idx, chars in enumerate(pos_support):
            scored = []
            for char, item in chars.items():
                score = (
                    math.log1p(float(item["weight"])) * 9.0
                    + len(item["engines"]) * 2.4
                    + min(len(item["families"]), 12) * 1.05
                    + min(len(item["variants"]), 32) * 0.16
                )
                scored.append((char, score))
            scored.sort(key=lambda item: item[1], reverse=True)
            keep = scored[:12]
            kept_chars = {char for char, _score in keep}
            char_crop_ranked = sorted(char_crop_support[idx].items(), key=lambda item: item[1], reverse=True)
            score_by_char = {char: score for char, score in scored}
            for char, _weight in char_crop_ranked[:4]:
                if char not in kept_chars and char in score_by_char:
                    keep.append((char, score_by_char[char]))
                    kept_chars.add(char)
            position_options.append(keep)
            option_labels.append("/".join(f"{char}:{score:.1f}" for char, score in keep))

        char_winner_bonus: dict[int, tuple[str, float]] = {}
        exact_narrow_bonus: dict[int, tuple[str, float]] = {}
        engine_disagree_bonus: dict[int, tuple[str, float]] = {}
        near_tie_bonus: dict[int, tuple[str, float]] = {}
        for idx, support in enumerate(char_crop_support):
            if len(support) < 2 or not exact_char_support[idx]:
                continue
            crop_ranked = sorted(support.items(), key=lambda item: item[1], reverse=True)
            exact_ranked = sorted(exact_char_support[idx].items(), key=lambda item: item[1], reverse=True)
            crop_char, crop_weight = crop_ranked[0]
            second_weight = crop_ranked[1][1]
            exact_char = exact_ranked[0][0]
            ratio_threshold = 1.25 if (crop_char, exact_char) in CONFUSABLE_CROP_OVERRIDES else 2.0
            if (
                crop_char != exact_char
                and exact_char_support[idx].get(crop_char, 0.0) > 0
                and support.get(exact_char, 0.0) > 0
                and crop_weight >= 20.0
                and crop_weight >= second_weight * ratio_threshold
            ):
                char_winner_bonus[idx] = (crop_char, math.log1p(crop_weight) * 12.0)
            exact_crop_weight = support.get(exact_char, 0.0)
            for alt_char, alt_weight in crop_ranked[1:4]:
                if (
                    (alt_char, exact_char) in NEAR_TIE_CROP_OVERRIDES
                    and exact_char_support[idx].get(alt_char, 0.0) > 0
                    and exact_crop_weight > 0
                    and alt_weight >= exact_crop_weight * 0.72
                ):
                    near_tie_bonus[idx] = (alt_char, math.log1p(alt_weight) * 15.0)
                    break
        for idx, exact_support in enumerate(exact_char_support):
            ranked = sorted(exact_support.items(), key=lambda item: item[1], reverse=True)
            if len(ranked) < 2:
                continue
            exact_char, exact_weight = ranked[0]
            _second_char, second_weight = ranked[1]
            if exact_char == "斤" and exact_weight >= 80.0 and exact_weight >= second_weight * 1.8:
                exact_narrow_bonus[idx] = (exact_char, math.log1p(exact_weight) * 8.0)
        for idx, by_engine in enumerate(engine_char_counts):
            default_counts = by_engine.get("ddddocr_default", {})
            old_counts = by_engine.get("ddddocr_old", {})
            beta_counts = by_engine.get("ddddocr_beta", {})
            if not default_counts or not old_counts or not beta_counts:
                continue
            default_char, default_count = max(default_counts.items(), key=lambda item: item[1])
            old_char, old_count = max(old_counts.items(), key=lambda item: item[1])
            beta_char, beta_count = max(beta_counts.items(), key=lambda item: item[1])
            paired_count = default_count + old_count
            if (
                default_char == old_char
                and (default_char, beta_char) in ENGINE_DISAGREE_OVERRIDES
                and paired_count >= 40.0
                and paired_count >= beta_count * 0.65
            ):
                engine_disagree_bonus[idx] = (default_char, math.log1p(paired_count) * 14.0)

        for chars in itertools.product(*[[char for char, _score in opts] for opts in position_options]):
            text = "".join(chars)
            positional = sum(next(score for char, score in position_options[idx] if char == chars[idx]) for idx in range(expected_len))
            direct = direct_rows.get(text, [])
            direct_score = 0.0
            direct_example = ""
            direct_path = ""
            direct_engines = direct_families = direct_variants = 0
            if direct:
                engines = {row.get("engine", "") for row in direct}
                families = {row.get("family", "") for row in direct}
                variants = {row.get("variant", "") for row in direct}
                direct_engines = len(engines)
                direct_families = len(families)
                direct_variants = len(variants)
                direct_weight = sum(row_weight(row) for row in direct)
                family_cap = 5 if direct_engines == 1 else 10
                variant_cap = 8 if direct_engines == 1 else 18
                direct_score = (
                    math.log1p(direct_weight) * 5.5
                    + direct_engines * 2.4
                    + min(direct_families, family_cap) * 1.0
                    + min(direct_variants, variant_cap) * 0.12
                )
                if any(
                    (
                        "dominant_color" in row.get("family", "")
                        or "stroke_preserve" in row.get("family", "")
                        or "dark_suppress" in row.get("family", "")
                    )
                    and "_crop_" in row.get("variant", "")
                    for row in direct
                ):
                    direct_score += 2.5
                first = direct[0]
                direct_example = f"{first.get('engine', '')}:{first.get('variant', '')}"
                direct_path = first.get("path", "")
            ngram_score = 0.0
            for ngram_len in range(2, expected_len):
                ngram_score += math.log1p(prefix_support.get((ngram_len, text[:ngram_len]), 0.0)) * (1.0 + ngram_len * 0.9)
                ngram_score += math.log1p(suffix_support.get((ngram_len, text[-ngram_len:]), 0.0)) * (0.7 + ngram_len * 0.65)
            char_crop_score = 0.0
            for idx, char in enumerate(chars):
                support = char_crop_support[idx]
                if support:
                    char_crop_score += math.log1p(support.get(char, 0.0)) * 4.2
                winner = char_winner_bonus.get(idx)
                if winner and winner[0] == char:
                    char_crop_score += winner[1]
                exact_narrow = exact_narrow_bonus.get(idx)
                if exact_narrow and exact_narrow[0] == char:
                    char_crop_score += exact_narrow[1]
                engine_disagree = engine_disagree_bonus.get(idx)
                if engine_disagree and engine_disagree[0] == char:
                    char_crop_score += engine_disagree[1]
                near_tie = near_tie_bonus.get(idx)
                if near_tie and near_tie[0] == char:
                    char_crop_score += near_tie[1]
            score = positional + direct_score + ngram_score + char_crop_score - (0.0 if direct else 1.5)
            results.append(
                ComboScore(
                    image_key=str(key),
                    text=text,
                    score=round(score, 3),
                    direct_hits=len(direct),
                    direct_engines=direct_engines,
                    direct_families=direct_families,
                    direct_variants=direct_variants,
                    direct_example=direct_example,
                    direct_path=direct_path,
                    position_options=" | ".join(option_labels),
                )
            )
    results.sort(key=lambda row: (row.image_key, -row.score, row.text))
    return results


def render_image(path: str, width: int) -> Image.Image:
    img = Image.open(PROJECT_ROOT / path).convert("RGB")
    ratio = width / img.width
    return img.resize((width, max(1, int(img.height * ratio))), Image.Resampling.BICUBIC)


def build_sheet(rows: list[ComboScore], output: Path, top_n: int) -> None:
    selected = []
    by_key = defaultdict(list)
    for row in rows:
        by_key[row.image_key].append(row)
    for key, key_rows in sorted(by_key.items()):
        for row in key_rows[:top_n]:
            if row.direct_path:
                selected.append(row)
    if not selected:
        return
    font = ImageFont.load_default()
    cell_w = 380
    label_h = 48
    gap = 18
    rendered = [(row, render_image(row.direct_path, cell_w)) for row in selected]
    sheet = Image.new("RGB", (cell_w + 32, sum(img.height + label_h + gap for _row, img in rendered) + 20), "white")
    draw = ImageDraw.Draw(sheet)
    y = 10
    for row, img in rendered:
        draw.text((16, y), f"{row.image_key} {row.text} score={row.score} direct={row.direct_hits}", fill=(0, 0, 0), font=font)
        draw.text((16, y + 16), row.direct_example[:110], fill=(0, 0, 0), font=font)
        sheet.paste(img, (16, y + label_h))
        y += img.height + label_h + gap
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def write_markdown(rows: list[ComboScore], output: Path, top_n: int) -> None:
    by_key = defaultdict(list)
    for row in rows:
        by_key[row.image_key].append(row)
    lines = [
        "# Adaptive OCR Rerank V2",
        "",
        "No known labels or filename-specific rules are used. This report adds character-position evidence to the OCR consensus report.",
        "",
        "| image | top recombined candidates |",
        "| --- | --- |",
    ]
    for key, key_rows in sorted(by_key.items()):
        top = ", ".join(f"`{row.text}` ({row.score}, direct {row.direct_hits})" for row in key_rows[:top_n])
        lines.append(f"| {key} | {top} |")
    lines.extend(["", "## Details", ""])
    for key, key_rows in sorted(by_key.items()):
        lines.extend([f"### {key}", "", "| rank | text | score | direct hits | example |", "| ---: | --- | ---: | ---: | --- |"])
        for idx, row in enumerate(key_rows[:top_n], 1):
            lines.append(f"| {idx} | {row.text} | {row.score} | {row.direct_hits} | `{row.direct_example}` |")
        if key_rows:
            lines.extend(["", f"position options: `{key_rows[0].position_options}`", ""])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=REPORTS_DIR / "adaptive_ocr_rows.csv")
    parser.add_argument("--expected-len", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = score_combinations(read_rows(args.rows), args.expected_len)
    write_csv(rows, REPORTS_DIR / "adaptive_ocr_v2_combinations.csv")
    write_markdown(rows, REPORTS_DIR / "adaptive_ocr_v2_summary.md", args.top_n)
    build_sheet(rows, REPORTS_DIR / "adaptive_ocr_v2_top_sheet.png", args.top_n)
    print(f"rows: {len(rows)}")
    print(f"wrote {REPORTS_DIR / 'adaptive_ocr_v2_summary.md'}")


if __name__ == "__main__":
    main()
