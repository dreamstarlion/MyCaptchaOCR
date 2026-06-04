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
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from ocr_project_env import PROJECT_ROOT


REPORTS_DIR = PROJECT_ROOT / "reports"


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


def row_weight(row: pd.Series) -> float:
    family = str(row["family"])
    variant = str(row["variant"])
    engine = str(row["engine"])
    weight = 1.0
    if engine == "ddddocr_beta":
        weight += 0.35
    elif engine in {"ddddocr_default", "ddddocr_old"}:
        weight += 0.15
    if "dominant_color" in family:
        weight += 1.8
    if "dark_suppress" in family:
        weight += 1.0
    if "hline" in family:
        weight += 0.5
    if "_crop_" in variant or "crop" in family:
        weight += 0.9
    if "strict" in family:
        weight -= 0.5
    return max(0.2, weight)


def write_csv(rows: list[ComboScore], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ComboScore.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def score_combinations(df: pd.DataFrame, expected_len: int) -> list[ComboScore]:
    results: list[ComboScore] = []
    df = df.fillna("")
    for key, group in df.groupby("image_key"):
        exact = group[group["text"].astype(str).str.len() == expected_len].copy()
        if exact.empty:
            continue

        pos_support = [defaultdict(lambda: {"weight": 0.0, "engines": set(), "families": set(), "variants": set()}) for _ in range(expected_len)]
        direct_rows: dict[str, list[pd.Series]] = defaultdict(list)
        for _idx, row in exact.iterrows():
            text = str(row["text"])
            weight = row_weight(row)
            direct_rows[text].append(row)
            for idx, char in enumerate(text):
                item = pos_support[idx][char]
                item["weight"] += weight
                item["engines"].add(row["engine"])
                item["families"].add(row["family"])
                item["variants"].add(row["variant"])

        position_options = []
        option_labels = []
        for idx, chars in enumerate(pos_support):
            scored = []
            for char, item in chars.items():
                score = (
                    min(float(item["weight"]), 40.0) * 0.35
                    + len(item["engines"]) * 1.5
                    + min(len(item["families"]), 8) * 0.8
                    + min(len(item["variants"]), 20) * 0.2
                )
                scored.append((char, score))
            scored.sort(key=lambda item: item[1], reverse=True)
            keep = scored[:4]
            position_options.append(keep)
            option_labels.append("/".join(f"{char}:{score:.1f}" for char, score in keep))

        for chars in itertools.product(*[[char for char, _score in opts] for opts in position_options]):
            text = "".join(chars)
            positional = sum(next(score for char, score in position_options[idx] if char == chars[idx]) for idx in range(expected_len))
            direct = direct_rows.get(text, [])
            direct_score = 0.0
            direct_example = ""
            direct_path = ""
            direct_engines = direct_families = direct_variants = 0
            if direct:
                engines = {row["engine"] for row in direct}
                families = {row["family"] for row in direct}
                variants = {row["variant"] for row in direct}
                direct_engines = len(engines)
                direct_families = len(families)
                direct_variants = len(variants)
                # Multiple preprocessing variants read by the same OCR model are
                # correlated evidence. Cap them harder unless another engine agrees.
                family_cap = 5 if direct_engines == 1 else 8
                variant_cap = 8 if direct_engines == 1 else 12
                direct_score = (
                    min(len(direct), 12) * 0.6
                    + direct_engines * 2.0
                    + min(direct_families, family_cap) * 1.3
                    + min(direct_variants, variant_cap) * 0.35
                )
                if any(
                    ("dominant_color" in str(row["family"]) or "dark_suppress" in str(row["family"]))
                    and "_crop_" in str(row["variant"])
                    for row in direct
                ):
                    direct_score += 8.0
                first = direct[0]
                direct_example = f"{first['engine']}:{first['variant']}"
                direct_path = str(first["path"])
            score = positional + direct_score - (0.0 if direct else 5.0)
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
    df = pd.read_csv(args.rows)
    rows = score_combinations(df, args.expected_len)
    write_csv(rows, REPORTS_DIR / "adaptive_ocr_v2_combinations.csv")
    write_markdown(rows, REPORTS_DIR / "adaptive_ocr_v2_summary.md", args.top_n)
    build_sheet(rows, REPORTS_DIR / "adaptive_ocr_v2_top_sheet.png", args.top_n)
    print(f"rows: {len(rows)}")
    print(f"wrote {REPORTS_DIR / 'adaptive_ocr_v2_summary.md'}")


if __name__ == "__main__":
    main()
