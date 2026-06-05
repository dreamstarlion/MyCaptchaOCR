#!/usr/bin/env python3
"""Evaluate reranked OCR candidates against an external truth CSV.

The truth file is intentionally external so labels are used only for evaluation,
not inside the OCR or reranking algorithms.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from ocr_project_env import PROJECT_ROOT


REPORTS_DIR = PROJECT_ROOT / "reports"


def read_truth(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        return {row["image_key"]: row["truth"] for row in rows if row.get("image_key") and row.get("truth")}


def read_candidates(path: Path) -> dict[str, list[dict[str, str]]]:
    by_key: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_key[row.get("image_key", "")].append(row)
    return by_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True, help="CSV with image_key,truth columns")
    parser.add_argument("--candidates", type=Path, default=REPORTS_DIR / "adaptive_ocr_v2_combinations.csv")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    truth = read_truth(args.truth)
    candidates = read_candidates(args.candidates)
    hit_counts = {1: 0, 3: 0, args.top_k: 0}

    print("| image | truth | rank | top candidates |")
    print("| --- | --- | ---: | --- |")
    for key, expected in sorted(truth.items()):
        rows = candidates.get(key, [])
        rank = next((idx + 1 for idx, row in enumerate(rows) if row.get("text") == expected), None)
        for cutoff in hit_counts:
            if rank is not None and rank <= cutoff:
                hit_counts[cutoff] += 1
        top = ", ".join(
            f"{row.get('text', '')}({row.get('score', '')},d{row.get('direct_hits', '')})"
            for row in rows[: args.top_k]
        )
        print(f"| {key} | {expected} | {rank or ''} | {top} |")

    total = len(truth)
    print()
    for cutoff in sorted(hit_counts):
        print(f"top{cutoff}: {hit_counts[cutoff]}/{total}")


if __name__ == "__main__":
    main()
