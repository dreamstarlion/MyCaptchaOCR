#!/usr/bin/env python3
"""Project-local environment setup for OCR scripts."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OCR_CACHE_ROOT = PROJECT_ROOT / ".ocr-cache"

# OCR 性能调优（按 CPU 物理核数设定；本机 i5-10400 = 6 物理核）。
# 原理：一张图的数百个预处理变体推理相互独立、最终靠汇总打分（与顺序无关），可并发；
# ddddocr 是小模型，单线程 session + 多候选并发，比单条推理开满线程更快、且不卡机。
# 实测：455 变体顺序 ~37s → 并发6 ~9s，识别结果与 score 逐字节一致（无损）。
OCR_WORKERS = 6            # 并发推理的候选数（建议 = 物理核数）
OCR_INTRA_OP_THREADS = 1   # 每个 onnxruntime session 的内部线程数
OCR_CV_THREADS = 6         # OpenCV 预处理的内部线程数

# 强制 adaptive 档始终跑 balanced（455 变体），不在 fast 早停。
# 简单图会从 ~2-3s 变成 ~9s，但换取稳定的最高准确率（永不走变体少的 fast）。设 False 恢复自适应早停。
OCR_FORCE_BALANCED = True


def configure_ocr_environment() -> dict[str, Path]:
    paths = {
        "cache_root": OCR_CACHE_ROOT,
        "ddddocr": OCR_CACHE_ROOT / "ddddocr",
        "xdg": OCR_CACHE_ROOT / "xdg",
        "tmp": OCR_CACHE_ROOT / "tmp",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("XDG_CACHE_HOME", str(paths["xdg"]))
    os.environ.setdefault("TMPDIR", str(paths["tmp"]))
    os.environ.setdefault("TEMP", str(paths["tmp"]))
    os.environ.setdefault("TMP", str(paths["tmp"]))

    # 限制 OpenCV 预处理的内部线程数，避免吃满所有核导致整机卡顿。
    try:
        import cv2

        cv2.setNumThreads(OCR_CV_THREADS)
    except Exception:
        pass
    return paths
