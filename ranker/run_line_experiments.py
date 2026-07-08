#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path("ranker")
DATA_DIR = BASE_DIR / "data" / "manual_line_stage3_light"
OUT_DIR = BASE_DIR / "outputs" / "line_experiments"

CONFIGS = [
    {
        "name": "main",
        "label": "Wide+Deep",
        "train_args": [],
    },
    {
        "name": "wide_only",
        "label": "Wide Only",
        "train_args": ["--disable-deep"],
    },
    {
        "name": "deep_only",
        "label": "Deep Only",
        "train_args": ["--disable-wide"],
    },
]


def run(cmd):
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg in CONFIGS:
        model_dir = OUT_DIR / cfg["name"]
        train_cmd = [
            sys.executable,
            "ranker/train_line_ranker.py",
            "--train-file",
            str(DATA_DIR / "train.jsonl"),
            "--valid-file",
            str(DATA_DIR / "valid.jsonl"),
            "--model-name",
            "distilroberta-base",
            "--output-dir",
            str(model_dir),
            "--epochs",
            "2",
            "--batch-size",
            "8",
            "--max-length",
            "256",
        ] + cfg["train_args"]
        run(train_cmd)

        test_metrics = model_dir / "test_metrics.json"
        eval_cmd = [
            sys.executable,
            "ranker/eval_line_ranker.py",
            "--model-dir",
            str(model_dir),
            "--test-file",
            str(DATA_DIR / "test.jsonl"),
            "--output-file",
            str(test_metrics),
        ]
        run(eval_cmd)
        metrics = json.loads(test_metrics.read_text(encoding="utf-8"))
        rows.append({
            "name": cfg["name"],
            "label": cfg["label"],
            **metrics["metrics"],
            "num_pairs": metrics["num_pairs"],
            "num_bugs": metrics["num_bugs"],
        })

    out = OUT_DIR / "model_ablation_summary.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
