#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path("ranker")
DATA_DIR = BASE_DIR / "data" / "manual_oracle_stage2_deep"
OUT_DIR = BASE_DIR / "outputs" / "ablations"

CONFIGS = [
    {
        "name": "full",
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
    {
        "name": "no_stack",
        "label": "w/o Stack",
        "train_args": ["--feature-groups", "structure", "domain"],
    },
    {
        "name": "no_structure",
        "label": "w/o Structure",
        "train_args": ["--feature-groups", "stack", "domain"],
    },
    {
        "name": "no_domain",
        "label": "w/o Domain",
        "train_args": ["--feature-groups", "stack", "structure"],
    },
]


def run(cmd):
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for cfg in CONFIGS:
        model_dir = OUT_DIR / cfg["name"]
        model_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable,
            "ranker/train_ranker.py",
            "--train-file",
            str(DATA_DIR / "train.jsonl"),
            "--valid-file",
            str(DATA_DIR / "valid.jsonl"),
            "--model-name",
            "distilroberta-base",
            "--output-dir",
            str(model_dir),
            "--epochs",
            "4",
            "--batch-size",
            "4",
            "--max-length",
            "384",
        ] + cfg["train_args"]
        run(train_cmd)

        metrics_path = model_dir / "test_metrics.json"
        eval_cmd = [
            sys.executable,
            "ranker/eval_ranker.py",
            "--model-dir",
            str(model_dir),
            "--test-file",
            str(DATA_DIR / "test.jsonl"),
            "--output-file",
            str(metrics_path),
        ]
        run(eval_cmd)

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary_rows.append(
            {
                "name": cfg["name"],
                "label": cfg["label"],
                **metrics["metrics"],
                "num_pairs": metrics["num_pairs"],
                "num_bugs": metrics["num_bugs"],
            }
        )

    summary_path = OUT_DIR / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "runs": [row["name"] for row in summary_rows]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
