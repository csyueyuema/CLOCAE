#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path("ranker")
DATA_DIR = BASE_DIR / "data" / "manual_oracle_stage2_deep"
OUT_DIR = BASE_DIR / "outputs" / "feature_ablations"

CONFIGS = [
    {
        "name": "wide_only_all",
        "label": "Wide Only",
        "train_args": ["--disable-deep"],
    },
    {
        "name": "wide_no_stack",
        "label": "Wide w/o Stack",
        "train_args": ["--disable-deep", "--feature-groups", "structure", "domain"],
    },
    {
        "name": "wide_no_structure",
        "label": "Wide w/o Structure",
        "train_args": ["--disable-deep", "--feature-groups", "stack", "domain"],
    },
    {
        "name": "wide_no_domain",
        "label": "Wide w/o Domain",
        "train_args": ["--disable-deep", "--feature-groups", "stack", "structure"],
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
        rows.append(
            {
                "name": cfg["name"],
                "label": cfg["label"],
                **metrics["metrics"],
                "num_pairs": metrics["num_pairs"],
                "num_bugs": metrics["num_bugs"],
            }
        )

    out = OUT_DIR / "feature_ablation_summary.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(out), "runs": [x["name"] for x in rows]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
