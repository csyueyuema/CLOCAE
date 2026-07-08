#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 220

BASE_DIR = Path("ranker")
ABLATION_DIR = BASE_DIR / "outputs" / "ablations"
FIG_DIR = BASE_DIR / "figures"

MODEL_RUNS = [
    ("full", "Wide+Deep"),
    ("wide_only", "Wide Only"),
    ("deep_only", "Deep Only"),
]


def main():
    rows = []
    for run_name, label in MODEL_RUNS:
        metrics_path = ABLATION_DIR / run_name / "test_metrics.json"
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({
            "name": run_name,
            "label": label,
            **data["metrics"],
            "num_pairs": data["num_pairs"],
            "num_bugs": data["num_bugs"],
        })

    (ABLATION_DIR / "model_ablation_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    labels = [row["label"] for row in rows]
    top1 = [row["top1"] for row in rows]
    top5 = [row["top5"] for row in rows]
    mrr = [row["mrr"] for row in rows]

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    ax.bar(x - width, top1, width, label="Top-1", color="#457b9d")
    ax.bar(x, top5, width, label="Top-5", color="#2a9d8f")
    ax.bar(x + width, mrr, width, label="MRR", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Score")
    ax.set_title("Model-Level Ablation on File-Level Ranking")
    ax.legend(frameon=True)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "model_ablation_comparison.png", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({
        "figure": str(FIG_DIR / "model_ablation_comparison.png"),
        "summary": str(ABLATION_DIR / "model_ablation_summary.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
