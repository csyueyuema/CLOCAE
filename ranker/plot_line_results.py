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
OUT_DIR = BASE_DIR / "outputs" / "line_experiments"
FIG_DIR = BASE_DIR / "figures"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def plot_statement_summary():
    data = load_json(OUT_DIR / "main" / "test_metrics.json")
    metrics = data["metrics"]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})

    left_names = ["Top-1", "Top-3", "Top-5", "MRR"]
    left_vals = [metrics["top1"], metrics["top3"], metrics["top5"], metrics["mrr"]]
    left_colors = ["#457b9d", "#1d3557", "#2a9d8f", "#e76f51"]
    bars = axes[0].bar(left_names, left_vals, color=left_colors, width=0.62)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Statement-Level Retrieval Performance")
    for bar, val in zip(bars, left_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    right_names = ["MFR", "MAR"]
    right_vals = [metrics["mfr"], metrics["mar"]]
    right_colors = ["#6d597a", "#264653"]
    bars = axes[1].bar(right_names, right_vals, color=right_colors, width=0.58)
    axes[1].set_ylabel("Average Rank")
    axes[1].set_title("Ranking Position Metrics")
    for bar, val in zip(bars, right_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + max(right_vals) * 0.03, f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle(f"Statement-Level Ranking Summary on Test Set ({data['num_bugs']} Bugs)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "statement_level_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_statement_model_ablation():
    rows = load_json(OUT_DIR / "model_ablation_summary.json")
    labels = [row["label"] for row in rows]
    top1 = [row["top1"] for row in rows]
    top5 = [row["top5"] for row in rows]
    mrr = [row["mrr"] for row in rows]

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(x - width, top1, width, label="Top-1", color="#457b9d")
    ax.bar(x, top5, width, label="Top-5", color="#2a9d8f")
    ax.bar(x + width, mrr, width, label="MRR", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("Score")
    ax.set_title("Model-Level Ablation on Statement-Level Ranking")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "statement_model_ablation.png", bbox_inches="tight")
    plt.close(fig)


def plot_statement_validation_curves():
    history = load_json(OUT_DIR / "main" / "train_history.json")
    epochs = [x["epoch"] for x in history]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(epochs, [x["top1"] for x in history], marker="o", color="#457b9d", label="Top-1")
    ax.plot(epochs, [x["top5"] for x in history], marker="o", color="#2a9d8f", label="Top-5")
    ax.plot(epochs, [x["mrr"] for x in history], marker="o", color="#e76f51", label="MRR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Score")
    ax.set_ylim(0.3, 0.7)
    ax.set_title("Statement-Level Validation Curves")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "statement_validation_curves.png", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_statement_summary()
    plot_statement_model_ablation()
    plot_statement_validation_curves()
    print(json.dumps({
        "figures": [
            str(FIG_DIR / "statement_level_summary.png"),
            str(FIG_DIR / "statement_model_ablation.png"),
            str(FIG_DIR / "statement_validation_curves.png"),
        ]
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
