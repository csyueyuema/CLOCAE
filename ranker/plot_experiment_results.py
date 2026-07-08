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
FIG_DIR = BASE_DIR / "figures"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def plot_summary(metrics_file: Path, out_name: str, title: str):
    data = load_json(metrics_file)
    metrics = data["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})

    names = ["Top-1", "Top-3", "Top-5", "Top-10", "MRR"]
    vals = [metrics["top1"], metrics["top3"], metrics["top5"], metrics["top10"], metrics["mrr"]]
    colors = ["#457b9d", "#1d3557", "#2a9d8f", "#90be6d", "#e76f51"]
    bars = axes[0].bar(names, vals, color=colors, width=0.62)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Retrieval Performance")
    for bar, val in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", fontsize=9)

    rank_names = ["MFR", "MAR"]
    rank_vals = [metrics["mfr"], metrics["mar"]]
    bars = axes[1].bar(rank_names, rank_vals, color=["#6d597a", "#264653"], width=0.58)
    axes[1].set_ylabel("Average Rank")
    axes[1].set_title("Ranking Position")
    for bar, val in zip(bars, rank_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + max(rank_vals) * 0.03, f"{val:.2f}", ha="center", fontsize=9)

    fig.suptitle(f"{title} ({data['num_bugs']} Bugs)", fontsize=13)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / out_name, bbox_inches="tight")
    plt.close(fig)


def plot_model_ablation(summary_file: Path, out_name: str, title: str):
    rows = load_json(summary_file)
    labels = [row["label"] for row in rows]
    top1 = [row["top1"] for row in rows]
    top5 = [row["top5"] for row in rows]
    top10 = [row["top10"] for row in rows]
    mrr = [row["mrr"] for row in rows]

    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.bar(x - 1.5 * width, top1, width, label="Top-1", color="#457b9d")
    ax.bar(x - 0.5 * width, top5, width, label="Top-5", color="#2a9d8f")
    ax.bar(x + 0.5 * width, top10, width, label="Top-10", color="#90be6d")
    ax.bar(x + 1.5 * width, mrr, width, label="MRR", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(frameon=True, ncols=4, loc="upper center")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / out_name, bbox_inches="tight")
    plt.close(fig)


def plot_feature_baselines(main_metrics_file: Path, baseline_file: Path, out_name: str):
    main = load_json(main_metrics_file)["metrics"]
    baselines = load_json(baseline_file)
    rows = [
        {"label": "Wide+Deep", "top1": main["top1"], "top5": main["top5"], "top10": main["top10"], "mrr": main["mrr"]},
    ]
    label_map = {
        "stack_only": "Stack Only",
        "domain_only": "CAE Domain",
        "heuristic": "Heuristic",
    }
    for row in baselines:
        rows.append({
            "label": label_map.get(row["baseline"], row["baseline"]),
            "top1": row["top1"],
            "top5": row["top5"],
            "top10": row["top10"],
            "mrr": row["mrr"],
        })

    labels = [row["label"] for row in rows]
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    ax.bar(x - 1.5 * width, [row["top1"] for row in rows], width, label="Top-1", color="#457b9d")
    ax.bar(x - 0.5 * width, [row["top5"] for row in rows], width, label="Top-5", color="#2a9d8f")
    ax.bar(x + 0.5 * width, [row["top10"] for row in rows], width, label="Top-10", color="#90be6d")
    ax.bar(x + 1.5 * width, [row["mrr"] for row in rows], width, label="MRR", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("Score")
    ax.set_title("File-Level Baselines")
    ax.legend(frameon=True, ncols=4, loc="upper center")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / out_name, bbox_inches="tight")
    plt.close(fig)


def plot_candidate_reduction(summary_file: Path, out_name: str):
    summary = load_json(summary_file)
    labels = ["Avg. Repository Files", "Avg. Module Candidates"]
    vals = [summary["avg_repo_files"], summary["avg_module_candidates"]]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    bars = ax.bar(labels, vals, color=["#264653", "#2a9d8f"], width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("Candidates per Bug (log scale)")
    ax.set_title(f"Module Mapping Candidate Reduction: {summary['avg_reduction_ratio'] * 100:.1f}%")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.08, f"{val:.1f}", ha="center", fontsize=10)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / out_name, bbox_inches="tight")
    plt.close(fig)


def plot_rq1_comparison(summary_file: Path, out_name: str):
    rows = load_json(summary_file)
    labels = [row["label"] for row in rows]
    top1 = [row["top1"] for row in rows]
    top5 = [row["top5"] for row in rows]
    top10 = [row["top10"] for row in rows]
    mrr = [row["mrr"] for row in rows]

    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    ax.bar(x - 1.5 * width, top1, width, label="Top-1", color="#457b9d")
    ax.bar(x - 0.5 * width, top5, width, label="Top-5", color="#2a9d8f")
    ax.bar(x + 0.5 * width, top10, width, label="Top-10", color="#90be6d")
    ax.bar(x + 1.5 * width, mrr, width, label="MRR", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("Score")
    ax.set_title("File-Level RQ1 Comparison")
    ax.legend(frameon=True, ncols=4, loc="upper center")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / out_name, bbox_inches="tight")
    plt.close(fig)


def main():
    plot_summary(
        BASE_DIR / "outputs" / "wide_deep_seed" / "test_metrics.json",
        "file_level_summary.png",
        "File-Level Ranking Summary",
    )
    plot_model_ablation(
        BASE_DIR / "outputs" / "model_ablation_summary.json",
        "file_model_ablation.png",
        "File-Level Model Ablation",
    )
    plot_summary(
        BASE_DIR / "outputs" / "line_main" / "test_metrics.json",
        "statement_level_summary.png",
        "Statement-Level Ranking Summary",
    )
    plot_model_ablation(
        BASE_DIR / "outputs" / "line_model_ablation_summary.json",
        "statement_model_ablation.png",
        "Statement-Level Model Ablation",
    )
    plot_feature_baselines(
        BASE_DIR / "outputs" / "wide_deep_seed" / "test_metrics.json",
        BASE_DIR / "outputs" / "feature_baselines.json",
        "file_feature_baselines.png",
    )
    plot_candidate_reduction(
        BASE_DIR / "outputs" / "efficiency" / "candidate_reduction_summary.json",
        "candidate_reduction.png",
    )
    plot_model_ablation(
        BASE_DIR / "outputs" / "feature_ablations" / "feature_ablation_summary.json",
        "file_feature_ablation.png",
        "File-Level Feature Ablation",
    )
    plot_rq1_comparison(
        BASE_DIR / "outputs" / "rq1_comparison_summary.json",
        "rq1_comparison.png",
    )
    print(json.dumps({
        "figures": [
            str(FIG_DIR / "file_level_summary.png"),
            str(FIG_DIR / "file_model_ablation.png"),
            str(FIG_DIR / "statement_level_summary.png"),
            str(FIG_DIR / "statement_model_ablation.png"),
            str(FIG_DIR / "file_feature_baselines.png"),
            str(FIG_DIR / "candidate_reduction.png"),
            str(FIG_DIR / "file_feature_ablation.png"),
            str(FIG_DIR / "rq1_comparison.png"),
        ]
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
