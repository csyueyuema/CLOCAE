#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from eval_ranker import evaluate_model
from train_ranker import FEATURE_KEYS, RankingDataset, WideDeepRanker, collate_fn


sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 220

BASE_DIR = Path("ranker")
MODEL_DIR = BASE_DIR / "outputs" / "wide_deep_seed"
TEST_FILE = BASE_DIR / "data" / "manual_oracle_stage2_deep" / "test.jsonl"
FIG_DIR = BASE_DIR / "figures"


def load_history():
    return json.loads((MODEL_DIR / "train_history.json").read_text(encoding="utf-8"))


def load_metrics():
    return json.loads((MODEL_DIR / "test_metrics.json").read_text(encoding="utf-8"))


def load_test_predictions():
    ckpt = torch.load(MODEL_DIR / "ranker.pt", map_location="cpu")
    model_args = ckpt["args"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "encoder")
    dataset = RankingDataset(TEST_FILE)
    loader = DataLoader(
        dataset,
        batch_size=model_args.get("batch_size", 4),
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, model_args.get("max_length", 384)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WideDeepRanker(
        model_name=str(MODEL_DIR / "encoder"),
        wide_dim=len(FEATURE_KEYS),
        deep_hidden=model_args.get("deep_hidden", 256),
        wide_hidden=model_args.get("wide_hidden", 64),
        dropout=model_args.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    _, rows = evaluate_model(model, loader, device)
    return rows


def group_rows(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["bug_id"], []).append(row)
    return groups


def compute_first_positive_ranks(rows):
    first_ranks = []
    candidate_sizes = []
    groups = group_rows(rows)
    for bug_id, items in groups.items():
        ranked = sorted(items, key=lambda x: (-x["score"], x["file"]))
        positives = [idx + 1 for idx, item in enumerate(ranked) if item["label"] > 0.5]
        if positives:
            first_ranks.append(min(positives))
            candidate_sizes.append(len(ranked))
    return first_ranks, candidate_sizes


def plot_hit_rates(metrics):
    vals = metrics["metrics"]
    names = ["Top-1", "Top-3", "Top-5"]
    scores = [vals["top1"], vals["top3"], vals["top5"]]
    colors = ["#457b9d", "#2a9d8f", "#e9c46a"]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    bars = ax.bar(names, scores, color=colors, width=0.6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Hit Rate")
    ax.set_title("File-Level Hit Rate on Test Set")
    ax.set_xlabel(f"84 Bugs, 5,544 (bug, file) Pairs")
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.02, f"{score:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "file_level_hit_rates.png", bbox_inches="tight")
    plt.close(fig)


def plot_rank_metrics(metrics):
    vals = metrics["metrics"]
    names = ["MRR", "MFR", "MAR"]
    scores = [vals["mrr"], vals["mfr"], vals["mar"]]
    colors = ["#e76f51", "#6d597a", "#264653"]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    axes[0].bar(["MRR"], [scores[0]], color=[colors[0]], width=0.5)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Reciprocal Rank")
    axes[0].text(0, scores[0] + 0.03, f"{scores[0]:.3f}", ha="center", va="bottom", fontsize=10)

    bars = axes[1].bar(["MFR", "MAR"], scores[1:], color=colors[1:], width=0.55)
    axes[1].set_ylabel("Average Rank")
    axes[1].set_title("Lower Is Better")
    for bar, score in zip(bars, scores[1:]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, score + 0.35, f"{score:.2f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("File-Level Ranking Metrics on Test Set")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "file_level_rank_metrics.png", bbox_inches="tight")
    plt.close(fig)


def plot_validation_curves(history):
    epochs = [x["epoch"] for x in history]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(epochs, [x["top1"] for x in history], marker="o", color="#457b9d", label="Top-1")
    ax.plot(epochs, [x["top5"] for x in history], marker="o", color="#2a9d8f", label="Top-5")
    ax.plot(epochs, [x["mrr"] for x in history], marker="o", color="#e76f51", label="MRR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Score")
    ax.set_ylim(0.35, 0.75)
    ax.set_title("Validation Ranking Curves")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "validation_ranking_curves.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(epochs, [x["train_loss"] for x in history], marker="o", color="#577590", label="Train Loss")
    ax.plot(epochs, [x["loss"] for x in history], marker="o", color="#f94144", label="Valid Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_loss_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_first_rank_distribution(rows):
    first_ranks, candidate_sizes = compute_first_positive_ranks(rows)
    capped = [min(rank, 20) for rank in first_ranks]
    counter = Counter(capped)
    xs = list(range(1, 21))
    ys = [counter.get(x, 0) for x in xs]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(xs, ys, color="#8ab17d", width=0.8)
    ax.set_xlabel("Rank of First Correct File (20 means >=20)")
    ax.set_ylabel("Bug Count")
    ax.set_title("Distribution of First Correct File Rank")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "first_correct_rank_distribution.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "candidate_size_mean": float(np.mean(candidate_sizes)) if candidate_sizes else 0.0,
        "candidate_size_median": float(np.median(candidate_sizes)) if candidate_sizes else 0.0,
        "first_rank_mean": float(np.mean(first_ranks)) if first_ranks else 0.0,
        "first_rank_median": float(np.median(first_ranks)) if first_ranks else 0.0,
        "first_rank_le_1": int(sum(1 for x in first_ranks if x <= 1)),
        "first_rank_le_3": int(sum(1 for x in first_ranks if x <= 3)),
        "first_rank_le_5": int(sum(1 for x in first_ranks if x <= 5)),
        "num_bugs": len(first_ranks),
    }
    (FIG_DIR / "ranking_distribution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    history = load_history()
    metrics = load_metrics()
    rows = load_test_predictions()

    plot_hit_rates(metrics)
    plot_rank_metrics(metrics)
    plot_validation_curves(history)
    plot_first_rank_distribution(rows)

    print(json.dumps({
        "figures": [
            str(FIG_DIR / "file_level_hit_rates.png"),
            str(FIG_DIR / "file_level_rank_metrics.png"),
            str(FIG_DIR / "validation_ranking_curves.png"),
            str(FIG_DIR / "training_loss_curves.png"),
            str(FIG_DIR / "first_correct_rank_distribution.png"),
        ],
        "summary": str(FIG_DIR / "ranking_distribution_summary.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
