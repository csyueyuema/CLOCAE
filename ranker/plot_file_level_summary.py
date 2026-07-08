#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 220

BASE_DIR = Path("ranker")
METRICS_FILE = BASE_DIR / "outputs" / "wide_deep_seed" / "test_metrics.json"
FIG_DIR = BASE_DIR / "figures"


def main():
    data = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    metrics = data["metrics"]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})

    hit_names = ["Top-1", "Top-3", "Top-5", "MRR"]
    hit_vals = [metrics["top1"], metrics["top3"], metrics["top5"], metrics["mrr"]]
    hit_colors = ["#457b9d", "#1d3557", "#2a9d8f", "#e76f51"]
    bars = axes[0].bar(hit_names, hit_vals, color=hit_colors, width=0.62)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Score")
    axes[0].set_title("File-Level Retrieval Performance")
    for bar, val in zip(bars, hit_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    rank_names = ["MFR", "MAR"]
    rank_vals = [metrics["mfr"], metrics["mar"]]
    rank_colors = ["#6d597a", "#264653"]
    bars = axes[1].bar(rank_names, rank_vals, color=rank_colors, width=0.58)
    axes[1].set_ylabel("Average Rank")
    axes[1].set_title("Ranking Position Metrics")
    for bar, val in zip(bars, rank_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.35, f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle(f"File-Level Ranking Summary on Test Set ({data['num_bugs']} Bugs)", fontsize=13)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "file_level_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"figure": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
