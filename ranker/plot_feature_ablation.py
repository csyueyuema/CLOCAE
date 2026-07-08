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
OUT_DIR = BASE_DIR / "outputs" / "feature_ablations"
FIG_DIR = BASE_DIR / "figures"


def main():
    rows = json.loads((OUT_DIR / "feature_ablation_summary.json").read_text(encoding="utf-8"))
    labels = [x["label"] for x in rows]
    top1 = [x["top1"] for x in rows]
    top5 = [x["top5"] for x in rows]
    mrr = [x["mrr"] for x in rows]

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(x - width, top1, width, label="Top-1", color="#457b9d")
    ax.bar(x, top5, width, label="Top-5", color="#2a9d8f")
    ax.bar(x + width, mrr, width, label="MRR", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Score")
    ax.set_title("Wide-Feature Ablation Study")
    ax.legend(frameon=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "wide_feature_ablation.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"figure": str(FIG_DIR / "wide_feature_ablation.png")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
