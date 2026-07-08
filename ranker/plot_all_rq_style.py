#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot all RQ results in a grouped-bar + improvement-line style."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "ranker" / "results" / "tables" / "all_rq_5x4x2_long.csv"
OUT_DIR = ROOT / "ranker" / "results" / "figures" / "all_rq_style"
INDEX = OUT_DIR / "FIGURE_INDEX.md"

METHOD_ORDER = ["CrashLocCAE", "LLM-Only", "CrashLocator", "Scaffle", "w/o CAE Features"]
LEVEL_ORDER = ["File", "Statement"]
CATEGORY_ORDER = ["1", "2", "3", "4"]
CATEGORY_NAMES = {
    "1": "FEM / Scientific Computing",
    "2": "CFD / Flow Simulation",
    "3": "Multiphysics Frameworks",
    "4": "Modeling / Dynamics",
}
RQ_TITLES = {
    "RQ1": "Effectiveness Compared with Baselines",
    "RQ2": "Module Reduction Effectiveness and Candidate-Space Reduction",
    "RQ3": "CAE-Specific Feature Contribution",
    "RQ4": "Efficiency / Search-Space Cost",
}

COLORS = {
    "CrashLocCAE": "#8fbf8f",
    "LLM-Only": "#c7d4e8",
    "CrashLocator": "#7ba3f4",
    "Scaffle": "#f0b35b",
    "w/o CAE Features": "#d9a3a3",
}

METRIC_LABELS = {
    "top1": "Top1 (#)",
    "top5": "Top5 (#)",
    "top10": "Top10 (#)",
    "mar": "MAR",
    "mfr": "MFR",
    "avg_candidate_units_per_bug": "Cand/Bug",
    "avg_module_candidates_after_reduction": "Module Cand.",
    "avg_module_reduction_ratio": "Reduction",
}

GOOD_HIGH = {"top1", "top5", "top10", "avg_module_reduction_ratio"}
GOOD_LOW = {"mar", "mfr", "avg_candidate_units_per_bug", "avg_module_candidates_after_reduction"}


def clean_category_id(value) -> str:
    if pd.isna(value):
        return ""
    try:
        return str(int(float(value)))
    except Exception:
        return str(value)


def to_plot_value(row: pd.Series, metric: str) -> float:
    value = float(row.get(metric, 0.0) or 0.0)
    if metric == "avg_module_reduction_ratio":
        return value * 100.0
    return value


def row_map(df: pd.DataFrame) -> Dict[str, pd.Series]:
    out = {}
    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        if not sub.empty:
            out[method] = sub.iloc[0]
    return out


def metric_values(df: pd.DataFrame, metrics: Sequence[str]) -> Dict[str, List[float]]:
    rows = row_map(df)
    values: Dict[str, List[float]] = {}
    for method in METHOD_ORDER:
        if method not in rows:
            continue
        values[method] = [to_plot_value(rows[method], metric) for metric in metrics]
    return values


def improvement_vs_reference(df: pd.DataFrame, metrics: Sequence[str], reference: str | None) -> List[float]:
    rows = row_map(df)
    if "CrashLocCAE" not in rows:
        return [0.0 for _ in metrics]
    proposed = rows["CrashLocCAE"]
    line = []
    for metric in metrics:
        p = to_plot_value(proposed, metric)
        if reference and reference in rows:
            ref = to_plot_value(rows[reference], metric)
        else:
            candidates = []
            for method, row in rows.items():
                if method == "CrashLocCAE":
                    continue
                candidates.append(to_plot_value(row, metric))
            if not candidates:
                ref = p
            elif metric in GOOD_LOW:
                ref = min(candidates)
            else:
                ref = max(candidates)
        if abs(ref) < 1e-12:
            line.append(0.0)
        elif metric in GOOD_LOW:
            line.append((ref - p) / abs(ref) * 100.0)
        else:
            line.append((p - ref) / abs(ref) * 100.0)
    return line


def reduction_line(df: pd.DataFrame, metrics: Sequence[str]) -> List[float] | None:
    if "avg_module_reduction_ratio" not in df.columns:
        return None
    vals = df["avg_module_reduction_ratio"].dropna()
    if vals.empty:
        return None
    value = float(vals.iloc[0]) * 100.0
    return [value for _ in metrics]


def annotate_bars(ax, bars, fontsize: int = 7) -> None:
    for bar in bars:
        height = bar.get_height()
        if not math.isfinite(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(ax.get_ylim()[1] * 0.01, 0.05),
            f"{int(round(height))}" if abs(height - round(height)) < 1e-6 else f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=35,
            color="#303030",
        )


def style_axes(ax) -> None:
    ax.grid(axis="y", linestyle="--", alpha=0.22, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color("#444444")
    ax.tick_params(axis="both", labelsize=10)


def draw_grouped_chart(
    ax,
    df: pd.DataFrame,
    rq: str,
    level: str,
    category_id: str,
    category_name: str,
    title_prefix: str = "",
    show_legend: bool = True,
    compact: bool = False,
) -> None:
    if rq in {"RQ1", "RQ3"}:
        metrics = ["top1", "top5", "top10", "mar", "mfr"]
        y_label = "Hit Counts / Rank Metrics"
        ref = "w/o CAE Features" if rq == "RQ3" else None
        improvement_label = "Improvement vs w/o CAE" if rq == "RQ3" else "Improvement vs Best Baseline"
        second_line = None
    elif rq == "RQ2":
        metrics = ["top1", "top5", "top10", "mar", "mfr", "avg_candidate_units_per_bug"]
        y_label = "Hit Counts / Candidate Cost"
        ref = None
        improvement_label = "Improvement vs Best Baseline"
        second_line = reduction_line(df, metrics)
    else:
        metrics = ["avg_candidate_units_per_bug", "avg_module_candidates_after_reduction", "mar", "mfr"]
        y_label = "Search Cost / Rank Metrics"
        ref = None
        improvement_label = "Improvement vs Best Baseline"
        second_line = reduction_line(df, metrics)

    values = metric_values(df, metrics)
    x = np.arange(len(metrics))
    methods = [m for m in METHOD_ORDER if m in values]
    width = min(0.16, 0.78 / max(len(methods), 1))
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width

    max_value = 0.0
    for offset, method in zip(offsets, methods):
        bars = ax.bar(
            x + offset,
            values[method],
            width,
            label=method,
            color=COLORS.get(method, "#cccccc"),
            edgecolor="white",
            linewidth=0.7,
            alpha=0.95,
        )
        if not compact:
            annotate_bars(ax, bars)
        max_value = max(max_value, max(values[method]) if values[method] else 0.0)

    ax.set_ylabel(y_label, fontsize=11 if compact else 12)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], fontsize=9 if compact else 11)
    ax.set_ylim(0, max(1.0, max_value * (1.30 if not compact else 1.18)))
    title = f"{title_prefix}{category_id}. {category_name}"
    ax.set_title(title, fontsize=11 if compact else 13, pad=8)
    style_axes(ax)

    ax2 = ax.twinx()
    imp = improvement_vs_reference(df, metrics, ref)
    ax2.plot(
        x,
        imp,
        color="#ff7f0e",
        marker="o",
        linewidth=1.5,
        markersize=4,
        label=improvement_label,
    )
    if second_line is not None:
        ax2.plot(
            x,
            second_line,
            color="#e31a1c",
            marker="s",
            linestyle="--",
            linewidth=1.3,
            markersize=4,
            label="Module Reduction",
        )
    finite_lines = [v for v in imp if math.isfinite(v)]
    if second_line is not None:
        finite_lines += [v for v in second_line if math.isfinite(v)]
    if finite_lines:
        low = min(finite_lines + [0.0])
        high = max(finite_lines + [0.0])
        pad = max((high - low) * 0.18, 5.0)
        ax2.set_ylim(low - pad, high + pad)
    ax2.set_ylabel("Improvement / Reduction (%)", fontsize=10 if compact else 12)
    ax2.tick_params(axis="y", labelsize=9 if compact else 10)

    if show_legend:
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            handles1 + handles2,
            labels1 + labels2,
            loc="upper left",
            fontsize=7.5 if compact else 8.5,
            frameon=True,
            framealpha=0.85,
        )


def safe_name(text: str) -> str:
    return (
        text.lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("-", "_")
        .replace(".", "")
    )


def make_single_figures(df: pd.DataFrame) -> List[Path]:
    paths: List[Path] = []
    single_dir = OUT_DIR / "single"
    single_dir.mkdir(parents=True, exist_ok=True)
    for rq in ["RQ1", "RQ2", "RQ3", "RQ4"]:
        for level in LEVEL_ORDER:
            for cat in CATEGORY_ORDER:
                subset = df[(df["rq"] == rq) & (df["level"] == level) & (df["category_id"] == cat)]
                if subset.empty:
                    continue
                fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=220)
                draw_grouped_chart(
                    ax,
                    subset,
                    rq,
                    level,
                    cat,
                    CATEGORY_NAMES[cat],
                    title_prefix=f"{rq} {level}: ",
                    show_legend=True,
                    compact=False,
                )
                fig.tight_layout()
                path = single_dir / f"{rq.lower()}_{safe_name(level)}_cat{cat}.png"
                fig.savefig(path, bbox_inches="tight")
                fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
                plt.close(fig)
                paths.append(path)
    return paths


def make_grid_figures(df: pd.DataFrame) -> List[Path]:
    paths: List[Path] = []
    grid_dir = OUT_DIR / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    for rq in ["RQ1", "RQ2", "RQ3", "RQ4"]:
        for level in LEVEL_ORDER:
            fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2), dpi=220)
            axes = axes.flatten()
            for ax, cat in zip(axes, CATEGORY_ORDER):
                subset = df[(df["rq"] == rq) & (df["level"] == level) & (df["category_id"] == cat)]
                draw_grouped_chart(
                    ax,
                    subset,
                    rq,
                    level,
                    cat,
                    CATEGORY_NAMES[cat],
                    title_prefix="",
                    show_legend=(cat == "1"),
                    compact=True,
                )
            fig.suptitle(f"{rq} {level}-Level: {RQ_TITLES[rq]}", fontsize=17, y=0.995)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            path = grid_dir / f"{rq.lower()}_{safe_name(level)}_all_categories.png"
            fig.savefig(path, bbox_inches="tight")
            fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
    return paths


def write_index(single_paths: Sequence[Path], grid_paths: Sequence[Path]) -> None:
    lines = [
        "# All RQ Experiment Figures",
        "",
        "图片风格统一为 grouped bars + right-axis improvement/reduction lines。PNG 用于快速查看，PDF 用于论文排版。",
        "",
        "## Grid Figures",
        "",
    ]
    for path in grid_paths:
        rel = path.relative_to(ROOT)
        lines.append(f"- `{rel}`")
    lines.extend(["", "## Single Figures", ""])
    for path in single_paths:
        rel = path.relative_to(ROOT)
        lines.append(f"- `{rel}`")
    INDEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)
    df["category_id"] = df["category_id"].map(clean_category_id)
    single_paths = make_single_figures(df)
    grid_paths = make_grid_figures(df)
    write_index(single_paths, grid_paths)
    print(f"Generated {len(single_paths)} single figures and {len(grid_paths)} grid figures.")
    print(INDEX)


if __name__ == "__main__":
    main()
