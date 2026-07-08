#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Normalize result reports so Top-N metrics are shown as hit counts.

The raw evaluator JSON files may store Top-N as rates.  For thesis tables we
report Top-1/Top-5/Top-10 as the number of bugs localized within Top-N.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "ranker" / "results"
TABLES = RESULTS_ROOT / "tables"
REPORTS = RESULTS_ROOT / "reports"

TOP_KEYS = ("top1", "top3", "top5", "top10")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def top_count(value: Any, num_bugs: Any) -> int:
    if value is None:
        return 0
    value = float(value)
    bugs = int(num_bugs or 0)
    if value > 1.0:
        return int(round(value))
    return int(round(value * bugs))


def fmt_float(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def row_label(row: Dict[str, Any]) -> str:
    return str(row.get("paper_name") or row.get("label") or row.get("method") or row.get("name") or "-")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def count_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        bugs = copied.get("num_bugs") or copied.get("bugs")
        if bugs is not None:
            for key in TOP_KEYS:
                if key in copied:
                    copied[key] = top_count(copied[key], bugs)
            copied["top_unit"] = "count"
        out.append(copied)
    return out


def write_count_table(path: Path, title: str, rows: Sequence[Dict[str, Any]], include_top3: bool = False) -> None:
    top_headers = ["Top-1 (#)"]
    if include_top3:
        top_headers.append("Top-3 (#)")
    top_headers.extend(["Top-5 (#)", "Top-10 (#)"])
    lines = [f"# {title}", "", "Top-N 指标均为命中的 bug 数量，不保留百分比。", ""]
    lines.append("| Method | " + " | ".join(top_headers) + " | MAR | MFR | Bugs |")
    aligns = ["---"] + ["---:"] * len(top_headers) + ["---:", "---:", "---:"]
    lines.append("| " + " | ".join(aligns) + " |")
    for row in rows:
        parts = [row_label(row), str(row.get("top1", 0))]
        if include_top3:
            parts.append(str(row.get("top3", 0)))
        parts.extend([
            str(row.get("top5", 0)),
            str(row.get("top10", 0)),
            fmt_float(row.get("mar")),
            fmt_float(row.get("mfr")),
            str(row.get("num_bugs") or row.get("bugs") or "-"),
        ])
        lines.append("| " + " | ".join(parts) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_main_level(path: Path, title: str, row: Dict[str, Any], method: str) -> None:
    rows = count_rows([{**row, "paper_name": method}])
    write_count_table(path, title, rows, include_top3=False)


def write_main_file_statement() -> None:
    rows = count_rows(read_json(TABLES / "main_file_statement_results.json"))
    lines = [
        "# File-Level and Statement-Level Results",
        "",
        "Top-N 指标均为命中的 bug 数量，不保留百分比。",
        "",
        "## Main Results",
        "",
        "| Level | Method | Top-1 (#) | Top-5 (#) | Top-10 (#) | MAR | MFR | Bugs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['level']} | {row['method']} | {row['top1']} | {row['top5']} | {row['top10']} | "
            f"{fmt_float(row['mar'])} | {fmt_float(row['mfr'])} | {row['num_bugs']} |"
        )

    sections = [
        ("File-Level RQ1 Comparison", read_json(TABLES / "rq1_main_comparison.json")),
        ("File-Level Category Results", read_json(TABLES / "rq1_category_crashloccae.json")),
        ("File-Level Model Ablation", read_json(TABLES / "rq3_file_model_ablation.json")),
        ("Statement-Level Model Ablation", read_json(TABLES / "statement_model_ablation.json")),
    ]
    for title, data in sections:
        section_rows = count_rows(data)
        lines.extend(["", f"## {title}", ""])
        if section_rows and "category_id" in section_rows[0]:
            lines.append("| Cat. | Category | Bugs | Top-1 (#) | Top-5 (#) | Top-10 (#) | MAR | MFR |")
            lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for row in section_rows:
                lines.append(
                    f"| {row['category_id']} | {row['category']} | {row['num_bugs']} | {row['top1']} | {row['top5']} | {row['top10']} | "
                    f"{fmt_float(row['mar'])} | {fmt_float(row['mfr'])} |"
                )
        else:
            lines.append("| Method | Top-1 (#) | Top-5 (#) | Top-10 (#) | MAR | MFR | Bugs |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for row in section_rows:
                lines.append(
                    f"| {row_label(row)} | {row['top1']} | {row['top5']} | {row['top10']} | "
                    f"{fmt_float(row['mar'])} | {fmt_float(row['mfr'])} | {row['num_bugs']} |"
                )

    reduction = read_json(TABLES / "rq2_candidate_reduction.json")
    lines.extend([
        "",
        "## Candidate Reduction",
        "",
        f"- Bugs: {reduction['bugs']}",
        f"- Avg. repository files: {fmt_float(reduction['avg_repo_files'])}",
        f"- Avg. module candidates: {fmt_float(reduction['avg_module_candidates'])}",
        f"- Avg. reduction ratio: {fmt_float(reduction['avg_reduction_ratio'], 4)}",
    ])
    (REPORTS / "file_statement_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_category_results_all_methods() -> None:
    rows = count_rows(read_json(TABLES / "category_results_all_methods.json"))
    lines = [
        "# Category Results",
        "",
        "Project categories follow the paper dataset table. Top-N 指标均为命中的 bug 数量，不保留百分比。",
        "",
        "| Method | Cat. | Category | Bugs | Top-1 (#) | Top-5 (#) | Top-10 (#) | MAR | MFR |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['category_id']} | {row['category']} | {row['num_bugs']} | "
            f"{row['top1']} | {row['top5']} | {row['top10']} | {fmt_float(row['mar'])} | {fmt_float(row['mfr'])} |"
        )
    (REPORTS / "category_results_all_methods.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rq1_category() -> None:
    rows = count_rows(read_json(TABLES / "rq1_category_crashloccae.json"))
    lines = [
        "# RQ1 Category Results: CrashLocCAE",
        "",
        "Top-N 指标均为命中的 bug 数量，不保留百分比。",
        "",
        "| Cat. | Category | Bugs | Top-1 (#) | Top-5 (#) | Top-10 (#) | MAR | MFR |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['category_id']} | {row['category']} | {row['num_bugs']} | {row['top1']} | {row['top5']} | {row['top10']} | "
            f"{fmt_float(row['mar'])} | {fmt_float(row['mfr'])} |"
        )
    (REPORTS / "rq1_category_crashloccae.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_experiment_report() -> None:
    lines = [
        "# Experiment Report",
        "",
        "本报告保留论文实验的结果入口。Top-N 指标统一为命中的 bug 数量，不保留百分比。",
        "",
        "## Canonical Result Files",
        "",
        "| File | Purpose |",
        "| --- | --- |",
        "| `ranker/results/ALL_RQ_RESULTS.md` | 所有 RQ 的 5 methods x 4 categories x file/statement 结果 |",
        "| `ranker/results/tables/all_rq_5x4x2_long.csv` | 所有 RQ 的机器可读长表 |",
        "| `ranker/results/reports/rq_matrix.md` | RQ1 5x4x2 主矩阵 |",
        "| `ranker/results/figures/all_rq_style/` | 论文风格图 |",
        "",
        "## Main File/Statement Summary",
        "",
    ]
    main_rows = count_rows(read_json(TABLES / "main_file_statement_results.json"))
    lines.append("| Level | Method | Top-1 (#) | Top-5 (#) | Top-10 (#) | MAR | MFR | Bugs |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in main_rows:
        lines.append(
            f"| {row['level']} | {row['method']} | {row['top1']} | {row['top5']} | {row['top10']} | "
            f"{fmt_float(row['mar'])} | {fmt_float(row['mfr'])} | {row['num_bugs']} |"
        )
    (REPORTS / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_tables_for_paper_copy() -> None:
    """Write count-normalized copies for the small CSV tables used by reports."""
    names = [
        "main_file_statement_results",
        "rq1_main_comparison",
        "rq1_supplementary_comparison",
        "rq1_category_crashloccae",
        "category_results_all_methods",
        "rq3_file_model_ablation",
        "rq3_file_feature_ablation",
        "statement_model_ablation",
    ]
    for name in names:
        src = TABLES / f"{name}.json"
        if not src.exists():
            continue
        data = read_json(src)
        rows = count_rows(data if isinstance(data, list) else [data])
        write_csv(TABLES / f"{name}_counts.csv", rows)
        (TABLES / f"{name}_counts.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    write_main_level(REPORTS / "file_level_main.md", "File-Level Main Result", read_json(TABLES / "file_level_main.json"), "CrashLocCAE")
    write_main_level(REPORTS / "statement_level_main.md", "Statement-Level Main Result", read_json(TABLES / "statement_level_main.json"), "CrashLocCAE")
    write_count_table(REPORTS / "rq1_main_comparison.md", "RQ1 Main Comparison", count_rows(read_json(TABLES / "rq1_main_comparison.json")))
    write_count_table(REPORTS / "rq1_supplementary_comparison.md", "RQ1 Supplementary Comparison", count_rows(read_json(TABLES / "rq1_supplementary_comparison.json")))
    write_count_table(REPORTS / "rq3_file_model_ablation.md", "RQ3 File-Level Model Ablation", count_rows(read_json(TABLES / "rq3_file_model_ablation.json")))
    write_count_table(REPORTS / "rq3_file_feature_ablation.md", "RQ3 File-Level Feature Ablation", count_rows(read_json(TABLES / "rq3_file_feature_ablation.json")))
    write_count_table(REPORTS / "statement_model_ablation.md", "Statement-Level Model Ablation", count_rows(read_json(TABLES / "statement_model_ablation.json")))
    write_category_results_all_methods()
    write_rq1_category()
    write_main_file_statement()
    write_experiment_report()
    normalize_tables_for_paper_copy()

    # Keep the top-level 5x4x2 report aligned with the current count-based report.
    shutil.copy2(REPORTS / "rq_matrix.md", RESULTS_ROOT / "ALL_RESULTS_5x4x2.md")

    print("Normalized Top-N reports to hit counts.")


if __name__ == "__main__":
    main()
