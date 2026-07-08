#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build all-RQ tables expanded by 5 methods x 4 categories x 2 levels.

The result is intentionally a long, thesis-friendly table pack:
- RQ1: effectiveness metrics for all five methods.
- RQ2: effectiveness plus module/candidate-space reduction context.
- RQ3: CAE-feature contribution, centered on CrashLocCAE vs w/o CAE Features,
       while still preserving the five-method 4x2 layout for comparison.
- RQ4: candidate/search-space efficiency for all five methods.

No wall-clock runtime is fabricated here; RQ4 uses candidate-space cost already
recorded by the pipeline.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

CATEGORY_NAMES = {
    "1": "FEM / Scientific Computing",
    "2": "CFD / Flow Simulation",
    "3": "Multiphysics Frameworks",
    "4": "Modeling / Dynamics",
}

CATEGORY_MAP = {
    "CalculiX-Examples": "1",
    "Calculix": "1",
    "calculix": "1",
    "deal.II": "1",
    "dealii": "1",
    "FireDrake": "1",
    "firedrake": "1",
    "FreeFem": "1",
    "FreeFEM": "1",
    "gridap": "1",
    "Gridap": "1",
    "MFEM": "1",
    "mfem": "1",
    "SfePy": "1",
    "sfepy": "1",
    "Sparselizard": "1",
    "sparselizard": "1",
    "code": "2",
    "Code Saturne": "2",
    "Code_Saturne": "2",
    "Code-Saturne": "2",
    "coolfluid": "2",
    "COOLFluiD": "2",
    "FDS": "2",
    "fds": "2",
    "Fluidity": "2",
    "Nek5000": "2",
    "SU2": "2",
    "su2": "2",
    "xcompact3d": "2",
    "Xcompact3d": "2",
    "Goma": "3",
    "Kratos": "3",
    "OpenModelica": "4",
    "ROSS": "4",
    "ross": "4",
}

METHOD_ORDER = ["CrashLocCAE", "LLM-Only", "CrashLocator", "Scaffle", "w/o CAE Features"]
LEVEL_ORDER = ["File", "Statement"]

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "ranker" / "results"
TABLES = RESULTS_ROOT / "tables"
REPORTS = RESULTS_ROOT / "reports"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def category_for(project: str) -> str:
    if project in CATEGORY_MAP:
        return CATEGORY_MAP[project]
    normalized = project.lower().replace("_", "").replace("-", "").replace(".", "").replace(" ", "")
    for name, cat in CATEGORY_MAP.items():
        key = name.lower().replace("_", "").replace("-", "").replace(".", "").replace(" ", "")
        if normalized == key:
            return cat
    return "unknown"


def pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def top_count(rate: float, num_bugs: int) -> int:
    """Convert a Top-k hit rate back to the number of localized bugs."""
    if float(rate) > 1.0:
        return int(round(float(rate)))
    return int(round(float(rate) * int(num_bugs)))


def fmt(v: float | int | None) -> str:
    if v is None:
        return "-"
    if isinstance(v, int):
        return str(v)
    return f"{v:.2f}"


def metric_row(row: Dict[str, Any]) -> str:
    return (
        f"| {row['method']} | {row['top1']} | {row['top5']} | {row['top10']} | "
        f"{fmt(row['mar'])} | {fmt(row['mfr'])} |"
    )


def metric_table(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "| Method | Top-1 (#) | Top-5 (#) | Top-10 (#) | MAR | MFR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_method = {r["method"]: r for r in rows}
    for method in METHOD_ORDER:
        if method in by_method:
            lines.append(metric_row(by_method[method]))
    return "\n".join(lines)


def category_level_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return row["level"], str(row["category_id"]), row["dataset"]


def load_effectiveness_rows() -> List[Dict[str, Any]]:
    rows = read_json(TABLES / "rq_matrix.json")
    # Keep a uniform numeric schema for all later RQ tables.
    out = []
    for r in rows:
        num_bugs = int(r.get("num_bugs", 0))
        if r.get("top_unit") == "count":
            top1 = int(round(float(r["top1"])))
            top5 = int(round(float(r["top5"])))
            top10 = int(round(float(r["top10"])))
        else:
            top1 = top_count(float(r["top1"]), num_bugs)
            top5 = top_count(float(r["top5"]), num_bugs)
            top10 = top_count(float(r["top10"]), num_bugs)
        out.append({
            "level": r["level"],
            "category_id": str(r["category_id"]),
            "dataset": r["dataset"],
            "method": r["method"],
            "top1": top1,
            "top5": top5,
            "top10": top10,
            "mar": float(r["mar"]),
            "mfr": float(r["mfr"]),
            "num_bugs": num_bugs,
            "num_pairs": int(r.get("num_pairs", 0)),
        })
    return out


def load_candidate_reduction_context() -> Dict[Tuple[str, str], Dict[str, float]]:
    current_rows = TABLES / "candidate_reduction_rows.jsonl"
    legacy_rows = ROOT / "ranker" / "archive" / "outputs_before_results" / "efficiency" / "candidate_reduction_rows.jsonl"
    rows = read_jsonl(current_rows if current_rows.exists() else legacy_rows)
    # Use test split when available, because all reported metrics are test metrics.
    test_rows = [r for r in rows if r.get("split") == "test"] or rows
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in test_rows:
        cat = category_for(row.get("project", ""))
        if cat in CATEGORY_NAMES:
            grouped[cat].append(row)

    context: Dict[Tuple[str, str], Dict[str, float]] = {}
    for cat, items in grouped.items():
        repo_files = [float(r["repo_files"]) for r in items]
        module_candidates = [float(r["module_candidates"]) for r in items]
        reductions = [float(r["reduction_ratio"]) for r in items]
        for level in LEVEL_ORDER:
            context[(level, cat)] = {
                "avg_repo_files": mean(repo_files),
                "avg_module_candidates": mean(module_candidates),
                "avg_reduction_ratio": mean(reductions),
                "median_repo_files": median(repo_files),
                "median_module_candidates": median(module_candidates),
                "bugs_for_reduction": len(items),
            }
    return context


def build_rq2_rows(effectiveness: Sequence[Dict[str, Any]], context: Dict[Tuple[str, str], Dict[str, float]]) -> List[Dict[str, Any]]:
    rows = []
    for r in effectiveness:
        ctx = context.get((r["level"], r["category_id"]), {})
        avg_candidate_units = r["num_pairs"] / r["num_bugs"] if r["num_bugs"] else 0.0
        rows.append({
            "rq": "RQ2",
            "level": r["level"],
            "category_id": r["category_id"],
            "dataset": r["dataset"],
            "method": r["method"],
            "top1": r["top1"],
            "top5": r["top5"],
            "top10": r["top10"],
            "mar": r["mar"],
            "mfr": r["mfr"],
            "num_bugs": r["num_bugs"],
            "num_pairs": r["num_pairs"],
            "avg_candidate_units_per_bug": avg_candidate_units,
            "avg_repo_files_before_reduction": ctx.get("avg_repo_files"),
            "avg_module_candidates_after_reduction": ctx.get("avg_module_candidates"),
            "avg_module_reduction_ratio": ctx.get("avg_reduction_ratio"),
        })
    return rows


def build_rq3_rows(effectiveness: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    grouped: Dict[Tuple[str, str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in effectiveness:
        grouped[category_level_key(r)][r["method"]] = r

    for key in sorted(grouped, key=lambda x: (LEVEL_ORDER.index(x[0]), int(x[1]))):
        methods = grouped[key]
        no_cae = methods.get("w/o CAE Features")
        for method in METHOD_ORDER:
            if method not in methods:
                continue
            r = methods[method]
            row = {
                "rq": "RQ3",
                "level": r["level"],
                "category_id": r["category_id"],
                "dataset": r["dataset"],
                "method": method,
                "top1": r["top1"],
                "top5": r["top5"],
                "top10": r["top10"],
                "mar": r["mar"],
                "mfr": r["mfr"],
                "num_bugs": r["num_bugs"],
                "num_pairs": r["num_pairs"],
            }
            if no_cae:
                row["delta_top1_vs_wo_cae"] = r["top1"] - no_cae["top1"]
                row["delta_top5_vs_wo_cae"] = r["top5"] - no_cae["top5"]
                row["delta_top10_vs_wo_cae"] = r["top10"] - no_cae["top10"]
                # Negative MAR/MFR deltas are improvements because lower is better.
                row["delta_mar_vs_wo_cae"] = r["mar"] - no_cae["mar"]
                row["delta_mfr_vs_wo_cae"] = r["mfr"] - no_cae["mfr"]
            rows.append(row)
    return rows


def build_rq4_rows(effectiveness: Sequence[Dict[str, Any]], context: Dict[Tuple[str, str], Dict[str, float]]) -> List[Dict[str, Any]]:
    rows = []
    for r in effectiveness:
        ctx = context.get((r["level"], r["category_id"]), {})
        avg_candidate_units = r["num_pairs"] / r["num_bugs"] if r["num_bugs"] else 0.0
        rows.append({
            "rq": "RQ4",
            "level": r["level"],
            "category_id": r["category_id"],
            "dataset": r["dataset"],
            "method": r["method"],
            "num_bugs": r["num_bugs"],
            "num_pairs": r["num_pairs"],
            "avg_candidate_units_per_bug": avg_candidate_units,
            "avg_repo_files_before_reduction": ctx.get("avg_repo_files"),
            "avg_module_candidates_after_reduction": ctx.get("avg_module_candidates"),
            "avg_module_reduction_ratio": ctx.get("avg_reduction_ratio"),
            "mar": r["mar"],
            "mfr": r["mfr"],
        })
    return rows


def rq2_table(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "| Method | Top-1 (#) | Top-5 (#) | Top-10 (#) | MAR | MFR | Bugs | Pairs | Avg. Candidate Units/Bug | Avg. Repo Files | Avg. Module Candidates | Reduction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_method = {r["method"]: r for r in rows}
    for method in METHOD_ORDER:
        if method not in by_method:
            continue
        r = by_method[method]
        lines.append(
            f"| {method} | {r['top1']} | {r['top5']} | {r['top10']} | {fmt(r['mar'])} | {fmt(r['mfr'])} | "
            f"{r['num_bugs']} | {r['num_pairs']} | {fmt(r['avg_candidate_units_per_bug'])} | "
            f"{fmt(r.get('avg_repo_files_before_reduction'))} | {fmt(r.get('avg_module_candidates_after_reduction'))} | "
            f"{pct(r.get('avg_module_reduction_ratio') or 0.0)} |"
        )
    return "\n".join(lines)


def rq3_table(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "| Method | Top-1 (#) | Top-5 (#) | Top-10 (#) | MAR | MFR | Delta Top-1 (#) vs w/o CAE | Delta MAR vs w/o CAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_method = {r["method"]: r for r in rows}
    for method in METHOD_ORDER:
        if method not in by_method:
            continue
        r = by_method[method]
        lines.append(
            f"| {method} | {r['top1']} | {r['top5']} | {r['top10']} | {fmt(r['mar'])} | {fmt(r['mfr'])} | "
            f"{r.get('delta_top1_vs_wo_cae', 0)} | {fmt(r.get('delta_mar_vs_wo_cae'))} |"
        )
    return "\n".join(lines)


def rq4_table(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "| Method | Bugs | Pairs | Avg. Candidate Units/Bug | Avg. Repo Files | Avg. Module Candidates | Reduction | MAR | MFR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_method = {r["method"]: r for r in rows}
    for method in METHOD_ORDER:
        if method not in by_method:
            continue
        r = by_method[method]
        lines.append(
            f"| {method} | {r['num_bugs']} | {r['num_pairs']} | {fmt(r['avg_candidate_units_per_bug'])} | "
            f"{fmt(r.get('avg_repo_files_before_reduction'))} | {fmt(r.get('avg_module_candidates_after_reduction'))} | "
            f"{pct(r.get('avg_module_reduction_ratio') or 0.0)} | {fmt(r['mar'])} | {fmt(r['mfr'])} |"
        )
    return "\n".join(lines)


def render_section_by_4x2(title: str, rows: Sequence[Dict[str, Any]], table_fn) -> List[str]:
    lines = [f"## {title}", ""]
    for level in LEVEL_ORDER:
        lines.append(f"### {level}-Level")
        lines.append("")
        for cat_id, cat_name in CATEGORY_NAMES.items():
            subset = [r for r in rows if r["level"] == level and r["category_id"] == cat_id]
            lines.append(f"#### {cat_id}. {cat_name}")
            lines.append("")
            lines.append(table_fn(subset))
            lines.append("")
    return lines


def main() -> None:
    effectiveness = load_effectiveness_rows()
    context = load_candidate_reduction_context()
    rq2 = build_rq2_rows(effectiveness, context)
    rq3 = build_rq3_rows(effectiveness)
    rq4 = build_rq4_rows(effectiveness, context)

    all_long: List[Dict[str, Any]] = []
    for r in effectiveness:
        all_long.append({"rq": "RQ1", **r})
    all_long.extend(rq2)
    all_long.extend(rq3)
    all_long.extend(rq4)

    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (TABLES / "all_rq_5x4x2_long.json").write_text(json.dumps(all_long, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(TABLES / "all_rq_5x4x2_long.csv", all_long)

    lines: List[str] = [
        "# All RQ Results: 5 Methods x 4 Datasets x 2 Levels",
        "",
        "本文件把所有 RQ 都展开到 `五类方法 × 四个数据集分类 × 文件级/语句级` 的粒度，便于论文写作和后续复制到表格中。",
        "",
        "说明：`Top-1 / Top-5 / Top-10` 显示为命中的 bug 数量，不显示百分比；CSV/JSON 中的 `top1 / top5 / top10` 也是整数命中数。",
        "",
        "说明：RQ4 当前统计的是候选空间/搜索空间效率；没有把未单独计时的 wall-clock runtime 写成实验结果。",
        "",
        "## Experiment Protocol",
        "",
        "| Item | Setting |",
        "| --- | --- |",
        "| Bug input | 仅使用 issue 首次 bug 报告，不使用讨论、修复说明或后续评论 |",
        "| Code input | 优先使用样本记录的 `Buggy SHA` 版本代码 |",
        "| File-level data | `ranker/data/manual_stage2_deep/` |",
        "| Statement-level data | `ranker/data/manual_line_stage3_light/` |",
        "| Methods | CrashLocCAE, LLM-Only, CrashLocator, Scaffle, w/o CAE Features |",
        "| Dataset categories | 1 FEM / Scientific Computing; 2 CFD / Flow Simulation; 3 Multiphysics Frameworks; 4 Modeling / Dynamics |",
        "",
    ]
    lines.extend(render_section_by_4x2("RQ1: Effectiveness Compared with Baselines", effectiveness, metric_table))
    lines.extend(render_section_by_4x2("RQ2: Module Reduction Effectiveness and Candidate-Space Reduction", rq2, rq2_table))
    lines.extend(render_section_by_4x2("RQ3: CAE-Specific Feature Contribution", rq3, rq3_table))
    lines.extend(render_section_by_4x2("RQ4: Efficiency / Search-Space Cost Compared with Baselines", rq4, rq4_table))
    lines.extend([
        "## Machine-Readable Files",
        "",
        "- Long CSV: `ranker/results/tables/all_rq_5x4x2_long.csv`",
        "- Long JSON: `ranker/results/tables/all_rq_5x4x2_long.json`",
        "- Main Markdown: `ranker/results/ALL_RQ_RESULTS.md`",
        "",
    ])

    md = "\n".join(lines)
    (RESULTS_ROOT / "ALL_RQ_RESULTS.md").write_text(md, encoding="utf-8")
    (REPORTS / "all_rq_5x4x2_results.md").write_text(md, encoding="utf-8")
    print(RESULTS_ROOT / "ALL_RQ_RESULTS.md")
    print(TABLES / "all_rq_5x4x2_long.csv")


if __name__ == "__main__":
    main()
