#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Collect current results into 5-method x 4-category x 2-level format."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Sequence, Tuple

CATEGORY_MAP = {
    "CalculiX-Examples": "1", "Calculix": "1", "calculix": "1",
    "deal.II": "1", "dealii": "1",
    "FireDrake": "1", "firedrake": "1",
    "FreeFem": "1", "FreeFEM": "1",
    "gridap": "1", "Gridap": "1",
    "MFEM": "1", "mfem": "1",
    "SfePy": "1", "sfepy": "1",
    "Sparselizard": "1", "sparselizard": "1",
    "code": "2", "Code Saturne": "2", "Code_Saturne": "2", "Code-Saturne": "2",
    "coolfluid": "2", "COOLFluiD": "2",
    "FDS": "2", "fds": "2",
    "Fluidity": "2",
    "Nek5000": "2",
    "SU2": "2", "su2": "2",
    "xcompact3d": "2", "Xcompact3d": "2",
    "Goma": "3", "Kratos": "3",
    "OpenModelica": "4", "ROSS": "4", "ross": "4",
}

CATEGORY_NAMES = {
    "1": "FEM / Scientific Computing",
    "2": "CFD / Flow Simulation",
    "3": "Multiphysics Frameworks",
    "4": "Modeling / Dynamics",
}

METHOD_ORDER = ["CrashLocCAE", "LLM-Only", "CrashLocator", "Scaffle", "w/o CAE Features"]
LEVEL_ORDER = ["File", "Statement"]

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "ranker" / "results"
TABLES = RESULTS_ROOT / "tables"
PRED = RESULTS_ROOT / "predictions"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def category_for(project: str) -> str:
    if project in CATEGORY_MAP:
        return CATEGORY_MAP[project]
    normalized = project.lower().replace("_", "").replace("-", "").replace(".", "").replace(" ", "")
    for name, cat in CATEGORY_MAP.items():
        key = name.lower().replace("_", "").replace("-", "").replace(".", "").replace(" ", "")
        if normalized == key:
            return cat
    return "unknown"


def compute_ranking_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    groups: Dict[str, List[Dict[str, float]]] = {}
    for row in rows:
        groups.setdefault(row["bug_id"], []).append(row)

    top1 = top3 = top5 = top10 = 0.0
    rr_total = 0.0
    fr_total = 0.0
    ar_total = 0.0
    valid_groups = 0

    for bug_id, items in groups.items():
        positives = sum(1 for x in items if x["label"] > 0.5)
        if positives == 0:
            continue
        valid_groups += 1
        ranked = sorted(items, key=lambda x: (-x["score"], x["file"]))
        positive_ranks = [idx + 1 for idx, x in enumerate(ranked) if x["label"] > 0.5]
        best_rank = min(positive_ranks)
        avg_rank = sum(positive_ranks) / len(positive_ranks)
        top1 += 1.0 if best_rank <= 1 else 0.0
        top3 += 1.0 if best_rank <= 3 else 0.0
        top5 += 1.0 if best_rank <= 5 else 0.0
        top10 += 1.0 if best_rank <= 10 else 0.0
        rr_total += 1.0 / best_rank
        fr_total += best_rank
        ar_total += avg_rank

    if valid_groups == 0:
        return {"top1": 0.0, "top3": 0.0, "top5": 0.0, "top10": 0.0, "mrr": 0.0, "mfr": 0.0, "mar": 0.0}
    return {
        "top1": top1 / valid_groups, "top3": top3 / valid_groups,
        "top5": top5 / valid_groups, "top10": top10 / valid_groups,
        "mrr": rr_total / valid_groups, "mfr": fr_total / valid_groups,
        "mar": ar_total / valid_groups,
    }


def collect_category_rows(level: str, method: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for cat_id, cat_name in CATEGORY_NAMES.items():
        cat_rows = [row for row in rows if category_for(row.get("project", "")) == cat_id]
        if not cat_rows:
            continue
        metrics = compute_ranking_metrics(cat_rows)
        positive_bugs = {row["bug_id"] for row in cat_rows if row["label"] > 0.5}
        num_bugs = len(positive_bugs)
        out.append({
            "level": level,
            "dataset": cat_name,
            "category_id": cat_id,
            "method": method,
            "top_unit": "count",
            "top1": int(round(metrics["top1"] * num_bugs)),
            "top5": int(round(metrics["top5"] * num_bugs)),
            "top10": int(round(metrics["top10"] * num_bugs)),
            "mar": round(metrics["mar"], 2),
            "mfr": round(metrics["mfr"], 2),
            "num_bugs": num_bugs,
            "num_pairs": len(cat_rows),
        })
    return out


def normalize_prediction_rows(rows: List[Dict[str, Any]], project_by_bug: Dict[str, str], method: str) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        out.append({
            "bug_id": row["bug_id"],
            "project": row.get("project") or project_by_bug.get(row["bug_id"], ""),
            "file": row["file"],
            "label": float(row["label"]),
            "score": float(row["score"]),
            "method": method,
        })
    return out


def bug_project_map(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    return {row["bug_id"]: row.get("project", "") for row in rows}


def rankings_to_scores(test_rows: List[Dict[str, Any]], ranking_file: Path, method: str) -> List[Dict[str, Any]]:
    predictions = {}
    for row in read_jsonl(ranking_file):
        predictions[row["bug_id"]] = row
    scored = []
    for row in test_rows:
        pred = predictions.get(row["bug_id"])
        if not pred:
            continue
        ranking = pred.get("ranking", [])
        score_map = {item: len(ranking) - idx for idx, item in enumerate(ranking)}
        scored.append({
            "bug_id": row["bug_id"],
            "project": row.get("project", ""),
            "file": row["file"],
            "label": float(row["label"]),
            "score": float(score_map.get(row["file"], 0.0)),
            "method": method,
        })
    return scored


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, int):
        return str(v)
    return f"{v:.2f}"


def pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def main():
    TABLES.mkdir(parents=True, exist_ok=True)

    file_test = read_jsonl(ROOT / "ranker" / "data" / "manual_stage2_deep" / "test.jsonl")
    statement_test = read_jsonl(ROOT / "ranker" / "data" / "manual_line_stage3_light" / "test.jsonl")
    statement_no_cae_test = read_jsonl(ROOT / "ranker" / "data" / "manual_line_stage3_no_cae" / "test.jsonl")
    file_project = bug_project_map(file_test)
    statement_project = bug_project_map(statement_test)
    statement_no_cae_project = bug_project_map(statement_no_cae_test)

    method_rows = {
        ("File", "CrashLocCAE"): normalize_prediction_rows(read_jsonl(PRED / "file_crashloccae.jsonl"), file_project, "CrashLocCAE"),
        ("File", "LLM-Only"): rankings_to_scores(file_test, PRED / "file_llm_only_rankings.jsonl", "LLM-Only"),
        ("File", "CrashLocator"): normalize_prediction_rows(read_jsonl(PRED / "file_crashlocator.jsonl"), file_project, "CrashLocator"),
        ("File", "Scaffle"): normalize_prediction_rows(read_jsonl(PRED / "file_scaffle.jsonl"), file_project, "Scaffle"),
        ("File", "w/o CAE Features"): normalize_prediction_rows(read_jsonl(PRED / "file_no_cae.jsonl"), file_project, "w/o CAE Features"),
        ("Statement", "CrashLocCAE"): normalize_prediction_rows(read_jsonl(PRED / "statement_crashloccae.jsonl"), statement_project, "CrashLocCAE"),
        ("Statement", "LLM-Only"): rankings_to_scores(statement_test, PRED / "statement_llm_only_rankings.jsonl", "LLM-Only"),
        ("Statement", "CrashLocator"): normalize_prediction_rows(read_jsonl(PRED / "statement_crashlocator.jsonl"), statement_project, "CrashLocator"),
        ("Statement", "Scaffle"): normalize_prediction_rows(read_jsonl(PRED / "statement_scaffle.jsonl"), statement_project, "Scaffle"),
        ("Statement", "w/o CAE Features"): normalize_prediction_rows(read_jsonl(PRED / "statement_no_cae.jsonl"), statement_no_cae_project, "w/o CAE Features"),
    }

    matrix = []
    for level in LEVEL_ORDER:
        for method in METHOD_ORDER:
            key = (level, method)
            if key in method_rows and method_rows[key]:
                matrix.extend(collect_category_rows(level, method, method_rows[key]))

    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "rq_matrix_current.json").write_text(json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8")

    # Build RQ tables
    lines = [
        "# All RQ Results: 5 Methods x 4 Datasets x 2 Levels",
        "",
        "Top-1 / Top-5 / Top-10 显示为命中的 bug 数量（整数），不显示百分比。",
        "",
        "## Experiment Protocol",
        "",
        "| Item | Setting |",
        "| --- | --- |",
        "| Bug input | 仅使用 issue 首次 bug 报告 |",
        "| Code input | 优先使用 Buggy SHA 版本代码 |",
        "| File-level data | `ranker/data/manual_stage2_deep/` |",
        "| Statement-level data | `ranker/data/manual_line_stage3_light/` |",
        "| Methods | CrashLocCAE, LLM-Only, CrashLocator, Scaffle, w/o CAE Features |",
        "| Dataset categories | 1 FEM / Scientific Computing; 2 CFD / Flow Simulation; 3 Multiphysics Frameworks; 4 Modeling / Dynamics |",
        "",
    ]

    # RQ1
    lines.append("## RQ1: Effectiveness Compared with Baselines")
    lines.append("")
    for level in LEVEL_ORDER:
        lines.append(f"### {level}-Level")
        lines.append("")
        for cat_id, cat_name in CATEGORY_NAMES.items():
            subset = [r for r in matrix if r["level"] == level and r["category_id"] == cat_id]
            if not subset:
                continue
            lines.append(f"#### {cat_id}. {cat_name}")
            lines.append("")
            lines.append("| Method | Top-1 (#) | Top-5 (#) | Top-10 (#) | MAR | MFR |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
            by_method = {r["method"]: r for r in subset}
            for method in METHOD_ORDER:
                if method in by_method:
                    r = by_method[method]
                    lines.append(f"| {method} | {r['top1']} | {r['top5']} | {r['top10']} | {fmt(r['mar'])} | {fmt(r['mfr'])} |")
            lines.append("")

    # RQ2: with candidate reduction context
    lines.append("## RQ2: Module Reduction Effectiveness and Candidate-Space Reduction")
    lines.append("")

    # Load reduction context
    reduction_rows = read_jsonl(TABLES / "candidate_reduction_rows.jsonl")
    test_reduction = [r for r in reduction_rows if r.get("split") == "test"] or reduction_rows
    reduction_by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for r in test_reduction:
        cat = category_for(r.get("project", ""))
        if cat in CATEGORY_NAMES:
            reduction_by_cat[cat].append(r)

    reduction_ctx: Dict[str, Dict] = {}
    for cat, items in reduction_by_cat.items():
        reduction_ctx[cat] = {
            "avg_repo_files": mean([float(r["repo_files"]) for r in items]),
            "avg_module_candidates": mean([float(r["module_candidates"]) for r in items]),
            "avg_reduction_ratio": mean([float(r["reduction_ratio"]) for r in items]),
        }

    for level in LEVEL_ORDER:
        lines.append(f"### {level}-Level")
        lines.append("")
        for cat_id, cat_name in CATEGORY_NAMES.items():
            subset = [r for r in matrix if r["level"] == level and r["category_id"] == cat_id]
            if not subset:
                continue
            lines.append(f"#### {cat_id}. {cat_name}")
            lines.append("")
            lines.append("| Method | Top-1 (#) | Top-5 (#) | Top-10 (#) | MAR | MFR | Bugs | Pairs | Avg. Candidate Units/Bug | Avg. Repo Files | Avg. Module Candidates | Reduction |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            by_method = {r["method"]: r for r in subset}
            ctx = reduction_ctx.get(cat_id, {})
            for method in METHOD_ORDER:
                if method not in by_method:
                    continue
                r = by_method[method]
                avg_cand = r["num_pairs"] / r["num_bugs"] if r["num_bugs"] else 0
                lines.append(
                    f"| {method} | {r['top1']} | {r['top5']} | {r['top10']} | {fmt(r['mar'])} | {fmt(r['mfr'])} | "
                    f"{r['num_bugs']} | {r['num_pairs']} | {fmt(avg_cand)} | "
                    f"{fmt(ctx.get('avg_repo_files'))} | {fmt(ctx.get('avg_module_candidates'))} | "
                    f"{pct(ctx.get('avg_reduction_ratio', 0))} |"
                )
            lines.append("")

    # RQ3: CAE feature contribution
    lines.append("## RQ3: CAE-Specific Feature Contribution")
    lines.append("")
    for level in LEVEL_ORDER:
        lines.append(f"### {level}-Level")
        lines.append("")
        for cat_id, cat_name in CATEGORY_NAMES.items():
            subset = [r for r in matrix if r["level"] == level and r["category_id"] == cat_id]
            if not subset:
                continue
            lines.append(f"#### {cat_id}. {cat_name}")
            lines.append("")
            lines.append("| Method | Top-1 (#) | Top-5 (#) | Top-10 (#) | MAR | MFR | Delta Top-1 (#) vs w/o CAE | Delta MAR vs w/o CAE |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            by_method = {r["method"]: r for r in subset}
            no_cae = by_method.get("w/o CAE Features")
            for method in METHOD_ORDER:
                if method not in by_method:
                    continue
                r = by_method[method]
                d_top1 = (r["top1"] - no_cae["top1"]) if no_cae else 0
                d_mar = (r["mar"] - no_cae["mar"]) if no_cae else 0
                lines.append(
                    f"| {method} | {r['top1']} | {r['top5']} | {r['top10']} | {fmt(r['mar'])} | {fmt(r['mfr'])} | "
                    f"{d_top1} | {fmt(d_mar)} |"
                )
            lines.append("")

    # RQ4: efficiency
    lines.append("## RQ4: Efficiency / Search-Space Cost Compared with Baselines")
    lines.append("")
    for level in LEVEL_ORDER:
        lines.append(f"### {level}-Level")
        lines.append("")
        for cat_id, cat_name in CATEGORY_NAMES.items():
            subset = [r for r in matrix if r["level"] == level and r["category_id"] == cat_id]
            if not subset:
                continue
            lines.append(f"#### {cat_id}. {cat_name}")
            lines.append("")
            lines.append("| Method | Bugs | Pairs | Avg. Candidate Units/Bug | Avg. Repo Files | Avg. Module Candidates | Reduction | MAR | MFR |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            by_method = {r["method"]: r for r in subset}
            ctx = reduction_ctx.get(cat_id, {})
            for method in METHOD_ORDER:
                if method not in by_method:
                    continue
                r = by_method[method]
                avg_cand = r["num_pairs"] / r["num_bugs"] if r["num_bugs"] else 0
                lines.append(
                    f"| {method} | {r['num_bugs']} | {r['num_pairs']} | {fmt(avg_cand)} | "
                    f"{fmt(ctx.get('avg_repo_files'))} | {fmt(ctx.get('avg_module_candidates'))} | "
                    f"{pct(ctx.get('avg_reduction_ratio', 0))} | {fmt(r['mar'])} | {fmt(r['mfr'])} |"
                )
            lines.append("")

    md = "\n".join(lines)
    (RESULTS_ROOT / "ALL_RQ_RESULTS.md").write_text(md, encoding="utf-8")
    print(f"[OK] Written to {RESULTS_ROOT / 'ALL_RQ_RESULTS.md'}")


if __name__ == "__main__":
    main()
