#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Collect the 5-method x 4-category x 2-granularity result matrix."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from train_ranker import compute_ranking_metrics


CATEGORY_MAP = {
    "CalculiX-Examples": "1",
    "Calculix": "1",
    "deal.II": "1",
    "dealii": "1",
    "FireDrake": "1",
    "firedrake": "1",
    "FreeFem": "1",
    "FreeFEM": "1",
    "Gridap": "1",
    "gridap": "1",
    "MFEM": "1",
    "mfem": "1",
    "SfePy": "1",
    "Sparselizard": "1",
    "code": "2",
    "Code Saturne": "2",
    "Code_Saturne": "2",
    "Code-Saturne": "2",
    "COOLFluiD": "2",
    "coolfluid": "2",
    "FDS": "2",
    "fds": "2",
    "Fluidity": "2",
    "Nek5000": "2",
    "SU2": "2",
    "su2": "2",
    "Xcompact3d": "2",
    "Goma": "3",
    "Kratos": "3",
    "OpenModelica": "4",
    "ROSS": "4",
    "ross": "4",
}

CATEGORY_NAMES = {
    "1": "FEM / Scientific Computing",
    "2": "CFD / Flow Simulation",
    "3": "Multiphysics Frameworks",
    "4": "Modeling / Dynamics",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def category_for(project: str) -> str:
    if project in CATEGORY_MAP:
        return CATEGORY_MAP[project]
    normalized = project.lower().replace("_", "").replace("-", "").replace(".", "").replace(" ", "")
    for name, cat in CATEGORY_MAP.items():
        if normalized == name.lower().replace("_", "").replace("-", "").replace(".", "").replace(" ", ""):
            return cat
    return "unknown"


def bug_project_map(rows: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    return {row["bug_id"]: row.get("project", "") for row in rows}


def rankings_to_scores(test_rows: Sequence[Dict[str, Any]], ranking_file: Path, method: str) -> List[Dict[str, Any]]:
    predictions = {row["bug_id"]: row for row in read_jsonl(ranking_file)}
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


def normalize_prediction_rows(rows: Sequence[Dict[str, Any]], project_by_bug: Dict[str, str], method: str) -> List[Dict[str, Any]]:
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


def collect_category_rows(level: str, method: str, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for cat_id, cat_name in CATEGORY_NAMES.items():
        cat_rows = [row for row in rows if category_for(row.get("project", "")) == cat_id]
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
            "mar": metrics["mar"],
            "mfr": metrics["mfr"],
            "num_bugs": num_bugs,
            "num_pairs": len(cat_rows),
        })
    return out


def pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def markdown_by_level(rows: Sequence[Dict[str, Any]]) -> str:
    lines = ["# RQ Matrix Results", ""]
    for level in ["File", "Statement"]:
        lines.append(f"## {level}-Level")
        lines.append("")
        for cat_id in ["1", "2", "3", "4"]:
            cat_name = CATEGORY_NAMES[cat_id]
            lines.append(f"### {cat_id}. {cat_name}")
            lines.append("")
            lines.append("| Method | Top-1 (#) | Top-5 (#) | Top-10 (#) | MAR | MFR |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
            for row in rows:
                if row["level"] == level and row["category_id"] == cat_id:
                    lines.append(
                        f"| {row['method']} | {row['top1']} | {row['top5']} | {row['top10']} | "
                        f"{row['mar']:.2f} | {row['mfr']:.2f} |"
                    )
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    root = Path("ranker")
    results_root = root / "results"
    pred_dir = results_root / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    archive = root / "archive" / "outputs_before_results"
    copy_sources = {
        "file_crashloccae.jsonl": archive / "wide_deep_seed" / "test_predictions.jsonl",
        "file_no_cae.jsonl": archive / "wide_deep_no_cae" / "test_predictions.jsonl",
        "file_crashlocator.jsonl": archive / "comparison_predictions" / "crashlocator_style.jsonl",
        "file_scaffle.jsonl": archive / "comparison_predictions" / "scaffle_style.jsonl",
    }
    for name, src in copy_sources.items():
        dst = pred_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    file_test = read_jsonl(root / "data" / "manual_stage2_deep" / "test.jsonl")
    statement_test = read_jsonl(root / "data" / "manual_line_stage3_light" / "test.jsonl")
    statement_no_cae_test = read_jsonl(root / "data" / "manual_line_stage3_no_cae" / "test.jsonl")
    file_project = bug_project_map(file_test)
    statement_project = bug_project_map(statement_test)
    statement_no_cae_project = bug_project_map(statement_no_cae_test)

    method_rows = {
        ("File", "CrashLocCAE"): normalize_prediction_rows(read_jsonl(pred_dir / "file_crashloccae.jsonl"), file_project, "CrashLocCAE"),
        ("File", "LLM-Only"): rankings_to_scores(file_test, pred_dir / "file_llm_only_rankings.jsonl", "LLM-Only"),
        ("File", "CrashLocator"): normalize_prediction_rows(read_jsonl(pred_dir / "file_crashlocator.jsonl"), file_project, "CrashLocator"),
        ("File", "Scaffle"): normalize_prediction_rows(read_jsonl(pred_dir / "file_scaffle.jsonl"), file_project, "Scaffle"),
        ("File", "w/o CAE Features"): normalize_prediction_rows(read_jsonl(pred_dir / "file_no_cae.jsonl"), file_project, "w/o CAE Features"),
        ("Statement", "CrashLocCAE"): normalize_prediction_rows(read_jsonl(pred_dir / "statement_crashloccae.jsonl"), statement_project, "CrashLocCAE"),
        ("Statement", "LLM-Only"): rankings_to_scores(statement_test, pred_dir / "statement_llm_only_rankings.jsonl", "LLM-Only"),
        ("Statement", "CrashLocator"): normalize_prediction_rows(read_jsonl(pred_dir / "statement_crashlocator.jsonl"), statement_project, "CrashLocator"),
        ("Statement", "Scaffle"): normalize_prediction_rows(read_jsonl(pred_dir / "statement_scaffle.jsonl"), statement_project, "Scaffle"),
        ("Statement", "w/o CAE Features"): normalize_prediction_rows(read_jsonl(pred_dir / "statement_no_cae.jsonl"), statement_no_cae_project, "w/o CAE Features"),
    }

    matrix = []
    method_order = ["CrashLocCAE", "LLM-Only", "CrashLocator", "Scaffle", "w/o CAE Features"]
    for level in ["File", "Statement"]:
        for method in method_order:
            matrix.extend(collect_category_rows(level, method, method_rows[(level, method)]))

    out_json = results_root / "tables" / "rq_matrix.json"
    out_csv = results_root / "tables" / "rq_matrix.csv"
    out_md = results_root / "reports" / "rq_matrix.md"
    out_json.write_text(json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_csv, matrix)
    out_md.write_text(markdown_by_level(matrix), encoding="utf-8")
    print(out_md)


if __name__ == "__main__":
    main()
