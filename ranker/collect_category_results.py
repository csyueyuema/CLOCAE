#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Collect file-level results by CAE project category.

The project categories follow the paper table:
1. FEM / Scientific Computing
2. CFD / Flow Simulation
3. Multiphysics Frameworks
4. Modeling / Dynamics
"""

from __future__ import annotations

import argparse
import csv
import json
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

CATEGORY_PROJECTS = {
    "1": ["Calculix", "deal.II", "FireDrake", "FreeFEM", "Gridap", "MFEM", "SfePy", "Sparselizard"],
    "2": ["Code Saturne", "COOLFluiD", "FDS", "Fluidity", "Nek5000", "SU2", "Xcompact3d"],
    "3": ["Goma", "Kratos"],
    "4": ["OpenModelica", "ROSS"],
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def category_for(project: str) -> str:
    if project in CATEGORY_MAP:
        return CATEGORY_MAP[project]
    normalized = project.lower().replace("_", "").replace("-", "").replace(".", "")
    for name, cat in CATEGORY_MAP.items():
        if normalized == name.lower().replace("_", "").replace("-", "").replace(".", ""):
            return cat
    return "unknown"


def format_method(label: str) -> str:
    return {
        "CrashLocCAE": "CrashLocCAE",
        "w/o CAE Features": "w/o CAE Features",
        "CrashLocator-style": "CrashLocator-style",
        "Scaffle-style": "Scaffle-style",
        "Stack Only": "Stack Only",
        "CAE Domain Only": "CAE Domain Only",
        "Heuristic": "Heuristic",
    }.get(label, label)


def load_model_predictions(metrics_file: Path) -> List[Dict[str, Any]]:
    data = read_json(metrics_file)
    if "predictions" in data:
        return data["predictions"]
    rows_file = metrics_file.with_name("test_predictions.jsonl")
    if rows_file.exists():
        return read_jsonl(rows_file)
    raise FileNotFoundError(f"Missing prediction rows for {metrics_file}. Run eval_ranker.py with prediction output support.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="ranker/data/manual_stage2_deep/test.jsonl")
    parser.add_argument("--main-metrics", default="ranker/outputs/wide_deep/test_metrics.json")
    parser.add_argument("--no-cae-metrics", default="ranker/outputs/wide_deep_no_cae/test_metrics.json")
    parser.add_argument("--comparison-dir", default="ranker/outputs/comparison_predictions")
    parser.add_argument("--output-json", default="ranker/outputs/category_results.json")
    parser.add_argument("--output-csv", default="ranker/outputs/category_results.csv")
    parser.add_argument("--output-md", default="ranker/outputs/category_results.md")
    args = parser.parse_args()

    test_rows = read_jsonl(Path(args.test_file))
    project_by_bug = {}
    for row in test_rows:
        project_by_bug[row["bug_id"]] = row.get("project", "")

    method_sources = [
        ("CrashLocCAE", load_model_predictions(Path(args.main_metrics))),
        ("w/o CAE Features", load_model_predictions(Path(args.no_cae_metrics))),
        ("CrashLocator-style", read_jsonl(Path(args.comparison_dir) / "crashlocator_style.jsonl")),
        ("Scaffle-style", read_jsonl(Path(args.comparison_dir) / "scaffle_style.jsonl")),
        ("Stack Only", read_jsonl(Path(args.comparison_dir) / "stack_only.jsonl")),
        ("CAE Domain Only", read_jsonl(Path(args.comparison_dir) / "domain_only.jsonl")),
        ("Heuristic", read_jsonl(Path(args.comparison_dir) / "heuristic.jsonl")),
    ]

    result_rows: List[Dict[str, Any]] = []
    for method, pred_rows in method_sources:
        by_cat: Dict[str, List[Dict[str, Any]]] = {cat: [] for cat in CATEGORY_NAMES}
        for row in pred_rows:
            project = row.get("project") or project_by_bug.get(row["bug_id"], "")
            cat = category_for(project)
            if cat in by_cat:
                by_cat[cat].append(
                    {
                        "bug_id": row["bug_id"],
                        "file": row["file"],
                        "label": float(row["label"]),
                        "score": float(row["score"]),
                    }
                )
        for cat in ["1", "2", "3", "4"]:
            rows = by_cat[cat]
            metrics = compute_ranking_metrics(rows)
            positive_bug_ids = {row["bug_id"] for row in rows if row["label"] > 0.5}
            result_rows.append(
                {
                    "method": format_method(method),
                    "category_id": cat,
                    "category": CATEGORY_NAMES[cat],
                    "projects": ", ".join(CATEGORY_PROJECTS[cat]),
                    "num_bugs": len(positive_bug_ids),
                    "num_pairs": len(rows),
                    "top1": metrics["top1"],
                    "top5": metrics["top5"],
                    "top10": metrics["top10"],
                    "mar": metrics["mar"],
                    "mfr": metrics["mfr"],
                }
            )

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(Path(args.output_csv), result_rows)

    md_lines = [
        "# Category Results",
        "",
        "Project categories follow the paper dataset table.",
        "",
        "| Method | Cat. | Category | Bugs | Top-1 | Top-5 | Top-10 | MAR | MFR |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result_rows:
        md_lines.append(
            f"| {row['method']} | {row['category_id']} | {row['category']} | {row['num_bugs']} | "
            f"{row['top1'] * 100:.2f}% | {row['top5'] * 100:.2f}% | {row['top10'] * 100:.2f}% | "
            f"{row['mar']:.2f} | {row['mfr']:.2f} |"
        )
    Path(args.output_md).write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps(result_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
