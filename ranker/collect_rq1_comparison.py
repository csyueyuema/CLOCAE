#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Collect RQ1/RQ3 comparison results into one JSON/CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(label: str, source: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "label": label,
        "source": source,
        "top1": metrics["top1"],
        "top3": metrics.get("top3", 0.0),
        "top5": metrics["top5"],
        "top10": metrics["top10"],
        "mrr": metrics["mrr"],
        "mfr": metrics["mfr"],
        "mar": metrics["mar"],
        "num_pairs": metrics.get("num_pairs"),
        "num_bugs": metrics.get("num_bugs"),
    }


def from_eval_json(path: Path, label: str, source: str) -> Dict[str, Any]:
    data = load_json(path)
    metrics = dict(data["metrics"])
    metrics["num_pairs"] = data.get("num_pairs")
    metrics["num_bugs"] = data.get("num_bugs")
    return metric_row(label, source, metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-file", default="ranker/outputs/wide_deep/test_metrics.json")
    parser.add_argument("--no-cae-file", default="ranker/outputs/wide_deep_no_cae/test_metrics.json")
    parser.add_argument("--comparison-file", default="ranker/outputs/comparison_methods.json")
    parser.add_argument("--llm-file", default="ranker/outputs/llm_only_metrics.json")
    parser.add_argument("--output-json", default="ranker/outputs/rq1_comparison_summary.json")
    parser.add_argument("--output-csv", default="ranker/outputs/rq1_comparison_summary.csv")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = [
        from_eval_json(Path(args.main_file), "CrashLocCAE", "wide_deep"),
        from_eval_json(Path(args.no_cae_file), "w/o CAE Features", "wide_deep_no_cae"),
    ]

    name_map = {
        "crashlocator_style": "CrashLocator-style",
        "scaffle_style": "Scaffle-style",
        "stack_only": "Stack Only",
        "domain_only": "CAE Domain Only",
        "heuristic": "Heuristic",
    }
    for item in load_json(Path(args.comparison_file)):
        rows.append(metric_row(name_map.get(item["method"], item["method"]), item["method"], item))

    llm_path = Path(args.llm_file)
    if llm_path.exists():
        llm = load_json(llm_path)
        rows.append(metric_row("LLM-Only", "llm_only", llm))

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
