#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from train_ranker import compute_ranking_metrics


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_row(row: Dict[str, Any], baseline: str) -> float:
    feats = row.get("features", {})
    if baseline == "stack_only":
        return 10.0 * float(feats.get("stack_file_exact_match", 0.0)) + float(feats.get("stack_file_suffix_match", 0.0))
    if baseline == "domain_only":
        return (
            float(feats.get("symptom_overlap_count", 0.0))
            + float(feats.get("domain_keyword_overlap", 0.0))
            + float(feats.get("teacher_module_prob", 0.0))
        )
    if baseline == "heuristic":
        return float(row.get("heuristic_score", 0.0))
    raise ValueError(f"Unknown baseline: {baseline}")


def evaluate(path: Path, baseline: str) -> Dict[str, Any]:
    rows = []
    for row in read_jsonl(path):
        rows.append({
            "bug_id": row["bug_id"],
            "file": row["file"],
            "label": float(row["label"]),
            "score": score_row(row, baseline),
        })
    metrics = compute_ranking_metrics(rows)
    return {
        "baseline": baseline,
        **metrics,
        "num_pairs": len(rows),
        "num_bugs": len({row["bug_id"] for row in rows}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="ranker/data/manual_stage2_deep/test.jsonl")
    parser.add_argument("--output-file", default="ranker/outputs/feature_baselines.json")
    args = parser.parse_args()

    baselines = ["stack_only", "domain_only", "heuristic"]
    rows = [evaluate(Path(args.test_file), baseline) for baseline in baselines]
    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
