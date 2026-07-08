#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate statement-level comparison methods on the line ranking data."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from train_line_ranker import compute_ranking_metrics


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def feat(row: Dict[str, Any], key: str) -> float:
    try:
        return float(row.get("features", {}).get(key, 0.0))
    except Exception:
        return 0.0


def score_line(row: Dict[str, Any], method: str) -> float:
    if method == "crashlocator":
        return (
            8.0 * feat(row, "stack_file_match")
            + 2.0 * feat(row, "line_token_overlap")
            + 1.0 * feat(row, "context_token_overlap")
            - 0.2 * feat(row, "is_comment_line")
        )
    if method == "scaffle":
        return (
            2.5 * feat(row, "context_token_overlap")
            + 1.8 * feat(row, "line_token_overlap")
            + 1.0 * feat(row, "stack_file_match")
            + 0.05 * min(feat(row, "line_length"), 120.0)
            - 0.2 * feat(row, "is_comment_line")
        )
    raise ValueError(f"Unknown line comparison method: {method}")


def rank_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["bug_id"]].append(dict(row))
    ranked_rows = []
    for _, items in groups.items():
        ranked = sorted(items, key=lambda x: (-float(x["score"]), x["file"]))
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
            ranked_rows.append(row)
    return ranked_rows


def evaluate(rows: Sequence[Dict[str, Any]], method: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    scored = []
    for row in rows:
        scored.append({
            "bug_id": row["bug_id"],
            "project": row.get("project", ""),
            "file": row["file"],
            "label": float(row["label"]),
            "score": score_line(row, method),
            "method": method,
        })
    metrics = compute_ranking_metrics(scored)
    return {
        "method": method,
        **metrics,
        "num_pairs": len(scored),
        "num_bugs": len({row["bug_id"] for row in scored}),
    }, rank_rows(scored)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="ranker/data/manual_line_stage3_light/test.jsonl")
    parser.add_argument("--output-file", default="ranker/results/tables/statement_comparison_methods.json")
    parser.add_argument("--predictions-dir", default="ranker/results/predictions")
    parser.add_argument("--methods", nargs="+", default=["crashlocator", "scaffle"])
    args = parser.parse_args()

    rows = read_jsonl(Path(args.test_file))
    results = []
    pred_dir = Path(args.predictions_dir)
    for method in args.methods:
        metrics, ranked = evaluate(rows, method)
        results.append(metrics)
        write_jsonl(pred_dir / f"statement_{method}.jsonl", ranked)

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
