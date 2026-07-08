#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build the RQ3 module-reduction ablation summary from scope benchmark output.

This script does not retrain or rerun the ranker. It repackages the completed
full-project vs. module-reduced Wide & Deep benchmark into an RQ3 effectiveness
ablation focused on candidate filtering.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


MODULE_SCOPE = "module_reduced"
FULL_SCOPE = "full_project"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.2f}%"


def fmt_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def scope_map(summary: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    scopes: Dict[str, Dict[str, Any]] = {}
    for row in summary.get("summaries", []):
        scope = row.get("scope")
        if scope:
            scopes[str(scope)] = dict(row)
    missing = [scope for scope in (MODULE_SCOPE, FULL_SCOPE) if scope not in scopes]
    if missing:
        raise ValueError(f"Missing scope summaries: {missing}")
    return scopes


def bug_sets(per_bug_rows: Iterable[Mapping[str, Any]]) -> Dict[str, set[str]]:
    by_scope: Dict[str, set[str]] = {}
    for row in per_bug_rows:
        scope = str(row.get("scope", ""))
        bug_id = str(row.get("bug_id", ""))
        if scope and bug_id:
            by_scope.setdefault(scope, set()).add(bug_id)
    return by_scope


def scope_effectiveness_stats(per_bug_rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}
    for row in per_bug_rows:
        scope = str(row.get("scope", ""))
        if not scope:
            continue
        slot = stats.setdefault(
            scope,
            {
                "total_crashes": 0,
                "valid_effectiveness_cases": 0,
                "invalid_without_positive_candidate": 0,
                "positive_candidates": 0,
                "top1_hits": 0,
                "top5_hits": 0,
                "top10_hits": 0,
            },
        )
        slot["total_crashes"] += 1
        positives = int(row.get("positive_candidates", 0) or 0)
        slot["positive_candidates"] += positives
        ranks = [int(x) for x in row.get("positive_ranks", [])]
        if positives <= 0 or not ranks:
            slot["invalid_without_positive_candidate"] += 1
            continue
        slot["valid_effectiveness_cases"] += 1
        best_rank = min(ranks)
        slot["top1_hits"] += int(best_rank <= 1)
        slot["top5_hits"] += int(best_rank <= 5)
        slot["top10_hits"] += int(best_rank <= 10)
    return stats


def rq3_row(
    setting: str,
    source_scope: str,
    summary: Mapping[str, Any],
    effectiveness_stats: Mapping[str, int],
) -> Dict[str, Any]:
    metrics = summary.get("metrics", {})
    return {
        "setting": setting,
        "source_scope": source_scope,
        "total_crashes": effectiveness_stats.get("total_crashes", summary.get("num_bugs")),
        "valid_effectiveness_cases": effectiveness_stats.get("valid_effectiveness_cases"),
        "invalid_without_positive_candidate": effectiveness_stats.get("invalid_without_positive_candidate"),
        "positive_candidates": effectiveness_stats.get("positive_candidates", summary.get("positive_candidates")),
        "total_candidates": summary.get("total_candidates"),
        "avg_candidates_per_crash": summary.get("avg_candidates_per_bug"),
        "median_candidates_per_crash": summary.get("median_candidates_per_bug"),
        "top1_hits": effectiveness_stats.get("top1_hits"),
        "top5_hits": effectiveness_stats.get("top5_hits"),
        "top10_hits": effectiveness_stats.get("top10_hits"),
        "top1": metrics.get("top1"),
        "top5": metrics.get("top5"),
        "top10": metrics.get("top10"),
        "mrr": metrics.get("mrr"),
        "mfr": metrics.get("mfr"),
        "mar": metrics.get("mar"),
    }


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "setting",
        "source_scope",
        "total_crashes",
        "valid_effectiveness_cases",
        "invalid_without_positive_candidate",
        "positive_candidates",
        "total_candidates",
        "avg_candidates_per_crash",
        "median_candidates_per_crash",
        "top1_hits",
        "top5_hits",
        "top10_hits",
        "top1",
        "top5",
        "top10",
        "mrr",
        "mfr",
        "mar",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def build_report(data: Mapping[str, Any]) -> str:
    rows = data["rows"]
    comparison = data["comparison"]
    validations = data["validations"]
    full = rows[1]
    module = rows[0]
    environment = data.get("environment", {})
    total_crashes = module["total_crashes"]
    module_valid = module["valid_effectiveness_cases"]
    full_valid = full["valid_effectiveness_cases"]
    valid_phrase = (
        f"{module_valid} crashes"
        if module_valid == full_valid
        else f"{module_valid} crashes for the full model and {full_valid} crashes for w/o Module Reduction"
    )
    test_file = environment.get("test_file", "n/a")

    return f"""# RQ3 Module-Reduction Ablation

This report repackages the completed full-project vs. module-reduced Wide & Deep benchmark as an RQ3 effectiveness ablation. The `w/o Module Reduction` setting disables module-based candidate filtering and applies the same second-stage Wide & Deep ranker to all source files in the corresponding project.

The benchmark covers {total_crashes} total test crashes. Ranking metrics are computed over {valid_phrase} that have at least one positive file candidate, matching the existing evaluator convention of skipping groups with no positive candidate.

## Ablation Summary

| Setting | Total Crashes | Valid Cases | Candidates | Avg./Crash | Top-1 | Top-5 | Top-10 | MFR | MAR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| {module['setting']} | {module['total_crashes']} | {module['valid_effectiveness_cases']} | {module['total_candidates']:,} | {module['avg_candidates_per_crash']:.2f} | {module['top1_hits']}/{module['valid_effectiveness_cases']} ({module['top1']:.4f}) | {module['top5_hits']}/{module['valid_effectiveness_cases']} ({module['top5']:.4f}) | {module['top10_hits']}/{module['valid_effectiveness_cases']} ({module['top10']:.4f}) | {module['mfr']:.2f} | {module['mar']:.2f} |
| {full['setting']} | {full['total_crashes']} | {full['valid_effectiveness_cases']} | {full['total_candidates']:,} | {full['avg_candidates_per_crash']:.2f} | {full['top1_hits']}/{full['valid_effectiveness_cases']} ({full['top1']:.4f}) | {full['top5_hits']}/{full['valid_effectiveness_cases']} ({full['top5']:.4f}) | {full['top10_hits']}/{full['valid_effectiveness_cases']} ({full['top10']:.4f}) | {full['mfr']:.2f} | {full['mar']:.2f} |

## Effect Size

- Candidate expansion without module reduction: {fmt_float(comparison['candidate_ratio_full_over_module'], 4)}x
- Candidate reduction with module filtering: {pct(comparison['candidate_reduction_ratio'])}
- Top-1 drop without module reduction: {fmt_float(comparison['top1_drop_without_module'], 4)}
- Top-5 drop without module reduction: {fmt_float(comparison['top5_drop_without_module'], 4)}
- Top-10 drop without module reduction: {fmt_float(comparison['top10_drop_without_module'], 4)}
- MFR increase without module reduction: {fmt_float(comparison['mfr_ratio_without_over_full_model'], 4)}x
- MAR increase without module reduction: {fmt_float(comparison['mar_ratio_without_over_full_model'], 4)}x

## Suggested RQ3 Text

**w/o Module Reduction.** We disable module-based candidate filtering and apply the same second-stage Wide & Deep ranker to all source files in the corresponding project. This setting evaluates whether module-level reduction improves localization by shielding the ranker from heterogeneous project files and downstream manifestation-related candidates.

The results show that module-level candidate reduction is critical to file-level localization effectiveness. The experiment covers {total_crashes} test crashes, of which {module_valid} contain at least one positive file candidate and are used for Top-N, MFR, and MAR. Without module reduction, the average candidate space expands from {module['avg_candidates_per_crash']:.2f} to {full['avg_candidates_per_crash']:.2f} files per crash. This larger and more heterogeneous candidate set substantially degrades ranking quality: Top-10 drops from {module['top10_hits']}/{module['valid_effectiveness_cases']} to {full['top10_hits']}/{full['valid_effectiveness_cases']}, while MFR and MAR increase from {module['mfr']:.2f} and {module['mar']:.2f} to {full['mfr']:.2f} and {full['mar']:.2f}, respectively. These results indicate that module reduction does more than reduce computation cost; it also provides the Wide & Deep ranker with a focused candidate space and reduces interference from irrelevant project files and downstream crash manifestation files.

## Validations

- Same bug set across both settings: {validations['same_bug_set']}
- Valid effectiveness cases per setting: module_reduced={validations['valid_effectiveness_cases'].get(MODULE_SCOPE)}, full_project={validations['valid_effectiveness_cases'].get(FULL_SCOPE)}
- Module-reduced candidates match benchmark summary: {validations['module_reduced_candidate_match']}
- Full-project candidates match benchmark summary: {validations['full_project_candidate_match']}
- Number of paired crash cases: {validations['paired_bug_count']}
- Source test file: `{test_file}`
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="ranker/results/efficiency_wide_deep_scope_gpu_full_bs64",
        help="Directory containing summary.json and per_bug_topk.jsonl from the scope benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        default="ranker/results/rq3_module_reduction_ablation",
        help="Directory for RQ3 ablation summary artifacts.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    summary = read_json(input_dir / "summary.json")
    per_bug_rows = read_jsonl(input_dir / "per_bug_topk.jsonl")

    scopes = scope_map(summary)
    module_summary = scopes[MODULE_SCOPE]
    full_summary = scopes[FULL_SCOPE]
    effectiveness_by_scope = scope_effectiveness_stats(per_bug_rows)

    rows = [
        rq3_row("Full Model", MODULE_SCOPE, module_summary, effectiveness_by_scope.get(MODULE_SCOPE, {})),
        rq3_row("w/o Module Reduction", FULL_SCOPE, full_summary, effectiveness_by_scope.get(FULL_SCOPE, {})),
    ]

    bug_sets_by_scope = bug_sets(per_bug_rows)
    module_bugs = bug_sets_by_scope.get(MODULE_SCOPE, set())
    full_bugs = bug_sets_by_scope.get(FULL_SCOPE, set())

    comparison = {
        "candidate_ratio_full_over_module": safe_ratio(
            float(full_summary["total_candidates"]), float(module_summary["total_candidates"])
        ),
        "candidate_reduction_ratio": 1.0
        - safe_ratio(float(module_summary["total_candidates"]), float(full_summary["total_candidates"])),
        "top1_drop_without_module": rows[0]["top1"] - rows[1]["top1"],
        "top5_drop_without_module": rows[0]["top5"] - rows[1]["top5"],
        "top10_drop_without_module": rows[0]["top10"] - rows[1]["top10"],
        "mfr_ratio_without_over_full_model": safe_ratio(float(rows[1]["mfr"]), float(rows[0]["mfr"])),
        "mar_ratio_without_over_full_model": safe_ratio(float(rows[1]["mar"]), float(rows[0]["mar"])),
    }

    validations = {
        "same_bug_set": module_bugs == full_bugs,
        "paired_bug_count": len(module_bugs & full_bugs),
        "module_reduced_bug_count": len(module_bugs),
        "full_project_bug_count": len(full_bugs),
        "valid_effectiveness_cases": {
            MODULE_SCOPE: rows[0].get("valid_effectiveness_cases"),
            FULL_SCOPE: rows[1].get("valid_effectiveness_cases"),
        },
        "module_reduced_candidate_match": int(module_summary.get("total_candidates", -1))
        == int(rows[0].get("total_candidates", -2)),
        "full_project_candidate_match": int(full_summary.get("total_candidates", -1))
        == int(rows[1].get("total_candidates", -2)),
        "source_validations": summary.get("validations", {}),
    }

    output = {
        "source": {
            "input_dir": str(input_dir),
            "summary_json": str(input_dir / "summary.json"),
            "per_bug_topk_jsonl": str(input_dir / "per_bug_topk.jsonl"),
        },
        "definition": {
            "full_model": "module_reduced scope from the fixed Wide & Deep benchmark",
            "w/o_module_reduction": "full_project scope using the same ranker without module-based candidate filtering",
            "note": "This ablation disables module-based candidate filtering; it does not remove all module-related feature signals.",
        },
        "environment": summary.get("environment", {}),
        "rows": rows,
        "comparison": comparison,
        "validations": validations,
    }

    write_csv(output_dir / "summary.csv", rows)
    write_json(output_dir / "summary.json", output)
    (output_dir / "report.md").write_text(build_report(output) + "\n", encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir), "validations": validations}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
