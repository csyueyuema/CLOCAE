#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Audit and repair the full-project candidate universe used by RQ3.

The original full-project benchmark reused the dataset index reference, which is
usually HEAD, and therefore can miss files that existed only in the buggy code
version.  It also omits a few CAE source extensions used by projects in the
dataset.  This script does not run the ranker; it checks which test bugs become
evaluable if the full-project candidate universe is rebuilt from the buggy
``code_ref`` with the additional CAE source extensions.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "ranker") not in sys.path:
    sys.path.insert(0, str(ROOT / "ranker"))

from build_ranking_dataset import EXTRA_EXTENSIONS, git_tree_files, read_json, read_jsonl  # type: ignore


DEFAULT_CAE_SOURCE_EXTENSIONS = (".jl", ".mo", ".tpl", ".edp")
FULL_SCOPE = "full_project"


def normalize_path(value: Any) -> str:
    return str(value).strip().replace("\\", "/")


def parse_extensions(value: str) -> List[str]:
    extensions: List[str] = []
    for item in value.split(","):
        ext = item.strip()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        extensions.append(ext)
    return sorted(dict.fromkeys(extensions))


def parse_csv_list(value: str) -> List[str]:
    return sorted(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_old_full_positive_counts(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    counts: Dict[str, int] = {}
    for row in read_jsonl(path):
        if row.get("scope") != FULL_SCOPE:
            continue
        bug_id = str(row.get("bug_id", ""))
        if not bug_id:
            continue
        counts[bug_id] = int(row.get("positive_candidates", 0) or 0)
    return counts


def group_rows(rows: Sequence[Mapping[str, Any]], project_filter: Optional[str]) -> Tuple[List[str], Dict[str, List[Mapping[str, Any]]]]:
    order: List[str] = []
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if project_filter and row.get("project") != project_filter:
            continue
        bug_id = str(row["bug_id"])
        if bug_id not in grouped:
            order.append(bug_id)
        grouped[bug_id].append(row)
    return order, grouped


class TreeCache:
    def __init__(self, repos_dir: Path):
        self.repos_dir = repos_dir
        self._trees: Dict[Tuple[str, str], Set[str]] = {}

    def files(self, project: str, ref: str) -> Set[str]:
        key = (project, ref)
        if key not in self._trees:
            repo_git_dir = self.repos_dir / project
            try:
                self._trees[key] = set(git_tree_files(repo_git_dir, ref=ref))
            except Exception:
                self._trees[key] = set()
        return self._trees[key]


def supported_extensions(config_path: Path, extra_extensions: Sequence[str]) -> Tuple[Set[str], Set[str]]:
    config = read_json(config_path)
    current = {str(ext) for ext in config.get("extensions", [])}
    current.update(str(ext) for ext in EXTRA_EXTENSIONS)
    rescued = set(current)
    rescued.update(extra_extensions)
    return current, rescued


def is_supported_source(path: str, extensions: Set[str], patterns: Sequence[str] = ()) -> bool:
    if Path(path).suffix in extensions:
        return True
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def positive_files_for_bug(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    first = rows[0]
    bug_meta = first.get("bug_meta", {}) or {}
    positives = first.get("positive_files") or bug_meta.get("modified_files") or []
    return sorted(dict.fromkeys(normalize_path(path) for path in positives))


def context_for_bug(rows: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    first = rows[0]
    bug_meta = first.get("bug_meta", {}) or {}
    return {
        "bug_id": str(first["bug_id"]),
        "project": str(first["project"]),
        "selected_module": str(first.get("selected_module") or bug_meta.get("teacher_primary_module") or ""),
        "code_ref": str(first.get("code_ref") or bug_meta.get("code_ref") or "HEAD"),
        "dataset_index_ref": str(first.get("index_ref") or bug_meta.get("index_ref") or "HEAD"),
        "split": str(first.get("split", "test")),
    }


def classify_file(
    path: str,
    current_tree: Set[str],
    code_tree: Set[str],
    current_extensions: Set[str],
    rescued_extensions: Set[str],
    current_patterns: Sequence[str],
    rescued_patterns: Sequence[str],
) -> Dict[str, Any]:
    if not path or path == "(none)":
        return {
            "file": path,
            "extension": "",
            "current_supported": False,
            "rescued_supported": False,
            "exists_dataset_ref": False,
            "exists_code_ref": False,
            "current_candidate": False,
            "rescued_candidate": False,
            "status": "no_modified_file_label",
        }

    current_supported = is_supported_source(path, current_extensions, current_patterns)
    rescued_supported = is_supported_source(path, rescued_extensions, rescued_patterns)
    exists_dataset_ref = path in current_tree
    exists_code_ref = path in code_tree
    current_candidate = current_supported and exists_dataset_ref
    rescued_candidate = rescued_supported and exists_code_ref

    if rescued_candidate:
        if current_candidate:
            status = "already_in_current_and_rescued_full_project"
        elif not current_supported and rescued_supported:
            status = "rescued_by_cae_source_extension"
        elif not exists_dataset_ref and exists_code_ref:
            status = "rescued_by_buggy_code_ref"
        else:
            status = "rescued_by_code_ref_or_extension"
    elif exists_code_ref and not rescued_supported:
        status = "unsupported_non_source_or_artifact"
    elif not exists_code_ref and exists_dataset_ref:
        status = "absent_in_buggy_code_ref_present_in_dataset_ref"
    elif not exists_code_ref:
        status = "absent_in_buggy_code_ref"
    else:
        status = "not_in_rescued_candidate_scope"

    return {
        "file": path,
        "extension": Path(path).suffix,
        "current_supported": current_supported,
        "rescued_supported": rescued_supported,
        "exists_dataset_ref": exists_dataset_ref,
        "exists_code_ref": exists_code_ref,
        "current_candidate": current_candidate,
        "rescued_candidate": rescued_candidate,
        "status": status,
    }


def classify_bug(file_statuses: Sequence[Mapping[str, Any]], old_full_positive_count: Optional[int]) -> str:
    rescued_count = sum(1 for item in file_statuses if item.get("rescued_candidate"))
    current_count = sum(1 for item in file_statuses if item.get("current_candidate"))
    observed_old_count = old_full_positive_count if old_full_positive_count is not None else current_count
    statuses = {str(item.get("status")) for item in file_statuses}

    if rescued_count > 0 and observed_old_count > 0:
        return "already_evaluable"
    if rescued_count > 0:
        return "rescued_evaluable"
    if observed_old_count > 0:
        return "old_evaluable_but_absent_from_buggy_code_ref"
    if statuses <= {"no_modified_file_label"}:
        return "still_invalid_no_modified_file_label"
    if statuses <= {"unsupported_non_source_or_artifact", "no_modified_file_label"}:
        return "still_invalid_non_source_or_unsupported_artifact"
    if any(status.startswith("absent_in_buggy_code_ref") for status in statuses):
        return "still_invalid_gold_absent_from_buggy_code_ref"
    return "still_invalid_mixed_or_manual_review"


def count_supported_files(files: Iterable[str], extensions: Set[str], patterns: Sequence[str] = ()) -> int:
    return sum(1 for path in files if is_supported_source(path, extensions, patterns))


def rebuilt_module_stats(path: Path, expected_bug_ids: Sequence[str], rescued_bug_ids: Sequence[str]) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    rows = read_jsonl(path)
    by_bug: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bug[str(row["bug_id"])].append(row)
    valid_bug_ids = {
        bug_id
        for bug_id, bug_rows in by_bug.items()
        if any(float(row.get("label", 0.0)) > 0.5 for row in bug_rows)
    }
    expected_set = set(expected_bug_ids)
    rescued_set = set(rescued_bug_ids)
    return {
        "path": str(path),
        "bugs": len(by_bug),
        "pairs": len(rows),
        "positive_pairs": sum(1 for row in rows if float(row.get("label", 0.0)) > 0.5),
        "valid_bugs": len(valid_bug_ids),
        "avg_candidates_per_bug": (len(rows) / len(by_bug)) if by_bug else 0.0,
        "bug_set_matches_audit": set(by_bug) == expected_set,
        "rescued_cases_with_positive_candidate": len(rescued_set & valid_bug_ids),
        "rescued_cases_missing_positive_candidate": sorted(rescued_set - valid_bug_ids),
    }


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    test_file = Path(args.test_file).resolve()
    repos_dir = (ROOT / args.repos_dir).resolve()
    config_path = (ROOT / args.config).resolve()
    extra_extensions = parse_extensions(args.rescue_extensions)
    extra_patterns = parse_csv_list(args.rescue_path_patterns)
    current_extensions, rescued_extensions = supported_extensions(config_path, extra_extensions)
    old_counts = load_old_full_positive_counts(Path(args.old_per_bug_topk).resolve())

    rows = read_jsonl(test_file)
    bug_order, rows_by_bug = group_rows(rows, args.project)
    tree_cache = TreeCache(repos_dir)

    detail_rows: List[Dict[str, Any]] = []
    file_detail_rows: List[Dict[str, Any]] = []

    for bug_id in bug_order:
        module_rows = rows_by_bug[bug_id]
        ctx = context_for_bug(module_rows)
        positives = positive_files_for_bug(module_rows)
        current_tree = tree_cache.files(ctx["project"], ctx["dataset_index_ref"])
        code_tree = tree_cache.files(ctx["project"], ctx["code_ref"])

        file_statuses = [
            classify_file(
                path=path,
                current_tree=current_tree,
                code_tree=code_tree,
                current_extensions=current_extensions,
                rescued_extensions=rescued_extensions,
                current_patterns=[],
                rescued_patterns=extra_patterns,
            )
            for path in positives
        ]
        current_positive_count = sum(1 for item in file_statuses if item["current_candidate"])
        rescued_positive_count = sum(1 for item in file_statuses if item["rescued_candidate"])
        module_positive_count = sum(1 for row in module_rows if float(row.get("label", 0.0)) > 0.5)
        old_full_positive_count = old_counts.get(bug_id)
        category = classify_bug(file_statuses, old_full_positive_count)

        detail = {
            **ctx,
            "positive_files": " | ".join(positives),
            "positive_file_count": len(positives),
            "module_reduced_positive_count": module_positive_count,
            "old_full_positive_count": old_full_positive_count,
            "tree_current_full_positive_count": current_positive_count,
            "rescued_full_positive_count": rescued_positive_count,
            "dataset_ref_supported_candidates": count_supported_files(current_tree, current_extensions),
            "rescued_supported_candidates": count_supported_files(code_tree, rescued_extensions, extra_patterns),
            "category": category,
            "file_statuses": " | ".join(f"{item['file']} => {item['status']}" for item in file_statuses),
        }
        detail_rows.append(detail)

        for item in file_statuses:
            file_detail_rows.append({**ctx, **item})

    category_counts = Counter(row["category"] for row in detail_rows)
    old_full_valid = sum(1 for row in detail_rows if (row["old_full_positive_count"] or 0) > 0)
    rescued_full_valid = sum(1 for row in detail_rows if row["rescued_full_positive_count"] > 0)
    module_valid = sum(1 for row in detail_rows if row["module_reduced_positive_count"] > 0)
    newly_rescued = sum(
        1
        for row in detail_rows
        if (row["old_full_positive_count"] or 0) <= 0 and row["rescued_full_positive_count"] > 0
    )
    still_invalid = sum(1 for row in detail_rows if row["rescued_full_positive_count"] <= 0)
    rescued_bug_ids = [row["bug_id"] for row in detail_rows if row["category"] == "rescued_evaluable"]
    rebuilt_stats = rebuilt_module_stats(Path(args.rebuilt_module_test).resolve(), bug_order, rescued_bug_ids)

    return {
        "inputs": {
            "test_file": str(test_file),
            "repos_dir": str(repos_dir),
            "config": str(config_path),
            "old_per_bug_topk": str(Path(args.old_per_bug_topk).resolve()),
            "rebuilt_module_test": str(Path(args.rebuilt_module_test).resolve()),
            "project": args.project,
        },
        "settings": {
            "current_index_ref_source": "dataset_index_ref",
            "rescued_index_ref_source": "code_ref",
            "current_extensions": sorted(current_extensions),
            "rescue_extensions": extra_extensions,
            "rescue_path_patterns": extra_patterns,
            "rescued_extensions": sorted(rescued_extensions),
        },
        "summary": {
            "total_test_crashes": len(detail_rows),
            "module_reduced_valid_crashes": module_valid,
            "old_full_project_valid_crashes": old_full_valid if old_counts else None,
            "rescued_full_project_valid_crashes": rescued_full_valid,
            "newly_rescued_full_project_crashes": newly_rescued if old_counts else None,
            "still_invalid_after_rescue": still_invalid,
            "category_counts": dict(sorted(category_counts.items())),
            "total_module_reduced_candidates": sum(len(rows_by_bug[bug_id]) for bug_id in bug_order),
            "total_rescued_full_project_candidates": sum(row["rescued_supported_candidates"] for row in detail_rows),
            "rebuilt_module_reduced": rebuilt_stats,
        },
        "bugs": detail_rows,
        "files": file_detail_rows,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def build_report(data: Mapping[str, Any]) -> str:
    summary = data["summary"]
    settings = data["settings"]
    lines = [
        "# Candidate Scope Rescue Audit",
        "",
        "This audit checks whether full-project candidates become evaluable when they are enumerated from the buggy `code_ref` and include CAE source extensions.",
        "",
        "## Summary",
        "",
        f"- Total crashes: {summary['total_test_crashes']}",
        f"- Old module-reduced crashes with positive candidates: {summary['module_reduced_valid_crashes']}",
        f"- Old full-project crashes with positive candidates: {summary['old_full_project_valid_crashes']}",
        f"- Rescued full-project crashes with positive candidates: {summary['rescued_full_project_valid_crashes']}",
        f"- Newly rescued full-project crashes: {summary['newly_rescued_full_project_crashes']}",
        f"- Still invalid after rescue: {summary['still_invalid_after_rescue']}",
        f"- Total rescued full-project candidates: {summary['total_rescued_full_project_candidates']:,}",
        "",
        "## Category Counts",
        "",
        "| Category | Cases |",
        "| --- | ---: |",
    ]
    for category, count in summary["category_counts"].items():
        lines.append(f"| {category} | {count} |")

    rebuilt = summary.get("rebuilt_module_reduced")
    if rebuilt:
        lines.extend(
            [
                "",
                "## Rebuilt Module-Reduced Test Split",
                "",
                f"- Path: `{rebuilt['path']}`",
                f"- Bugs: {rebuilt['bugs']}",
                f"- Pairs: {rebuilt['pairs']:,}",
                f"- Positive pairs: {rebuilt['positive_pairs']}",
                f"- Crashes with positive candidates: {rebuilt['valid_bugs']}",
                f"- Avg. candidates per crash: {rebuilt['avg_candidates_per_bug']:.2f}",
                f"- Bug set matches audit: {rebuilt['bug_set_matches_audit']}",
                f"- Rescued cases with positive candidates: {rebuilt['rescued_cases_with_positive_candidate']}",
                f"- Rescued cases still missing positives: {len(rebuilt['rescued_cases_missing_positive_candidate'])}",
            ]
        )

    lines.extend(
        [
            "",
            "## Rescue Settings",
            "",
            f"- Current full-project index source: {settings['current_index_ref_source']}",
            f"- Rescued full-project index source: {settings['rescued_index_ref_source']}",
            f"- Added CAE source extensions: {', '.join(settings['rescue_extensions'])}",
            f"- Added path patterns: {', '.join(settings['rescue_path_patterns']) if settings['rescue_path_patterns'] else '(none)'}",
            "",
            "## Still Invalid After Rescue",
            "",
            "| Bug | Project | Category | Positive files | File status |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    invalid_rows = [row for row in data["bugs"] if int(row["rescued_full_positive_count"]) <= 0]
    for row in invalid_rows:
        lines.append(
            f"| {row['bug_id']} | {row['project']} | {row['category']} | "
            f"`{row['positive_files']}` | `{row['file_statuses']}` |"
        )

    benchmark_test_file = data.get("inputs", {}).get(
        "rebuilt_module_test",
        "ranker/data/manual_stage2_deep_code_ref_cae_ext_testonly/test.jsonl",
    )
    run_suffix = "source_plus_build" if settings["rescue_path_patterns"] else "cae_ext"
    command_lines = [
        "python ranker/benchmark_wide_deep_scope_efficiency.py \\",
        f"  --test-file {benchmark_test_file} \\",
        "  --scopes module_reduced full_project \\",
        "  --full-project-index-ref-source code \\",
        f"  --extra-index-extensions {','.join(settings['rescue_extensions'])} \\",
    ]
    if settings["rescue_path_patterns"]:
        command_lines.append(f"  --extra-index-path-patterns '{','.join(settings['rescue_path_patterns'])}' \\")
    command_lines.extend(
        [
            f"  --cache-dir ranker/cache/strict_module_index_code_ref_{run_suffix} \\",
            f"  --output-dir ranker/results/efficiency_wide_deep_scope_code_ref_{run_suffix}",
        ]
    )
    lines.extend(["", "## Reproducible Corrected Benchmark Command", "", "```bash", *command_lines, "```"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="ranker/data/manual_stage2_deep/test.jsonl")
    parser.add_argument("--repos-dir", default="repos")
    parser.add_argument("--config", default="module_map/configs/config_base.json")
    parser.add_argument(
        "--old-per-bug-topk",
        default="ranker/results/efficiency_wide_deep_scope_gpu_full_bs64/per_bug_topk.jsonl",
    )
    parser.add_argument("--project", default=None)
    parser.add_argument("--rescue-extensions", default=",".join(DEFAULT_CAE_SOURCE_EXTENSIONS))
    parser.add_argument("--rescue-path-patterns", default="")
    parser.add_argument(
        "--rebuilt-module-test",
        default="ranker/data/manual_stage2_deep_code_ref_cae_ext_testonly/test.jsonl",
    )
    parser.add_argument("--output-dir", default="ranker/results/rq3_module_reduction_ablation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    data = audit(args)

    write_json(output_dir / "candidate_scope_rescue_audit.json", data)
    write_csv(output_dir / "candidate_scope_rescue_audit.csv", data["bugs"])
    write_csv(output_dir / "candidate_scope_rescue_file_status.csv", data["files"])
    (output_dir / "candidate_scope_rescue_audit.md").write_text(build_report(data), encoding="utf-8")

    summary = data["summary"]
    print(
        "rescued_full_project_valid="
        f"{summary['rescued_full_project_valid_crashes']} "
        "newly_rescued="
        f"{summary['newly_rescued_full_project_crashes']} "
        "still_invalid="
        f"{summary['still_invalid_after_rescue']}"
    )


if __name__ == "__main__":
    main()
