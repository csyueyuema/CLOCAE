#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate non-neural comparison methods on the file-level ranking data.

The implementations are intentionally lightweight re-implementations that use
the same candidate set and labels as CrashLocCAE:

- crashlocator_style: stack-distance and structural fallback features.
- scaffle_style: trace-informed BM25 retrieval with stack/file-path boost.
- stack_only/domain_only/heuristic: simple feature baselines for sanity checks.

They are not drop-in reproductions of the original papers because this dataset
does not include full static call graphs or Scaffle's production trace parser.
The goal is to provide fair, reproducible baselines under the same candidate
set, split, and Top-k/MFR/MAR metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from train_ranker import compute_ranking_metrics


RE_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*|[0-9]+")
RE_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "i",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "we",
    "with",
    "you",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def split_identifier(text: str) -> List[str]:
    pieces: List[str] = []
    for raw in RE_TOKEN.findall(text.replace("/", " ").replace(".", " ").replace("_", " ")):
        for part in RE_CAMEL.split(raw):
            part = part.lower().strip()
            if len(part) >= 2 and part not in STOPWORDS:
                pieces.append(part)
    return pieces


def bounded_terms(tokens: Sequence[str], max_terms: int = 160) -> List[str]:
    counts = Counter(tokens)
    return [tok for tok, _ in counts.most_common(max_terms)]


def doc_tokens(row: Dict[str, Any]) -> List[str]:
    return split_identifier(
        " ".join(
            [
                row.get("file", ""),
                row.get("file_path_text", ""),
                row.get("file_name_text", ""),
                row.get("file_text", ""),
            ]
        )
    )


def report_tokens(row: Dict[str, Any]) -> List[str]:
    return split_identifier(row.get("bug_text", ""))


def stack_files(row: Dict[str, Any]) -> List[str]:
    meta = row.get("bug_meta", {})
    files = meta.get("stack_files") or []
    return [str(x).replace("\\", "/") for x in files if str(x).strip()]


def stack_tokens(row: Dict[str, Any]) -> List[str]:
    toks: List[str] = []
    for sf in stack_files(row):
        toks.extend(split_identifier(sf))
    return toks


def suffix_match_score(file_path: str, stack: Sequence[str]) -> Tuple[float, int]:
    best = 0.0
    best_idx = 10_000
    norm_file = file_path.replace("\\", "/")
    for idx, sf in enumerate(stack):
        sf = sf.replace("\\", "/")
        if norm_file == sf:
            return 1.0, idx
        if norm_file.endswith(sf) or sf.endswith(norm_file):
            shorter = min(len(norm_file), len(sf))
            longer = max(len(norm_file), len(sf))
            score = shorter / max(longer, 1)
            if score > best:
                best = score
                best_idx = idx
    return best, best_idx


class BM25Index:
    def __init__(self, rows: Sequence[Dict[str, Any]]):
        self.doc_by_key: Dict[Tuple[str, str], List[str]] = {}
        df: Counter[str] = Counter()
        for row in rows:
            key = (row["bug_id"], row["file"])
            toks = doc_tokens(row)
            self.doc_by_key[key] = toks
            df.update(set(toks))
        self.num_docs = max(len(self.doc_by_key), 1)
        self.avgdl = sum(len(toks) for toks in self.doc_by_key.values()) / self.num_docs
        self.idf = {
            tok: math.log(1.0 + (self.num_docs - freq + 0.5) / (freq + 0.5))
            for tok, freq in df.items()
        }

    def score(self, query: Sequence[str], row: Dict[str, Any], k1: float = 1.5, b: float = 0.75) -> float:
        if not query:
            return 0.0
        key = (row["bug_id"], row["file"])
        doc = self.doc_by_key.get(key, [])
        if not doc:
            return 0.0
        tf = Counter(doc)
        dl = len(doc)
        norm = k1 * (1.0 - b + b * dl / max(self.avgdl, 1e-9))
        score = 0.0
        for term in bounded_terms(query):
            freq = tf.get(term, 0)
            if freq <= 0:
                continue
            score += self.idf.get(term, 0.0) * freq * (k1 + 1.0) / (freq + norm)
        return score


def feature(row: Dict[str, Any], key: str) -> float:
    try:
        return float(row.get("features", {}).get(key, 0.0))
    except Exception:
        return 0.0


def crashlocator_style(row: Dict[str, Any]) -> float:
    """Stack-distance score plus a weak dependency/name fallback.

    Original CrashLocator expands stack frames through static call graphs. The
    current dataset does not store project call graphs, so dependency/name
    overlap acts as a deterministic structural fallback.
    """

    suffix, idx = suffix_match_score(row["file"], stack_files(row))
    frame_weight = 1.0 / math.log2(idx + 2.0) if idx < 10_000 else 0.0
    stack_score = 18.0 * suffix * frame_weight
    structural_fallback = (
        0.65 * feature(row, "bug_dep_token_overlap")
        + 0.45 * feature(row, "bug_name_token_overlap")
        + 0.25 * feature(row, "bug_path_token_overlap")
    )
    return stack_score + structural_fallback


def stack_only(row: Dict[str, Any]) -> float:
    suffix, idx = suffix_match_score(row["file"], stack_files(row))
    frame_weight = 1.0 / math.log2(idx + 2.0) if idx < 10_000 else 0.0
    return 10.0 * feature(row, "stack_file_exact_match") + 7.0 * suffix * frame_weight


def domain_only(row: Dict[str, Any]) -> float:
    return (
        1.2 * feature(row, "symptom_overlap_count")
        + feature(row, "domain_keyword_overlap")
        + 2.0 * feature(row, "teacher_module_prob")
    )


def heuristic(row: Dict[str, Any]) -> float:
    return float(row.get("heuristic_score", 0.0))


def scaffle_style(row: Dict[str, Any], bm25: BM25Index, query_cache: Dict[Tuple[str, str], List[str]]) -> float:
    """Trace-informed IR inspired by Scaffle.

    We emphasize stack-derived path/function tokens, then blend in the full bug
    report query and a small direct stack/file suffix boost.
    """

    bug_id = row["bug_id"]
    trace_q = query_cache.setdefault((bug_id, "trace"), stack_tokens(row))
    report_q = query_cache.setdefault((bug_id, "report"), report_tokens(row))
    suffix, idx = suffix_match_score(row["file"], stack_files(row))
    frame_weight = 1.0 / math.log2(idx + 2.0) if idx < 10_000 else 0.0
    return 0.70 * bm25.score(trace_q, row) + 0.30 * bm25.score(report_q, row) + 5.0 * suffix * frame_weight


def score_rows(rows: Sequence[Dict[str, Any]], method: str, bm25: BM25Index) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    query_cache: Dict[Any, List[str]] = {}
    for row in rows:
        if method == "crashlocator_style":
            score = crashlocator_style(row)
        elif method == "scaffle_style":
            score = scaffle_style(row, bm25, query_cache)
        elif method == "stack_only":
            score = stack_only(row)
        elif method == "domain_only":
            score = domain_only(row)
        elif method == "heuristic":
            score = heuristic(row)
        else:
            raise ValueError(f"Unknown comparison method: {method}")

        scored.append(
            {
                "bug_id": row["bug_id"],
                "project": row.get("project", ""),
                "file": row["file"],
                "label": float(row["label"]),
                "score": float(score),
                "method": method,
            }
        )
    return scored


def grouped_rank_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["bug_id"]].append(dict(row))

    ranked_rows: List[Dict[str, Any]] = []
    for bug_id, items in grouped.items():
        ranked = sorted(items, key=lambda x: (-x["score"], x["file"]))
        for idx, item in enumerate(ranked, start=1):
            item["rank"] = idx
            ranked_rows.append(item)
    return ranked_rows


def evaluate_method(rows: Sequence[Dict[str, Any]], method: str, bm25: BM25Index) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    scored = score_rows(rows, method, bm25)
    metrics = compute_ranking_metrics(scored)
    result = {
        "method": method,
        **metrics,
        "num_pairs": len(scored),
        "num_bugs": len({row["bug_id"] for row in scored}),
    }
    return result, grouped_rank_rows(scored)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="ranker/data/manual_stage2_deep/test.jsonl")
    parser.add_argument("--output-file", default="ranker/outputs/comparison_methods.json")
    parser.add_argument("--predictions-dir", default="ranker/outputs/comparison_predictions")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["crashlocator_style", "scaffle_style", "stack_only", "domain_only", "heuristic"],
    )
    args = parser.parse_args()

    rows = read_jsonl(Path(args.test_file))
    bm25 = BM25Index(rows)

    results: List[Dict[str, Any]] = []
    pred_dir = Path(args.predictions_dir)
    for method in args.methods:
        metrics, ranked = evaluate_method(rows, method, bm25)
        results.append(metrics)
        write_jsonl(pred_dir / f"{method}.jsonl", ranked)

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
