#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Benchmark Wide & Deep file ranking with and without module reduction.

The benchmark fixes the trained ranker and changes only the candidate file
space:

* module_reduced: rows from the existing module-reduced test dataset.
* full_project: all supported repository files for the same test bugs.

It reports candidate counts, ranking time, and peak memory.  The timing bucket
named ranking_time_sec includes tokenization, model forward, score collection,
and sorting.  Candidate construction is measured separately and included in
end_to_end_time_sec.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import platform
import subprocess
import statistics
import sys
import threading
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "ranker") not in sys.path:
    sys.path.insert(0, str(ROOT / "ranker"))
if str(ROOT / "module_map") not in sys.path:
    sys.path.insert(0, str(ROOT / "module_map"))

from build_ranking_dataset import (  # type: ignore
    BUG_TEXT_MAX_CHARS,
    DEFAULT_MAX_BYTES,
    BareRepoModuleIndex,
    build_bug_tokens,
    build_file_text,
    git_show_text,
    keyword_hits,
    normalize_tokens,
    read_json,
    suffix_match_score,
)
from train_ranker_current import (  # type: ignore
    FEATURE_KEYS,
    FeatureNormalizer,
    WideDeepRankerV2,
    compute_ranking_metrics,
)

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None


SCOPE_MODULE = "module_reduced"
SCOPE_FULL = "full_project"
SCOPES = (SCOPE_MODULE, SCOPE_FULL)
CAE_SOURCE_EXTENSIONS = (".jl", ".mo", ".tpl", ".edp")


def parse_extension_list(value: str) -> List[str]:
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


def chunks(items: Sequence[Dict[str, Any]], size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def path_chunks(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def round_or_none(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return round(float(value), digits)


def current_rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


class ScopeMemoryTracker:
    def __init__(self, device: torch.device, interval_sec: float = 0.02):
        self.device = device
        self.interval_sec = interval_sec
        self.baseline_rss_mb: Optional[float] = None
        self.peak_rss_mb: Optional[float] = None
        self.peak_python_alloc_mb: Optional[float] = None
        self.peak_cuda_alloc_mb: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "ScopeMemoryTracker":
        gc.collect()
        self.baseline_rss_mb = current_rss_mb()
        self.peak_rss_mb = self.baseline_rss_mb
        tracemalloc.start()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        if psutil is not None:
            self._thread = threading.Thread(target=self._sample_rss, daemon=True)
            self._thread.start()
        return self

    def _sample_rss(self) -> None:
        while not self._stop.wait(self.interval_sec):
            rss = current_rss_mb()
            if rss is not None:
                if self.peak_rss_mb is None or rss > self.peak_rss_mb:
                    self.peak_rss_mb = rss

    def __exit__(self, exc_type, exc, tb) -> None:
        rss = current_rss_mb()
        if rss is not None:
            if self.peak_rss_mb is None or rss > self.peak_rss_mb:
                self.peak_rss_mb = rss

        _, peak = tracemalloc.get_traced_memory()
        self.peak_python_alloc_mb = peak / (1024 * 1024)
        tracemalloc.stop()

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.peak_cuda_alloc_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def to_dict(self) -> Dict[str, Optional[float]]:
        rss_delta = None
        if self.peak_rss_mb is not None and self.baseline_rss_mb is not None:
            rss_delta = self.peak_rss_mb - self.baseline_rss_mb
        return {
            "baseline_rss_mb": round_or_none(self.baseline_rss_mb),
            "peak_rss_mb": round_or_none(self.peak_rss_mb),
            "peak_rss_delta_mb": round_or_none(rss_delta),
            "peak_python_alloc_mb": round_or_none(self.peak_python_alloc_mb),
            "peak_cuda_alloc_mb": round_or_none(self.peak_cuda_alloc_mb),
        }


@dataclass
class BugContext:
    bug_id: str
    project: str
    bug_text: str
    selected_module: str
    positive_files: List[str]
    stack_files: List[str]
    code_ref: str
    index_ref: str
    split: str
    source_file: str
    buggy_sha: Optional[str]


@dataclass
class ModelBundle:
    model_dir: Path
    model: WideDeepRankerV2
    tokenizer: Optional[Any]
    feature_keys: List[str]
    normalizer: Optional[FeatureNormalizer]
    args: Dict[str, Any]
    device: torch.device
    use_deep: bool


def load_model(model_dir: Path, batch_size_override: Optional[int]) -> ModelBundle:
    ckpt = torch.load(model_dir / "ranker.pt", map_location="cpu")
    model_args = dict(ckpt["args"])
    if batch_size_override is not None:
        model_args["batch_size"] = batch_size_override

    feature_keys = list(ckpt.get("feature_keys", FEATURE_KEYS))
    use_wide = not model_args.get("disable_wide", False)
    use_deep = not model_args.get("disable_deep", False)

    normalizer = None
    if ckpt.get("normalizer") is not None:
        normalizer = FeatureNormalizer(feature_keys)
        normalizer.load_state_dict(ckpt["normalizer"])

    tokenizer = AutoTokenizer.from_pretrained(model_dir / "encoder") if use_deep else None
    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WideDeepRankerV2(
        model_name=str(model_dir / "encoder") if use_deep else model_args.get("model_name", "microsoft/codebert-base"),
        wide_dim=len(feature_keys),
        deep_hidden=model_args.get("deep_hidden", 256),
        wide_hidden=model_args.get("wide_hidden", 64),
        dropout=model_args.get("dropout", 0.2),
        use_wide=use_wide,
        use_deep=use_deep,
        pooling=model_args.get("pooling", "mean"),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return ModelBundle(
        model_dir=model_dir,
        model=model,
        tokenizer=tokenizer,
        feature_keys=feature_keys,
        normalizer=normalizer,
        args=model_args,
        device=device,
        use_deep=use_deep,
    )


class FullProjectCandidateBuilder:
    def __init__(
        self,
        root: Path,
        repos_dir: Path,
        cache_dir: Path,
        config_path: Path,
        symptoms_path: Path,
        modules_kb_path: Path,
        teacher_file: Path,
        max_file_bytes: int,
        git_batch_size: int,
        full_project_index_ref_source: str = "dataset",
        extra_index_extensions: Optional[Sequence[str]] = None,
        extra_index_path_patterns: Optional[Sequence[str]] = None,
    ):
        self.root = root
        self.repos_dir = repos_dir
        self.cache_dir = cache_dir
        self.config = read_json(config_path)
        if extra_index_extensions:
            extensions = set(self.config.get("extensions", []))
            extensions.update(extra_index_extensions)
            self.config["extensions"] = sorted(extensions)
        if extra_index_path_patterns:
            patterns = set(self.config.get("extra_path_patterns", []))
            patterns.update(extra_index_path_patterns)
            self.config["extra_path_patterns"] = sorted(patterns)
        self.symptoms = read_json(symptoms_path)
        self.modules_kb = read_json(modules_kb_path)
        self.teacher_map = self._load_teacher_map(teacher_file)
        self.domain_keywords = self._build_domain_keywords()
        self.repo_indices: Dict[Tuple[str, str], BareRepoModuleIndex] = {}
        self.max_file_bytes = max_file_bytes
        self.git_batch_size = git_batch_size
        self.full_project_index_ref_source = full_project_index_ref_source
        self.extra_index_extensions = sorted(extra_index_extensions or [])
        self.extra_index_path_patterns = sorted(extra_index_path_patterns or [])

    def _load_teacher_map(self, path: Path) -> Dict[str, Dict[str, Any]]:
        if not path.exists():
            return {}
        rows = read_jsonl(path)
        return {row["bug_id"]: row for row in rows if "bug_id" in row}

    def _build_domain_keywords(self) -> List[str]:
        keywords = set()
        for mod_data in self.modules_kb.values():
            symptoms = mod_data.get("symptoms", {})
            for group in symptoms.values():
                if isinstance(group, list):
                    keywords.update(str(x).lower() for x in group)
            for group_name in ("rootcause_trigger", "confusion"):
                group = mod_data.get(group_name)
                if isinstance(group, list):
                    for item in group:
                        keywords.update(x for x in normalize_tokens(str(item)) if len(x) >= 3)
        for mod_map in self.symptoms.values():
            keywords.update(str(k).lower() for k in mod_map.keys())
        return sorted(keywords)

    def _project_index(self, project: str, ref: str) -> BareRepoModuleIndex:
        key = (project, ref)
        if key not in self.repo_indices:
            self.repo_indices[key] = BareRepoModuleIndex(
                self.repos_dir / project,
                self.config,
                self.cache_dir,
                ref=ref,
            )
        return self.repo_indices[key]

    def _full_project_index_ref(self, ctx: BugContext) -> str:
        if self.full_project_index_ref_source == "dataset":
            return ctx.index_ref
        if self.full_project_index_ref_source == "code":
            return ctx.code_ref
        if self.full_project_index_ref_source == "head":
            return "HEAD"
        raise ValueError(f"Unknown full-project index ref source: {self.full_project_index_ref_source}")

    def _teacher_prob(self, bug_id: str, module: str) -> float:
        teacher = self.teacher_map.get(bug_id, {})
        probs_map = teacher.get("teacher_probs_map", {})
        try:
            return float(probs_map.get(module, 0.0))
        except Exception:
            return 0.0

    def _git_show_texts_batch(self, repo_git_dir: Path, ref: str, rel_paths: Sequence[str]) -> Dict[str, str]:
        if not rel_paths:
            return {}

        cmd = ["git", f"--git-dir={repo_git_dir}", "cat-file", "--batch"]
        payload = "".join(f"{ref}:{path}\n" for path in rel_paths).encode("utf-8", errors="ignore")
        try:
            proc = subprocess.run(
                cmd,
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except Exception:
            return {
                path: git_show_text(repo_git_dir, path, max_bytes=self.max_file_bytes, ref=ref)
                for path in rel_paths
            }

        if proc.returncode != 0 and not proc.stdout:
            return {
                path: git_show_text(repo_git_dir, path, max_bytes=self.max_file_bytes, ref=ref)
                for path in rel_paths
            }

        out = proc.stdout
        pos = 0
        texts: Dict[str, str] = {}
        for path in rel_paths:
            line_end = out.find(b"\n", pos)
            if line_end < 0:
                texts[path] = ""
                continue
            header = out[pos:line_end].decode("utf-8", errors="ignore")
            pos = line_end + 1
            parts = header.split()
            if len(parts) >= 2 and parts[-1] == "missing":
                texts[path] = ""
                continue
            if len(parts) < 3:
                texts[path] = ""
                continue
            try:
                size = int(parts[2])
            except ValueError:
                texts[path] = ""
                continue
            blob = out[pos : pos + size]
            pos += size
            if pos < len(out) and out[pos : pos + 1] == b"\n":
                pos += 1
            texts[path] = blob[: self.max_file_bytes].decode("utf-8", errors="ignore")
        return texts

    def _git_show_texts(self, repo_git_dir: Path, ref: str, rel_paths: Sequence[str]) -> Dict[str, str]:
        texts: Dict[str, str] = {}
        batch_size = max(int(self.git_batch_size), 1)
        for group in path_chunks(rel_paths, batch_size):
            texts.update(self._git_show_texts_batch(repo_git_dir, ref, group))
        return texts

    def build_rows(self, ctx: BugContext) -> List[Dict[str, Any]]:
        index_ref = self._full_project_index_ref(ctx)
        repo_index = self._project_index(ctx.project, index_ref)
        positive_set = set(ctx.positive_files)
        bug_tokens = set(build_bug_tokens(ctx.bug_text))
        bug_domain_hits = set(keyword_hits(ctx.bug_text.lower(), self.domain_keywords))
        rows: List[Dict[str, Any]] = []
        text_by_path = self._git_show_texts(
            repo_index.repo_git_dir,
            ctx.code_ref,
            [record["file"] for record in repo_index.records],
        )

        for record in repo_index.records:
            file_path = record["file"]
            stack_exact = 1.0 if file_path in ctx.stack_files else 0.0
            stack_suffix = suffix_match_score(file_path, ctx.stack_files)
            path_overlap = len(bug_tokens & set(record.get("path_tokens", [])))
            name_overlap = len(bug_tokens & set(record.get("name_tokens", [])))
            dep_overlap = len(bug_tokens & set(record.get("dep_tokens", [])))
            module = record.get("module", "")
            module_match = 1 if module == ctx.selected_module else 0
            symptom_hits = keyword_hits(ctx.bug_text, list(self.symptoms.get(module, {}).keys()))
            domain_source = " ".join(
                list(record.get("path_tokens", []))
                + list(record.get("name_tokens", []))
                + list(record.get("dep_tokens", []))
                + list(record.get("api_hits", []))
            )
            domain_hits = bug_domain_hits & set(keyword_hits(domain_source, list(bug_domain_hits)))
            teacher_prob = self._teacher_prob(ctx.bug_id, module)

            raw_file_text = text_by_path.get(file_path, "")
            file_text = build_file_text(file_path, raw_file_text)

            rows.append(
                {
                    "bug_id": ctx.bug_id,
                    "project": ctx.project,
                    "split": ctx.split,
                    "selected_module": ctx.selected_module,
                    "file": file_path,
                    "file_module": module,
                    "label": 1 if file_path in positive_set else 0,
                    "bug_text": ctx.bug_text[:BUG_TEXT_MAX_CHARS],
                    "file_path_text": file_path.replace("/", " "),
                    "file_name_text": Path(file_path).name,
                    "file_text": file_text,
                    "features": {
                        "stack_file_exact_match": stack_exact,
                        "stack_file_suffix_match": round(stack_suffix, 6),
                        "bug_path_token_overlap": path_overlap,
                        "bug_name_token_overlap": name_overlap,
                        "bug_dep_token_overlap": dep_overlap,
                        "module_match_flag": module_match,
                        "module_margin": round(float(record.get("margin", 0.0)), 6),
                        "symptom_overlap_count": len(symptom_hits),
                        "domain_keyword_overlap": len(domain_hits),
                        "teacher_module_prob": round(teacher_prob, 6),
                    },
                    "source_file": ctx.source_file,
                    "positive_files": ctx.positive_files,
                    "code_ref": ctx.code_ref,
                    "index_ref": index_ref,
                    "dataset_index_ref": ctx.index_ref,
                    "buggy_sha": ctx.buggy_sha,
                }
            )

        return rows


def group_module_rows(rows: Sequence[Dict[str, Any]], project_filter: Optional[str]) -> Tuple[List[str], Dict[str, List[Dict[str, Any]]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    order: List[str] = []
    for row in rows:
        if project_filter and row.get("project") != project_filter:
            continue
        bug_id = row["bug_id"]
        if bug_id not in grouped:
            order.append(bug_id)
        grouped[bug_id].append(row)
    return order, grouped


def make_bug_context(rows: Sequence[Dict[str, Any]]) -> BugContext:
    first = rows[0]
    bug_meta = first.get("bug_meta", {}) or {}
    positive_files = first.get("positive_files") or bug_meta.get("modified_files") or []
    stack_files = bug_meta.get("stack_files") or []
    code_ref = first.get("code_ref") or bug_meta.get("code_ref") or "HEAD"
    index_ref = first.get("index_ref") or bug_meta.get("index_ref") or "HEAD"
    return BugContext(
        bug_id=first["bug_id"],
        project=first["project"],
        bug_text=first.get("bug_text") or bug_meta.get("bug_text") or "",
        selected_module=first.get("selected_module") or bug_meta.get("teacher_primary_module") or "",
        positive_files=sorted(dict.fromkeys(str(x).replace("\\", "/") for x in positive_files)),
        stack_files=sorted(dict.fromkeys(str(x).replace("\\", "/") for x in stack_files)),
        code_ref=code_ref,
        index_ref=index_ref,
        split=first.get("split", "test"),
        source_file=first.get("source_file", ""),
        buggy_sha=first.get("buggy_sha") or bug_meta.get("buggy_sha"),
    )


def row_text(row: Dict[str, Any]) -> str:
    return (
        f"[BUG] {row.get('bug_text', '')}\n"
        f"[FILE_PATH] {row.get('file_path_text', row.get('file', '').replace('/', ' '))}\n"
        f"[FILE_NAME] {row.get('file_name_text', Path(row.get('file', '')).name)}\n"
        f"[FILE_TEXT] {row.get('file_text', '')}"
    )


def row_features(row: Dict[str, Any], feature_keys: Sequence[str], normalizer: Optional[FeatureNormalizer]) -> List[float]:
    features = row.get("features", {})
    raw = [float(features.get(key, 0.0)) for key in feature_keys]
    return normalizer.transform(raw) if normalizer is not None else raw


def score_rows(
    rows: Sequence[Dict[str, Any]],
    bundle: ModelBundle,
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], float]:
    scored: List[Dict[str, Any]] = []
    elapsed = 0.0
    with torch.no_grad():
        for batch_rows in chunks(rows, batch_size):
            start = time.perf_counter()
            if bundle.use_deep:
                assert bundle.tokenizer is not None
                texts = [row_text(row) for row in batch_rows]
                enc = bundle.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=bundle.args.get("max_length", 384),
                    return_tensors="pt",
                )
            else:
                batch_len = len(batch_rows)
                enc = {
                    "input_ids": torch.zeros((batch_len, 1), dtype=torch.long),
                    "attention_mask": torch.zeros((batch_len, 1), dtype=torch.long),
                }

            input_ids = enc["input_ids"].to(bundle.device)
            attention_mask = enc["attention_mask"].to(bundle.device)
            wide_features = torch.tensor(
                [row_features(row, bundle.feature_keys, bundle.normalizer) for row in batch_rows],
                dtype=torch.float,
                device=bundle.device,
            )
            logits = bundle.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                wide_features=wide_features,
            )
            scores = torch.sigmoid(logits).detach().cpu().tolist()
            if bundle.device.type == "cuda":
                torch.cuda.synchronize(bundle.device)
            elapsed += time.perf_counter() - start

            for row, score in zip(batch_rows, scores):
                scored.append(
                    {
                        "bug_id": row["bug_id"],
                        "file": row["file"],
                        "score": float(score),
                        "label": float(row.get("label", 0.0)),
                    }
                )
    return scored, elapsed


def warmup_model(
    rows: Sequence[Dict[str, Any]],
    bundle: ModelBundle,
    batch_size: int,
    warmup_batches: int,
) -> None:
    if warmup_batches <= 0 or not rows:
        return
    limit = max(batch_size * warmup_batches, 1)
    score_rows(rows[:limit], bundle, batch_size)
    if bundle.device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def limit_rows_for_scoring(rows: Sequence[Dict[str, Any]], limit: int) -> Tuple[List[Dict[str, Any]], bool]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows), False
    selected: List[Dict[str, Any]] = []
    seen = set()
    for row in list(rows[:limit]) + [row for row in rows if float(row.get("label", 0.0)) > 0.5]:
        key = row["file"]
        if key not in seen:
            selected.append(row)
            seen.add(key)
    return selected, True


def positive_ranks(ranked: Sequence[Dict[str, Any]]) -> List[int]:
    return [idx + 1 for idx, row in enumerate(ranked) if float(row.get("label", 0.0)) > 0.5]


def benchmark_scope(
    scope: str,
    bug_ids: Sequence[str],
    module_rows_by_bug: Dict[str, List[Dict[str, Any]]],
    builder: FullProjectCandidateBuilder,
    bundle: ModelBundle,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    batch_size = int(args.batch_size or bundle.args.get("batch_size", 4))
    candidate_counts: List[int] = []
    scored_counts: List[int] = []
    positive_candidate_counts: List[int] = []
    per_bug_topk: List[Dict[str, Any]] = []
    all_scored: List[Dict[str, Any]] = []
    candidate_build_time = 0.0
    model_score_time = 0.0
    sort_time = 0.0
    truncated = False

    with ScopeMemoryTracker(bundle.device) as memory:
        scope_start = time.perf_counter()
        for bug_idx, bug_id in enumerate(bug_ids, start=1):
            bug_start = time.perf_counter()
            module_rows = module_rows_by_bug[bug_id]
            ctx = make_bug_context(module_rows)

            build_start = time.perf_counter()
            if scope == SCOPE_MODULE:
                candidate_rows = [dict(row) for row in module_rows]
            elif scope == SCOPE_FULL:
                candidate_rows = builder.build_rows(ctx)
            else:
                raise ValueError(f"Unknown scope: {scope}")
            bug_candidate_build_time = time.perf_counter() - build_start
            candidate_build_time += bug_candidate_build_time

            candidate_counts.append(len(candidate_rows))
            positive_candidate_counts.append(sum(1 for row in candidate_rows if float(row.get("label", 0.0)) > 0.5))

            rows_to_score, limited = limit_rows_for_scoring(candidate_rows, int(args.score_candidate_limit_per_bug))
            truncated = truncated or limited
            scored_counts.append(len(rows_to_score))

            scored, bug_model_score_time = score_rows(rows_to_score, bundle, batch_size)
            model_score_time += bug_model_score_time

            sort_start = time.perf_counter()
            ranked = sorted(scored, key=lambda row: (-float(row["score"]), row["file"]))
            bug_sort_time = time.perf_counter() - sort_start
            sort_time += bug_sort_time
            ranks = positive_ranks(ranked)
            all_scored.extend(scored)
            bug_end_to_end_time = time.perf_counter() - bug_start

            per_bug_topk.append(
                {
                    "scope": scope,
                    "bug_id": bug_id,
                    "project": ctx.project,
                    "candidate_count": len(candidate_rows),
                    "scored_candidate_count": len(rows_to_score),
                    "positive_candidates": positive_candidate_counts[-1],
                    "positive_ranks": ranks,
                    "candidate_build_time_sec": bug_candidate_build_time,
                    "model_score_time_sec": bug_model_score_time,
                    "sort_time_sec": bug_sort_time,
                    "ranking_time_sec": bug_model_score_time + bug_sort_time,
                    "end_to_end_time_sec": bug_end_to_end_time,
                    "topk": [
                        {
                            "rank": idx + 1,
                            "file": row["file"],
                            "score": row["score"],
                            "label": row["label"],
                        }
                        for idx, row in enumerate(ranked[: int(args.topk_save)])
                    ],
                }
            )

            del candidate_rows
            del rows_to_score
            del scored

            if args.progress_every > 0 and (bug_idx % args.progress_every == 0 or bug_idx == len(bug_ids)):
                print(
                    f"[PROGRESS] scope={scope} bug={bug_idx}/{len(bug_ids)} "
                    f"id={bug_id} candidates={candidate_counts[-1]} "
                    f"ranking_sec={bug_model_score_time + bug_sort_time:.3f} "
                    f"elapsed_sec={time.perf_counter() - scope_start:.1f}",
                    flush=True,
                )

        end_to_end_time = time.perf_counter() - scope_start

    metrics = compute_ranking_metrics(all_scored)
    ranking_time = model_score_time + sort_time
    total_candidates = sum(candidate_counts)
    total_scored = sum(scored_counts)
    summary: Dict[str, Any] = {
        "scope": scope,
        "num_bugs": len(bug_ids),
        "total_candidates": total_candidates,
        "total_scored_candidates": total_scored,
        "positive_candidates": sum(positive_candidate_counts),
        "avg_candidates_per_bug": mean([float(x) for x in candidate_counts]),
        "median_candidates_per_bug": median([float(x) for x in candidate_counts]),
        "avg_scored_candidates_per_bug": mean([float(x) for x in scored_counts]),
        "candidate_build_time_sec": candidate_build_time,
        "model_score_time_sec": model_score_time,
        "sort_time_sec": sort_time,
        "ranking_time_sec": ranking_time,
        "end_to_end_time_sec": end_to_end_time,
        "avg_ranking_time_sec_per_bug": safe_ratio(ranking_time, len(bug_ids)),
        "avg_end_to_end_time_sec_per_bug": safe_ratio(end_to_end_time, len(bug_ids)),
        "ranking_ms_per_candidate": safe_ratio(ranking_time * 1000.0, total_scored),
        "end_to_end_ms_per_candidate": safe_ratio(end_to_end_time * 1000.0, total_candidates),
        "score_candidate_limit_per_bug": int(args.score_candidate_limit_per_bug),
        "scoring_truncated": truncated,
        "metrics": metrics,
    }
    summary.update(memory.to_dict())
    return summary, per_bug_topk, all_scored


def build_comparison(scope_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if SCOPE_FULL not in scope_summaries or SCOPE_MODULE not in scope_summaries:
        return {}
    full = scope_summaries[SCOPE_FULL]
    module = scope_summaries[SCOPE_MODULE]
    comparison = {
        "full_module_candidate_ratio": safe_ratio(full["total_candidates"], module["total_candidates"]),
        "candidate_reduction_ratio": None,
        "ranking_time_speedup_full_over_module": safe_ratio(full["ranking_time_sec"], module["ranking_time_sec"]),
        "end_to_end_time_speedup_full_over_module": safe_ratio(full["end_to_end_time_sec"], module["end_to_end_time_sec"]),
        "ranking_time_reduction_ratio": None,
        "end_to_end_time_reduction_ratio": None,
        "peak_rss_delta_reduction_ratio": None,
        "peak_python_alloc_reduction_ratio": None,
    }
    if full["total_candidates"]:
        comparison["candidate_reduction_ratio"] = 1.0 - (module["total_candidates"] / full["total_candidates"])
    if full["ranking_time_sec"]:
        comparison["ranking_time_reduction_ratio"] = 1.0 - (module["ranking_time_sec"] / full["ranking_time_sec"])
    if full["end_to_end_time_sec"]:
        comparison["end_to_end_time_reduction_ratio"] = 1.0 - (module["end_to_end_time_sec"] / full["end_to_end_time_sec"])
    full_rss = full.get("peak_rss_delta_mb")
    module_rss = module.get("peak_rss_delta_mb")
    if full_rss and full_rss > 0 and module_rss is not None:
        comparison["peak_rss_delta_reduction_ratio"] = 1.0 - (module_rss / full_rss)
    full_py = full.get("peak_python_alloc_mb")
    module_py = module.get("peak_python_alloc_mb")
    if full_py and full_py > 0 and module_py is not None:
        comparison["peak_python_alloc_reduction_ratio"] = 1.0 - (module_py / full_py)
    return comparison


def environment_info(bundle: ModelBundle, args: argparse.Namespace) -> Dict[str, Any]:
    cuda_devices = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            cuda_devices.append(torch.cuda.get_device_name(idx))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": cuda_devices,
        "device": str(bundle.device),
        "model_dir": str(bundle.model_dir),
        "test_file": str(Path(args.test_file).resolve()),
        "batch_size": int(args.batch_size or bundle.args.get("batch_size", 4)),
        "max_length": int(bundle.args.get("max_length", 384)),
        "feature_keys": bundle.feature_keys,
        "scopes": args.scopes,
        "max_bugs": args.max_bugs,
        "bug_offset": args.bug_offset,
        "project": args.project,
        "score_candidate_limit_per_bug": args.score_candidate_limit_per_bug,
        "max_file_bytes": args.max_file_bytes,
        "git_batch_size": args.git_batch_size,
        "warmup_batches": args.warmup_batches,
        "full_project_index_ref_source": args.full_project_index_ref_source,
        "extra_index_extensions": parse_extension_list(args.extra_index_extensions),
        "extra_index_path_patterns": parse_csv_list(args.extra_index_path_patterns),
    }


def write_summary_csv(path: Path, summaries: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scope",
        "num_bugs",
        "total_candidates",
        "total_scored_candidates",
        "positive_candidates",
        "avg_candidates_per_bug",
        "median_candidates_per_bug",
        "ranking_time_sec",
        "end_to_end_time_sec",
        "avg_ranking_time_sec_per_bug",
        "ranking_ms_per_candidate",
        "peak_rss_mb",
        "peak_rss_delta_mb",
        "peak_python_alloc_mb",
        "peak_cuda_alloc_mb",
        "top1",
        "top3",
        "top5",
        "top10",
        "mrr",
        "mfr",
        "mar",
        "scoring_truncated",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            metrics = summary.get("metrics", {})
            row = {field: summary.get(field) for field in fields}
            for key in ("top1", "top3", "top5", "top10", "mrr", "mfr", "mar"):
                row[key] = metrics.get(key)
            writer.writerow(row)


def pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(path: Path, summaries: Sequence[Dict[str, Any]], comparison: Dict[str, Any], env: Dict[str, Any]) -> None:
    by_scope = {s["scope"]: s for s in summaries}
    lines = [
        "# Wide & Deep Candidate-Scope Efficiency Benchmark",
        "",
        "This benchmark fixes the trained Wide & Deep file ranker and changes only the candidate file scope.",
        "",
        "## Scope Summary",
        "",
        "| Scope | Bugs | Candidates | Avg/Bug | Ranking Time (s) | End-to-End (s) | ms/Candidate | Peak RSS Delta (MB) | Peak Python Alloc (MB) | Top-10 | MFR | MAR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scope in SCOPES:
        if scope not in by_scope:
            continue
        s = by_scope[scope]
        m = s.get("metrics", {})
        lines.append(
            f"| {scope} | {s['num_bugs']} | {s['total_candidates']} | {fmt(s['avg_candidates_per_bug'], 2)} | "
            f"{fmt(s['ranking_time_sec'], 4)} | {fmt(s['end_to_end_time_sec'], 4)} | "
            f"{fmt(s['ranking_ms_per_candidate'], 4)} | {fmt(s.get('peak_rss_delta_mb'), 2)} | "
            f"{fmt(s.get('peak_python_alloc_mb'), 2)} | {fmt(m.get('top10'), 4)} | {fmt(m.get('mfr'), 4)} | {fmt(m.get('mar'), 4)} |"
        )

    if comparison:
        lines.extend(
            [
                "",
                "## Full vs Module-Reduced",
                "",
                f"- Candidate ratio (full/module): {fmt(comparison.get('full_module_candidate_ratio'), 4)}",
                f"- Candidate reduction: {pct(comparison.get('candidate_reduction_ratio'))}",
                f"- Ranking-time speedup (full/module): {fmt(comparison.get('ranking_time_speedup_full_over_module'), 4)}",
                f"- Ranking-time reduction: {pct(comparison.get('ranking_time_reduction_ratio'))}",
                f"- End-to-end speedup (full/module): {fmt(comparison.get('end_to_end_time_speedup_full_over_module'), 4)}",
                f"- Peak RSS delta reduction: {pct(comparison.get('peak_rss_delta_reduction_ratio'))}",
                f"- Peak Python allocation reduction: {pct(comparison.get('peak_python_alloc_reduction_ratio'))}",
            ]
        )

    if any(s.get("scoring_truncated") for s in summaries):
        lines.extend(
            [
                "",
                "## Warning",
                "",
                "This run used score_candidate_limit_per_bug, so timing/effectiveness values are smoke-test values, not publication-ready final results.",
            ]
        )

    lines.extend(
        [
            "",
            "## Environment",
            "",
            f"- Generated at: {env['generated_at']}",
            f"- Python: {env['python']}",
            f"- Platform: {env['platform']}",
            f"- PyTorch: {env['torch']}",
            f"- CUDA available: {env['cuda_available']}",
            f"- Device: {env['device']}",
            f"- Model: `{env['model_dir']}`",
            f"- Test file: `{env['test_file']}`",
            f"- Batch size: {env['batch_size']}",
            f"- Max length: {env['max_length']}",
            f"- Max file bytes: {env['max_file_bytes']}",
            f"- Git batch size: {env['git_batch_size']}",
            f"- Warmup batches: {env['warmup_batches']}",
            f"- Full-project index ref source: {env['full_project_index_ref_source']}",
            f"- Extra index extensions: {', '.join(env['extra_index_extensions']) if env['extra_index_extensions'] else '(none)'}",
            f"- Extra index path patterns: {', '.join(env['extra_index_path_patterns']) if env['extra_index_path_patterns'] else '(none)'}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="ranker/outputs/wide_deep")
    parser.add_argument("--test-file", default="ranker/data/manual_stage2_deep/test.jsonl")
    parser.add_argument("--output-dir", default="ranker/results/efficiency_wide_deep_scope")
    parser.add_argument("--repos-dir", default="repos")
    parser.add_argument("--config", default="module_map/configs/config_base.json")
    parser.add_argument("--symptoms", default="module_map/kb/module_symptoms.json")
    parser.add_argument("--modules-kb", default="module_map/configs/modules_kb.json")
    parser.add_argument("--teacher-file", default="distill/data/manual_seed/teacher_labels.jsonl")
    parser.add_argument("--cache-dir", default="ranker/cache/strict_module_index_fast")
    parser.add_argument("--scopes", nargs="+", choices=SCOPES, default=list(SCOPES))
    parser.add_argument("--project", default=None)
    parser.add_argument("--bug-offset", type=int, default=0)
    parser.add_argument("--max-bugs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--git-batch-size", type=int, default=256)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument(
        "--full-project-index-ref-source",
        choices=["dataset", "code", "head"],
        default="dataset",
        help=(
            "Reference used to enumerate full-project candidates. "
            "dataset preserves old behavior, code uses the buggy code_ref, and head uses HEAD."
        ),
    )
    parser.add_argument(
        "--extra-index-extensions",
        default="",
        help=(
            "Comma-separated source extensions to add to the project index. "
            f"For CAE rescue runs, use: {','.join(CAE_SOURCE_EXTENSIONS)}"
        ),
    )
    parser.add_argument(
        "--extra-index-path-patterns",
        default="",
        help="Comma-separated fnmatch path patterns for supported extensionless files.",
    )
    parser.add_argument(
        "--score-candidate-limit-per-bug",
        type=int,
        default=0,
        help="Smoke-test only. Count all candidates but score at most this many per bug plus positives.",
    )
    parser.add_argument("--topk-save", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_test_rows = read_jsonl(Path(args.test_file).resolve())
    bug_order, module_rows_by_bug = group_module_rows(all_test_rows, args.project)
    if args.bug_offset:
        bug_order = bug_order[args.bug_offset :]
    if args.max_bugs is not None:
        bug_order = bug_order[: args.max_bugs]
    if not bug_order:
        raise SystemExit("No test bugs matched the requested filters.")

    bundle = load_model(Path(args.model_dir).resolve(), args.batch_size)
    builder = FullProjectCandidateBuilder(
        root=ROOT,
        repos_dir=ROOT / args.repos_dir,
        cache_dir=ROOT / args.cache_dir,
        config_path=ROOT / args.config,
        symptoms_path=ROOT / args.symptoms,
        modules_kb_path=ROOT / args.modules_kb,
        teacher_file=ROOT / args.teacher_file,
        max_file_bytes=args.max_file_bytes,
        git_batch_size=args.git_batch_size,
        full_project_index_ref_source=args.full_project_index_ref_source,
        extra_index_extensions=parse_extension_list(args.extra_index_extensions),
        extra_index_path_patterns=parse_csv_list(args.extra_index_path_patterns),
    )

    batch_size = int(args.batch_size or bundle.args.get("batch_size", 4))
    warmup_rows = module_rows_by_bug[bug_order[0]]
    warmup_model(warmup_rows, bundle, batch_size, int(args.warmup_batches))

    summaries: List[Dict[str, Any]] = []
    topk_rows: List[Dict[str, Any]] = []
    scored_rows_path = output_dir / "scored_predictions_sample.jsonl"
    if scored_rows_path.exists():
        scored_rows_path.unlink()

    for scope in args.scopes:
        print(f"[BENCH] scope={scope} bugs={len(bug_order)}", flush=True)
        summary, per_bug_topk, all_scored = benchmark_scope(
            scope=scope,
            bug_ids=bug_order,
            module_rows_by_bug=module_rows_by_bug,
            builder=builder,
            bundle=bundle,
            args=args,
        )
        summaries.append(summary)
        topk_rows.extend(per_bug_topk)

        sample_rows = []
        for row in sorted(all_scored, key=lambda r: (r["bug_id"], -r["score"], r["file"]))[:1000]:
            sample = dict(row)
            sample["scope"] = scope
            sample_rows.append(sample)
        if sample_rows:
            mode = "a" if scored_rows_path.exists() else "w"
            with scored_rows_path.open(mode, encoding="utf-8") as f:
                for row in sample_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        del all_scored
        gc.collect()

    scope_summaries = {s["scope"]: s for s in summaries}
    comparison = build_comparison(scope_summaries)
    env = environment_info(bundle, args)

    summary_payload = {
        "environment": env,
        "summaries": summaries,
        "comparison": comparison,
        "validations": {
            "module_reduced_expected_pairs": len([r for r in all_test_rows if (not args.project or r.get("project") == args.project)]),
            "module_reduced_candidate_match": (
                scope_summaries.get(SCOPE_MODULE, {}).get("total_candidates")
                == len([r for r in all_test_rows if (not args.project or r.get("project") == args.project)])
                if SCOPE_MODULE in scope_summaries and args.max_bugs is None and args.bug_offset == 0
                else None
            ),
            "paired_bug_records": all(
                len([row for row in topk_rows if row["bug_id"] == bug_id]) == len(args.scopes)
                for bug_id in bug_order
            ),
        },
    }

    (output_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(output_dir / "summary.csv", summaries)
    write_jsonl(output_dir / "per_bug_topk.jsonl", topk_rows)
    write_report(output_dir / "report.md", summaries, comparison, env)

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    print(f"[OK] Wrote benchmark outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
