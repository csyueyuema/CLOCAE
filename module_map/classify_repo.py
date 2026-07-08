#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CAE-ModMap (V1, aligned to reasoning modules)
Config-driven weak-semantic file-to-module classifier for CAE repos.

Aligned module IDs:
- M1_CORE
- M2_MESH
- M3_FEM
- M4_NUMERICAL
- M5_SOLVER
- M6_PARALLEL
- M7_EXTERNAL

Design goals:
- No AST
- No call graph
- Deterministic & reproducible
- Scales to large repos (walk + regex + token scoring)
- Multi-language lightweight dependency extraction:
  - Fortran: use / include
  - C/C++:   #include
  - Python:  import / from ... import ...

Usage:
  python classify_repo.py --repo /path/to/repo --config configs/config_base.json --out out_dir

Outputs:
  out_dir/modules.jsonl
  out_dir/modules.csv
  out_dir/summary.json
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any


# -----------------------------
# Regex + tokenization
# -----------------------------
_SPLIT_RE = re.compile(r"[^a-z0-9]+")

# Fortran
RE_F_USE = re.compile(r"^\s*use\s+([a-z0-9_]+)", re.IGNORECASE | re.MULTILINE)
RE_F_INCLUDE = re.compile(
    r"^\s*include\s*[\"']([^\"']+)[\"']", re.IGNORECASE | re.MULTILINE)

# C/C++
RE_C_INCLUDE = re.compile(
    r"^\s*#\s*include\s*[<\"]([^\">]+)[\">]", re.MULTILINE)

# Python
RE_PY_IMPORT = re.compile(r"^\s*import\s+([a-zA-Z0-9_\.]+)", re.MULTILINE)
RE_PY_FROM = re.compile(
    r"^\s*from\s+([a-zA-Z0-9_\.]+)\s+import\s+", re.MULTILINE)

DEFAULT_MAX_BYTES = 2_000_000


@dataclass
class Evidence:
    path_tokens: List[str]
    name_tokens: List[str]
    deps: List[str]
    api_hits: List[str]


@dataclass
class Record:
    file: str
    module: str
    module_name: str
    margin: float
    scores_total: Dict[str, float]
    score_breakdown: Dict[str, Dict[str, float]]
    evidence: Evidence


def normalize_tokens(s: str) -> List[str]:
    s = s.lower()
    parts = _SPLIT_RE.split(s)
    return [p for p in parts if p]


def read_text_safely(p: Path, max_bytes: int = DEFAULT_MAX_BYTES) -> str:
    try:
        data = p.read_bytes()
    except Exception:
        return ""
    if len(data) > max_bytes:
        data = data[:max_bytes]
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return data.decode("latin-1", errors="ignore")


def extract_deps(p: Path, text: str) -> List[str]:
    """Lightweight dependency extraction without AST."""
    deps: List[str] = []
    ext = p.suffix.lower()

    # Fortran
    if ext in {".f90", ".f", ".for", ".f95", ".f03", ".f08"} or p.suffix in {".F90", ".F"}:
        deps += [m.group(1).lower() for m in RE_F_USE.finditer(text)]
        deps += [m.group(1).lower() for m in RE_F_INCLUDE.finditer(text)]
    # C/C++
    elif ext in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh"}:
        deps += [m.group(1).lower() for m in RE_C_INCLUDE.finditer(text)]
    # Python
    elif ext == ".py":
        deps += [m.group(1).lower() for m in RE_PY_IMPORT.finditer(text)]
        deps += [m.group(1).lower() for m in RE_PY_FROM.finditer(text)]

    return sorted(set(deps))


def extract_api_hits(text: str, api_lexicon: Dict[str, Dict[str, float]]) -> List[str]:
    tl = text.lower()
    hits = [k for k in api_lexicon.keys() if k.lower() in tl]  # weak signal
    hits.sort()
    return hits


def init_scores(modules: Dict[str, str]) -> Dict[str, float]:
    return {m: 0.0 for m in modules.keys()}


def add_mapping(scores: Dict[str, float], token: str, mapping: Dict[str, Dict[str, float]]) -> None:
    if token in mapping:
        for mod, w in mapping[token].items():
            scores[mod] += float(w)


def score_tokens(tokens: List[str], modules: Dict[str, str], mapping: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    s = init_scores(modules)
    for t in tokens:
        add_mapping(s, t, mapping)
        # small alias heuristic
        if t.startswith("press") and "press" in mapping:
            add_mapping(s, "press", mapping)
    return s


def score_deps_tokens(
    deps: List[str],
    modules: Dict[str, str],
    name_lexicon: Dict[str, Dict[str, float]],
    path_hints: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    s = init_scores(modules)
    for d in deps:
        stem = Path(d).stem
        for t in normalize_tokens(stem):
            add_mapping(s, t, name_lexicon)
            add_mapping(s, t, path_hints)
    return s


def weighted_sum(
    parts: Dict[str, Dict[str, float]],
    weights: Dict[str, float],
    modules: Dict[str, str]
) -> Dict[str, float]:
    out = init_scores(modules)

    w_path = float(weights.get("path", 0.35))
    w_name = float(weights.get("name", 0.35))
    w_dep = float(weights.get("dep", 0.20))
    w_api = float(weights.get("api", 0.10))

    for m in out.keys():
        out[m] = (
            w_path * parts["path"][m] +
            w_name * parts["name"][m] +
            w_dep * parts["dep"][m] +
            w_api * parts["api"][m]
        )
    return out


def pick(scores_total: Dict[str, float], tau: float, fallback: str) -> Tuple[str, float]:
    """
    Deterministic selection:
    - choose argmax by score (tie-break by module id)
    - margin returned for downstream use
    - if all scores ~0 -> fallback module (default M1_CORE)
    """
    items = sorted(scores_total.items(), key=lambda kv: (-kv[1], kv[0]))
    best_m, best_s = items[0]
    second_s = items[1][1] if len(items) > 1 else 0.0
    margin = best_s - second_s

    if best_s < 1e-12:
        return fallback, margin
    return best_m, margin


def iter_files(repo: Path, extensions: List[str]) -> Iterable[Path]:
    exts = set(extensions)
    for dirpath, _, filenames in os.walk(repo):
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix in exts:
                yield p


def classify_one(p: Path, repo: Path, cfg: Dict[str, Any]) -> Record:
    modules = cfg["modules"]
    weights = cfg["weights"]
    tau = float(cfg.get("tau", 0.15))
    fallback = cfg.get("fallback_module", "M1_CORE")

    path_hints = cfg.get("path_hints", {})
    name_lexicon = cfg.get("name_lexicon", {})
    api_lexicon = cfg.get("api_lexicon", {})

    rel = p.relative_to(repo)

    # tokens from path + file name
    path_tokens: List[str] = []
    for part in rel.parts[:-1]:
        path_tokens.extend(normalize_tokens(part))
    name_tokens = normalize_tokens(p.stem)

    # weak deps + api hits
    text = read_text_safely(p)
    deps = extract_deps(p, text) if text else []
    api_hits = extract_api_hits(text, api_lexicon) if text else []

    parts = {
        "path": score_tokens(path_tokens, modules, path_hints),
        "name": score_tokens(name_tokens, modules, name_lexicon),
        "dep": score_deps_tokens(deps, modules, name_lexicon, path_hints),
        "api": init_scores(modules),
    }

    # api scoring
    for hit in api_hits:
        for m, w in api_lexicon.get(hit, {}).items():
            parts["api"][m] += float(w)

    total = weighted_sum(parts, weights, modules)
    mod, margin = pick(total, tau, fallback=fallback)

    evidence = Evidence(
        path_tokens=path_tokens,
        name_tokens=name_tokens,
        deps=deps,
        api_hits=api_hits,
    )

    return Record(
        file=str(rel),
        module=mod,
        module_name=modules.get(mod, mod),
        margin=float(margin),
        scores_total={k: float(v) for k, v in total.items()},
        score_breakdown={k: {m: float(v) for m, v in d.items()}
                         for k, d in parts.items()},
        evidence=evidence,
    )


def write_outputs(records: List[Record], out_dir: Path, cfg: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSONL
    with (out_dir / "modules.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    # CSV
    modules = list(cfg["modules"].keys())
    fields = (
        ["file", "module", "module_name", "margin"]
        + [f"score_{m}" for m in modules]
        + ["path_tokens", "name_tokens", "deps", "api_hits"]
    )
    with (out_dir / "modules.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            row = {
                "file": r.file,
                "module": r.module,
                "module_name": r.module_name,
                "margin": f"{r.margin:.6f}",
                "path_tokens": " ".join(r.evidence.path_tokens),
                "name_tokens": " ".join(r.evidence.name_tokens),
                "deps": " ".join(r.evidence.deps),
                "api_hits": " ".join(r.evidence.api_hits),
            }
            for m in modules:
                row[f"score_{m}"] = f"{r.scores_total.get(m, 0.0):.6f}"
            w.writerow(row)

    # summary
    tau = float(cfg.get("tau", 0.15))
    counts = {m: 0 for m in cfg["modules"].keys()}
    low_margin = 0
    for r in records:
        counts[r.module] = counts.get(r.module, 0) + 1
        if r.margin < tau:
            low_margin += 1

    summary = {
        "total_files": len(records),
        "module_counts": counts,
        "low_margin_files": low_margin,
        "tau": tau,
        "weights": cfg.get("weights", {}),
        "fallback_module": cfg.get("fallback_module", "M1_CORE"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary,
                                                     ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Path to repository root")
    ap.add_argument("--config", required=True, help="Path to config json")
    ap.add_argument("--out", default="out_modules_v1", help="Output directory")
    args = ap.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    extensions = cfg.get("extensions", [])
    if not extensions:
        raise SystemExit("ERROR: config.extensions is empty")

    records: List[Record] = []
    for p in iter_files(repo, extensions):
        records.append(classify_one(p, repo, cfg))

    records.sort(key=lambda r: r.file)

    out_dir = Path(args.out).expanduser().resolve()
    write_outputs(records, out_dir, cfg)
    print(f"[OK] Classified {len(records)} files -> {out_dir}")


if __name__ == "__main__":

    main()
