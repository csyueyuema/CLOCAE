#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CAE-ModMap (V1)
Token suggestion for extending lexicons (no ML, no AST).

It scans:
- path tokens (directory names)
- filename tokens (stems)
- dependency tokens (regex import/include/use)

Usage:
  python suggest_tokens.py --repo /path/to/repo --config configs/config_base.json --out suggestions.json --top 200
"""

from __future__ import annotations
import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

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

MAX_BYTES = 2_000_000


def normalize_tokens(s: str) -> List[str]:
    s = s.lower()
    parts = _SPLIT_RE.split(s)
    return [p for p in parts if p]


def read_text_safely(p: Path) -> str:
    try:
        b = p.read_bytes()
    except Exception:
        return ""
    if len(b) > MAX_BYTES:
        b = b[:MAX_BYTES]
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return b.decode("latin-1", errors="ignore")


def extract_deps(p: Path, text: str) -> List[str]:
    deps: List[str] = []
    ext = p.suffix.lower()

    if ext in {".f90", ".f", ".for", ".f95", ".f03", ".f08"} or p.suffix in {".F90", ".F"}:
        deps += [m.group(1).lower() for m in RE_F_USE.finditer(text)]
        deps += [m.group(1).lower() for m in RE_F_INCLUDE.finditer(text)]
    elif ext in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh"}:
        deps += [m.group(1).lower() for m in RE_C_INCLUDE.finditer(text)]
    elif ext == ".py":
        deps += [m.group(1).lower() for m in RE_PY_IMPORT.finditer(text)]
        deps += [m.group(1).lower() for m in RE_PY_FROM.finditer(text)]

    return sorted(set(deps))


def iter_files(repo: Path, exts: List[str]):
    exts = set(exts)
    for dirpath, _, filenames in os.walk(repo):
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix in exts:
                yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="suggestions.json")
    ap.add_argument("--top", type=int, default=200)
    args = ap.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    exts = cfg.get("extensions", [])
    if not exts:
        raise SystemExit("ERROR: config.extensions is empty")

    # known tokens (already present in base lexicons)
    known = set()
    for k in ["path_hints", "name_lexicon", "api_lexicon"]:
        mapping = cfg.get(k, {})
        known.update(mapping.keys())

    path_c = Counter()
    name_c = Counter()
    dep_c = Counter()

    for p in iter_files(repo, exts):
        rel = p.relative_to(repo)

        # path tokens
        for part in rel.parts[:-1]:
            for t in normalize_tokens(part):
                path_c[t] += 1

        # file name tokens
        for t in normalize_tokens(p.stem):
            name_c[t] += 1

        # dep tokens
        text = read_text_safely(p)
        deps = extract_deps(p, text) if text else []
        for d in deps:
            for t in normalize_tokens(Path(d).stem):
                dep_c[t] += 1

    def top_items(counter: Counter):
        out = []
        for t, c in counter.most_common(args.top):
            if t in known:
                continue
            if len(t) < 3:
                continue
            out.append([t, int(c)])
            if len(out) >= args.top:
                break
        return out

    out = {
        "repo": str(repo),
        "top": args.top,
        "suggest_path_tokens": top_items(path_c),
        "suggest_name_tokens": top_items(name_c),
        "suggest_dep_tokens": top_items(dep_c),
    }

    Path(args.out).write_text(json.dumps(
        out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote suggestions -> {args.out}")


if __name__ == "__main__":
    main()
