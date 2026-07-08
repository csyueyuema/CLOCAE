#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CAE-ModMap (V1)
Crash/Log -> Module ranking analyzer.

Inputs:
- crash log text file
- file->module map produced by classify_repo.py (modules.jsonl)
- symptom lexicon (kb/module_symptoms.json)

Outputs:
- topk_modules.json : module ranking with evidence

Usage:
  python crash_analyze.py \
    --log crash.log \
    --modules_jsonl out_modules_v1/modules.jsonl \
    --symptoms kb/module_symptoms.json \
    --out topk_modules.json
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any


# naive file-path detector in stack traces/logs
RE_FILEPATH = re.compile(
    r"([A-Za-z0-9_\-./]+?\.(?:f90|F90|f|for|f95|c|cc|cpp|cxx|h|hpp|hh|py))")


def load_modules_jsonl(path: Path) -> Dict[str, str]:
    """
    Returns: {file_relative_path: module_id}
    """
    m = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            m[obj["file"]] = obj["module"]
    return m


def extract_files_from_log(text: str) -> List[str]:
    hits = RE_FILEPATH.findall(text)
    # normalize slashes
    files = [h.replace("\\", "/") for h in hits]
    # keep order but unique
    seen = set()
    out = []
    for x in files:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def score_by_stack(stack_files: List[str], file2mod: Dict[str, str]) -> Dict[str, float]:
    """
    Very simple V1:
    - if a file appears in stack, its module gets +1
    - if not found exactly, try suffix match (repo logs often show partial paths)
    """
    mod_score: Dict[str, float] = {}

    def add(m: str, v: float):
        mod_score[m] = mod_score.get(m, 0.0) + v

    # build suffix index for fallback matching
    keys = list(file2mod.keys())

    for sf in stack_files:
        if sf in file2mod:
            add(file2mod[sf], 1.0)
            continue

        # suffix match: find the longest match
        best = None
        best_len = 0
        for k in keys:
            if k.endswith(sf) or sf.endswith(k):
                L = min(len(k), len(sf))
                if L > best_len:
                    best = k
                    best_len = L
        if best is not None:
            add(file2mod[best], 0.7)  # weaker confidence
    return mod_score


def score_by_symptoms(text: str, symptom_lexicon: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """
    Returns:
      score: {module: score}
      evidence: {module: [matched keywords]}
    """
    tl = text.lower()
    score: Dict[str, float] = {}
    evidence: Dict[str, List[str]] = {}

    for mod, kwmap in symptom_lexicon.items():
        for kw, w in kwmap.items():
            kw_l = kw.lower()
            if kw_l in tl:
                score[mod] = score.get(mod, 0.0) + float(w)
                evidence.setdefault(mod, []).append(kw)

    # stable evidence order
    for mod in evidence:
        evidence[mod] = sorted(set(evidence[mod]))
    return score, evidence


def merge_scores(
    stack_score: Dict[str, float],
    symptom_score: Dict[str, float],
    alpha: float = 0.6
) -> Dict[str, float]:
    """
    Susp(M) = alpha*stack + (1-alpha)*symptom
    """
    mods = set(stack_score.keys()) | set(symptom_score.keys())
    out: Dict[str, float] = {}
    for m in mods:
        out[m] = alpha * \
            stack_score.get(m, 0.0) + (1 - alpha) * symptom_score.get(m, 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Crash log file path")
    ap.add_argument("--modules_jsonl", required=True,
                    help="Output from classify_repo.py (modules.jsonl)")
    ap.add_argument("--symptoms", required=True,
                    help="Symptom lexicon JSON (kb/module_symptoms.json)")
    ap.add_argument("--kb", default=None,
                    help="Optional module KB json (kb/modules_kb.json) for names/desc")
    ap.add_argument("--out", default="topk_modules.json")
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    log_text = Path(args.log).read_text(encoding="utf-8", errors="ignore")
    file2mod = load_modules_jsonl(Path(args.modules_jsonl))
    symptom_lexicon = json.loads(
        Path(args.symptoms).read_text(encoding="utf-8"))

    kb = None
    if args.kb:
        kb = json.loads(Path(args.kb).read_text(encoding="utf-8"))

    stack_files = extract_files_from_log(log_text)
    stack_score = score_by_stack(stack_files, file2mod)

    symptom_score, symptom_evidence = score_by_symptoms(
        log_text, symptom_lexicon)
    merged = merge_scores(stack_score, symptom_score, alpha=args.alpha)

    ranking = sorted(
        merged.items(), key=lambda kv: (-kv[1], kv[0]))[: args.topk]

    def mod_name(mod_id: str) -> str:
        if kb and mod_id in kb and "name" in kb[mod_id]:
            return kb[mod_id]["name"]
        return mod_id

    out_obj: Dict[str, Any] = {
        "alpha": args.alpha,
        "topk": args.topk,
        "stack_files_detected": stack_files[:50],
        "scores": {
            "stack": stack_score,
            "symptom": symptom_score,
            "merged": merged
        },
        "top_modules": []
    }

    for mid, sc in ranking:
        out_obj["top_modules"].append({
            "module": mid,
            "module_name": mod_name(mid),
            "score": sc,
            "symptom_keywords_matched": symptom_evidence.get(mid, [])
        })

    Path(args.out).write_text(json.dumps(
        out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote -> {args.out}")


if __name__ == "__main__":
    main()
