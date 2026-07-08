#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""OpenAI-compatible LLM-only statement ranking baseline."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from train_line_ranker import compute_ranking_metrics


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_api_config(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def group_by_bug(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["bug_id"]].append(row)
    return grouped


def build_prompt(bug_id: str, bug_text: str, candidates: Sequence[Dict[str, Any]], bug_chars: int = 5000) -> str:
    lines = []
    for idx, row in enumerate(candidates, start=1):
        snippet = str(row.get("file_text", "")).replace("\n", " ")[:260]
        lines.append(f"{idx}. {row['file']} :: {snippet}")
    candidate_text = "\n".join(lines)
    return f"""You are evaluating a crash bug localization task at statement level.

Given only the initial bug report and candidate statements, rank the most suspicious statements.

Rules:
- Return valid JSON only.
- Use exactly this schema: {{"ranking": ["file:line", "file:line", "..."]}}
- Include at least the top 10 most suspicious statement IDs.
- Use candidate statement IDs exactly as provided.

Bug ID:
{bug_id}

Initial bug report:
{bug_text[:bug_chars]}

Candidate statements:
{candidate_text}
"""


def chat_completion(prompt: str, model: str, base_url: str, api_key: str, temperature: float, timeout: int) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "You are a careful software fault localization assistant. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    return obj["choices"][0]["message"]["content"]


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(m.group(0))


def normalize_ranking(response_text: str, candidates: Sequence[str]) -> List[str]:
    obj = extract_json_object(response_text)
    candidate_set = set(candidates)
    ranking = []
    for item in obj.get("ranking", []):
        value = str(item).strip()
        if value in candidate_set and value not in ranking:
            ranking.append(value)
    for value in candidates:
        if value not in ranking:
            ranking.append(value)
    return ranking


def load_existing_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["bug_id"]: row for row in read_jsonl(path) if "bug_id" in row}


def metrics_from_predictions(grouped: Dict[str, List[Dict[str, Any]]], predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    scored = []
    for bug_id, items in grouped.items():
        pred = predictions.get(bug_id)
        if not pred:
            continue
        ranking = pred.get("ranking", [])
        rank_score = {sid: len(ranking) - idx for idx, sid in enumerate(ranking)}
        for row in items:
            scored.append({
                "bug_id": bug_id,
                "file": row["file"],
                "label": float(row["label"]),
                "score": float(rank_score.get(row["file"], 0.0)),
            })
    metrics = compute_ranking_metrics(scored)
    return {
        "method": "llm_only",
        **metrics,
        "num_pairs": len(scored),
        "num_bugs": len({row["bug_id"] for row in scored}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="ranker/data/manual_line_stage3_light/test.jsonl")
    parser.add_argument("--predictions-file", default="ranker/results/predictions/statement_llm_only_rankings.jsonl")
    parser.add_argument("--output-file", default="ranker/results/tables/statement_llm_only_metrics.json")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-config", default="ranker/config/llm_api.local.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--bug-chars", type=int, default=5000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    api_config = load_api_config(Path(args.api_config) if args.api_config else None)
    api_key = api_config.get("api_key") or os.getenv(args.api_key_env)
    args.base_url = api_config.get("base_url", args.base_url)
    args.model = api_config.get("model", args.model)
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env} in the environment.")

    rows = read_jsonl(Path(args.test_file))
    grouped = group_by_bug(rows)
    bug_ids = sorted(grouped)
    if args.limit > 0:
        bug_ids = bug_ids[: args.limit]

    pred_path = Path(args.predictions_file)
    predictions = load_existing_predictions(pred_path)

    for idx, bug_id in enumerate(bug_ids, start=1):
        if bug_id in predictions:
            continue
        items = grouped[bug_id]
        ranked_candidates = sorted(items, key=lambda r: (-float(r.get("heuristic_score", 0.0)), r["file"]))
        candidates = ranked_candidates[: args.max_candidates]
        candidate_ids = [row["file"] for row in candidates]
        bug_text = items[0].get("bug_text", "")
        prompt = build_prompt(bug_id, bug_text, candidates, bug_chars=args.bug_chars)
        print(f"[{idx}/{len(bug_ids)}] LLM-only statement ranking {bug_id} ({len(candidate_ids)} candidates)")
        try:
            response_text = chat_completion(prompt, args.model, args.base_url, api_key, args.temperature, args.timeout)
            ranking = normalize_ranking(response_text, candidate_ids)
            predictions[bug_id] = {
                "bug_id": bug_id,
                "model": args.model,
                "ranking": ranking,
                "raw_response": response_text,
            }
            write_jsonl(pred_path, predictions.values())
            time.sleep(args.sleep)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError) as exc:
            print(f"[WARN] {bug_id}: {exc}")
            write_jsonl(pred_path, predictions.values())

    metrics = metrics_from_predictions(grouped, predictions)
    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
