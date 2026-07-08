#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Enhanced CrashLocator with full feature set.

Features:
1. Stack distance (existing)
2. Call graph expansion via header dependency graph (1-2 hops)
3. Historical changes via git log
4. File frequency across crash reports
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]


def git_log_file_stats(repo_git_dir: Path, ref: str = 'HEAD', months: int = 12) -> Dict[str, Dict[str, int]]:
    cmd = [
        'git', f'--git-dir={repo_git_dir}',
        'log', ref, f'--since={months} months ago',
        '--pretty=format:', '--name-only', '--diff-filter=ACDMR',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    file_counts = Counter()
    for line in result.stdout.split('\n'):
        line = line.strip()
        if line and not line.startswith(':'):
            file_counts[line] += 1
    return dict(file_counts)


def build_dep_graph(records: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    graph = defaultdict(set)
    file_by_dep_token = defaultdict(set)
    for rec in records:
        for dep_token in rec.get('dep_tokens', []):
            file_by_dep_token[dep_token].add(rec['file'])
    for rec in records:
        src = rec['file']
        for dep_token in rec.get('dep_tokens', []):
            for tgt in file_by_dep_token.get(dep_token, []):
                if tgt != src:
                    graph[src].add(tgt)
                    graph[tgt].add(src)
    return graph


def expand_stack_files(stack_files: List[str], dep_graph: Dict[str, Set[str]], hops: int = 2) -> Set[str]:
    expanded = set(stack_files)
    frontier = set(stack_files)
    for _ in range(hops):
        next_frontier = set()
        for f in frontier:
            for neighbor in dep_graph.get(f, []):
                if neighbor not in expanded:
                    next_frontier.add(neighbor)
                    expanded.add(neighbor)
        frontier = next_frontier
    return expanded


def compute_file_frequency(all_dataset_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    file_freq = Counter()
    for row in all_dataset_rows:
        if row.get('label', 0) > 0.5:
            file_freq[row['file']] += 1
    max_freq = max(file_freq.values()) if file_freq else 1
    return {f: c / max_freq for f, c in file_freq.items()}


class EnhancedCrashLocator:
    def __init__(
        self,
        dataset_rows: List[Dict[str, Any]],
        repos_dir: Path,
        cache_dir: Path,
    ):
        self.repos_dir = repos_dir
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.dep_graphs: Dict[str, Dict[str, Set[str]]] = {}
        self.git_stats: Dict[str, Dict[str, int]] = {}
        self.file_freq = compute_file_frequency(dataset_rows)

        projects = set(r.get('project', '') for r in dataset_rows)
        for proj in projects:
            if not proj:
                continue
            dep_cache = cache_dir / f'{proj}_dep_graph.json'
            if dep_cache.exists():
                raw = json.loads(dep_cache.read_text())
                self.dep_graphs[proj] = {k: set(v) for k, v in raw.items()}
            else:
                records = [r for r in dataset_rows if r.get('project') == proj]
                self.dep_graphs[proj] = build_dep_graph(records)
                raw = {k: list(v) for k, v in self.dep_graphs[proj].items()}
                dep_cache.write_text(json.dumps(raw))

            git_cache = cache_dir / f'{proj}_git_stats.json'
            if git_cache.exists():
                self.git_stats[proj] = json.loads(git_cache.read_text())
            else:
                repo_dir = repos_dir / proj
                if repo_dir.exists():
                    self.git_stats[proj] = git_log_file_stats(repo_dir)
                    git_cache.write_text(json.dumps(self.git_stats[proj]))
                else:
                    self.git_stats[proj] = {}

    def score(self, row: Dict[str, Any]) -> float:
        project = row.get('project', '')
        file_path = row['file']
        bug_text = row.get('bug_text', '')

        stack_files_raw = row.get('bug_meta', {}).get('stack_files', [])
        if not stack_files_raw:
            stack_files_raw = []
            for line in bug_text.split('\n'):
                line = line.strip()
                if '.f90' in line or '.cpp' in line or '.cc' in line or '.py' in line or '.hpp' in line:
                    parts = line.split()
                    for p in parts:
                        if any(p.endswith(ext) for ext in ['.f90', '.cpp', '.cc', '.py', '.hpp', '.h', '.F90']):
                            stack_files_raw.append(p.split('/')[-1])

        dep_graph = self.dep_graphs.get(project, {})
        expanded = expand_stack_files(stack_files_raw, dep_graph, hops=2)

        stack_score = 0.0
        norm_file = file_path.replace('\\', '/')

        for idx, sf in enumerate(stack_files_raw):
            sf = sf.replace('\\', '/').split('/')[-1]
            if norm_file == sf or norm_file.endswith('/' + sf) or sf.endswith('/' + norm_file.split('/')[-1]):
                frame_weight = 1.0 / math.log2(idx + 2.0)
                stack_score = max(stack_score, 18.0 * frame_weight)

        if file_path in expanded:
            stack_score = max(stack_score, 3.0)

        dep_overlap = float(row.get('features', {}).get('bug_dep_token_overlap', 0))
        name_overlap = float(row.get('features', {}).get('bug_name_token_overlap', 0))
        path_overlap = float(row.get('features', {}).get('bug_path_token_overlap', 0))
        structural = 0.65 * dep_overlap + 0.45 * name_overlap + 0.25 * path_overlap

        git_hits = self.git_stats.get(project, {}).get(file_path, 0)
        git_score = math.log2(git_hits + 1) * 0.5

        freq_score = self.file_freq.get(file_path, 0.0) * 2.0

        return stack_score + structural + git_score + freq_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', default='ranker/data/manual_stage2_deep/all.jsonl')
    parser.add_argument('--test-file', default='ranker/data/manual_stage2_deep/test.jsonl')
    parser.add_argument('--repos-dir', default='repos')
    parser.add_argument('--cache-dir', default='ranker/cache/enhanced_crashlocator')
    parser.add_argument('--output-file', default='ranker/results/predictions/file_crashlocator_enhanced.jsonl')
    args = parser.parse_args()

    root = ROOT
    all_rows = []
    with open(root / args.data_file) as f:
        for line in f:
            line = line.strip()
            if line:
                all_rows.append(json.loads(line))

    test_rows = []
    with open(root / args.test_file) as f:
        for line in f:
            line = line.strip()
            if line:
                test_rows.append(json.loads(line))

    cl = EnhancedCrashLocator(all_rows, root / args.repos_dir, root / args.cache_dir)

    out_rows = []
    for row in test_rows:
        score = cl.score(row)
        out_rows.append({
            'bug_id': row['bug_id'],
            'project': row.get('project', ''),
            'file': row['file'],
            'label': float(row['label']),
            'score': float(score),
            'method': 'CrashLocator',
        })

    out_path = root / args.output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    from collections import defaultdict
    groups = defaultdict(list)
    for r in out_rows:
        groups[r['bug_id']].append(r)
    top1 = 0
    for items in groups.values():
        ranked = sorted(items, key=lambda x: -x['score'])
        pos_ranks = [i+1 for i, x in enumerate(ranked) if x['label'] > 0.5]
        if pos_ranks and min(pos_ranks) <= 1:
            top1 += 1
    print(f'[OK] {len(out_rows)} predictions -> {out_path}')
    print(f'  Top-1: {top1}/{len(groups)}')


if __name__ == '__main__':
    main()
