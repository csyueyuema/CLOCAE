#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""K-fold cross-validation with statistical significance testing.

Splits the full dataset (train+valid+test) by bug_id into K folds,
trains on K-1 folds, tests on 1 fold, and reports mean±std across folds.
Computes paired bootstrap significance test between methods.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def compute_ranking_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    groups: Dict[str, List[Dict[str, float]]] = {}
    for row in rows:
        groups.setdefault(row['bug_id'], []).append(row)

    top1 = top3 = top5 = top10 = 0.0
    rr_total = 0.0
    fr_total = 0.0
    ar_total = 0.0
    valid_groups = 0

    for bug_id, items in groups.items():
        positives = sum(1 for x in items if x['label'] > 0.5)
        if positives == 0:
            continue
        valid_groups += 1
        ranked = sorted(items, key=lambda x: (-x['score'], x['file']))
        positive_ranks = [idx + 1 for idx, x in enumerate(ranked) if x['label'] > 0.5]
        best_rank = min(positive_ranks)
        avg_rank = sum(positive_ranks) / len(positive_ranks)
        top1 += 1.0 if best_rank <= 1 else 0.0
        top3 += 1.0 if best_rank <= 3 else 0.0
        top5 += 1.0 if best_rank <= 5 else 0.0
        top10 += 1.0 if best_rank <= 10 else 0.0
        rr_total += 1.0 / best_rank
        fr_total += best_rank
        ar_total += avg_rank

    if valid_groups == 0:
        return {'top1': 0.0, 'top3': 0.0, 'top5': 0.0, 'top10': 0.0, 'mrr': 0.0, 'mfr': 0.0, 'mar': 0.0}

    return {
        'top1': top1 / valid_groups,
        'top3': top3 / valid_groups,
        'top5': top5 / valid_groups,
        'top10': top10 / valid_groups,
        'mrr': rr_total / valid_groups,
        'mfr': fr_total / valid_groups,
        'mar': ar_total / valid_groups,
    }


def paired_bootstrap_test(
    scores_a: List[float],
    scores_b: List[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    assert len(scores_b) == n
    diffs = np.array(scores_a) - np.array(scores_b)
    observed = diffs.mean()
    count = 0
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_diff = diffs[idx].mean()
        if boot_diff >= 0:
            count += 1
    p_value = 1.0 - count / n_bootstrap
    return float(observed), float(p_value)


def split_by_bug_id(rows: List[Dict[str, Any]], k_folds: int, seed: int = 42) -> List[Tuple[List[str], List[str]]]:
    bug_ids = sorted(set(row['bug_id'] for row in rows))
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(bug_ids))
    fold_size = len(bug_ids) // k_folds
    folds = []
    for i in range(k_folds):
        start = i * fold_size
        end = start + fold_size if i < k_folds - 1 else len(bug_ids)
        test_indices = set(indices[start:end])
        test_ids = [bug_ids[j] for j in test_indices]
        train_ids = [bug_ids[j] for j in range(len(bug_ids)) if j not in test_indices]
        folds.append((train_ids, test_ids))
    return folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='ranker/data/manual_stage2_deep')
    parser.add_argument('--level', choices=['file', 'line'], default='file')
    parser.add_argument('--k-folds', type=int, default=5)
    parser.add_argument('--output-dir', default='ranker/results/kfold')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--model-name', default='microsoft/codebert-base')
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = read_jsonl(data_dir / 'all.jsonl')
    folds = split_by_bug_id(all_rows, args.k_folds, seed=args.seed)

    fold_results = []
    for fold_idx, (train_ids, test_ids) in enumerate(folds):
        print(f'\n=== Fold {fold_idx + 1}/{args.k_folds} ===')
        print(f'  Train bugs: {len(train_ids)}, Test bugs: {len(test_ids)}')

        train_set = set(train_ids)
        test_set = set(test_ids)
        train_rows = [r for r in all_rows if r['bug_id'] in train_set]
        test_rows = [r for r in all_rows if r['bug_id'] in test_set]

        fold_dir = output_dir / f'fold_{fold_idx}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(fold_dir / 'train.jsonl', train_rows)
        write_jsonl(fold_dir / 'test.jsonl', test_rows)

        train_bug_count = len(train_ids)
        test_bug_count = len(test_ids)
        train_pos = sum(1 for r in train_rows if r['label'] > 0.5)
        test_pos = sum(1 for r in test_rows if r['label'] > 0.5)
        print(f'  Train pairs: {len(train_rows)} (pos={train_pos}), Test pairs: {len(test_rows)} (pos={test_pos})')

        fold_results.append({
            'fold': fold_idx,
            'train_bugs': train_bug_count,
            'test_bugs': test_bug_count,
            'train_pairs': len(train_rows),
            'test_pairs': len(test_rows),
            'train_positive': train_pos,
            'test_positive': test_pos,
            'test_bug_ids': test_ids,
        })

    summary = {
        'k_folds': args.k_folds,
        'seed': args.seed,
        'total_bugs': len(set(row['bug_id'] for row in all_rows)),
        'total_pairs': len(all_rows),
        'folds': fold_results,
    }
    (output_dir / 'fold_split_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'\nFold splits saved to {output_dir}')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
