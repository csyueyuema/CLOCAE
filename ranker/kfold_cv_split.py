#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""5-fold cross-validation for all methods on the full dataset.

Splits all 392 bugs into 5 folds by bug_id, trains CrashLocCAE on 4 folds,
evaluates all methods on the held-out fold, and collects per-category mean±std.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ranker'))

CATEGORY_MAP = {
    'CalculiX-Examples': '1', 'Calculix': '1', 'calculix': '1',
    'deal.II': '1', 'dealii': '1',
    'FireDrake': '1', 'firedrake': '1',
    'FreeFem': '1', 'FreeFEM': '1',
    'gridap': '1', 'Gridap': '1',
    'MFEM': '1', 'mfem': '1',
    'SfePy': '1', 'sfepy': '1',
    'Sparselizard': '1', 'sparselizard': '1',
    'code': '2', 'Code Saturne': '2', 'Code_Saturne': '2', 'Code-Saturne': '2',
    'coolfluid': '2', 'COOLFluiD': '2',
    'FDS': '2', 'fds': '2',
    'Fluidity': '2',
    'Nek5000': '2',
    'SU2': '2', 'su2': '2',
    'xcompact3d': '2', 'Xcompact3d': '2',
    'Goma': '3', 'goma': '3', 'Kratos': '3', 'kratos': '3',
    'OpenModelica': '4', 'openmodelica': '4', 'ROSS': '4', 'ross': '4',
}
CAT_NAMES = {'1': 'FEM', '2': 'CFD', '3': 'Multiphysics', '4': 'Modeling'}


def cat_for(proj: str) -> str:
    if proj in CATEGORY_MAP:
        return CATEGORY_MAP[proj]
    n = proj.lower().replace('_', '').replace('-', '').replace('.', '').replace(' ', '')
    for name, cat in CATEGORY_MAP.items():
        if n == name.lower().replace('_', '').replace('-', '').replace('.', '').replace(' ', ''):
            return cat
    return 'unknown'


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    groups = defaultdict(list)
    for r in rows:
        groups[r['bug_id']].append(r)
    top1 = top3 = top5 = top10 = 0.0
    rr = fr = ar = 0.0
    valid = 0
    for bid, items in groups.items():
        pos = sum(1 for x in items if x['label'] > 0.5)
        if pos == 0:
            continue
        valid += 1
        ranked = sorted(items, key=lambda x: (-x['score'], x['file']))
        pr = [i + 1 for i, x in enumerate(ranked) if x['label'] > 0.5]
        best = min(pr)
        avg = sum(pr) / len(pr)
        top1 += 1 if best <= 1 else 0
        top3 += 1 if best <= 3 else 0
        top5 += 1 if best <= 5 else 0
        top10 += 1 if best <= 10 else 0
        rr += 1.0 / best
        fr += best
        ar += avg
    if valid == 0:
        return {'top1': 0, 'top3': 0, 'top5': 0, 'top10': 0, 'mrr': 0, 'mfr': 0, 'mar': 0, 'bugs': 0}
    return {
        'top1': top1, 'top3': top3, 'top5': top5, 'top10': top10,
        'mrr': rr / valid, 'mfr': fr / valid, 'mar': ar / valid, 'bugs': valid,
    }


def split_folds(bug_ids: List[str], k: int, seed: int) -> List[Tuple[List[str], List[str]]]:
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(bug_ids))
    fold_size = len(bug_ids) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else len(bug_ids)
        test_idx = set(indices[start:end])
        test_ids = [bug_ids[j] for j in test_idx]
        train_ids = [bug_ids[j] for j in range(len(bug_ids)) if j not in test_idx]
        folds.append((train_ids, test_ids))
    return folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-file', default='ranker/data/manual_stage2_deep/all.jsonl')
    parser.add_argument('--repos-dir', default='repos')
    parser.add_argument('--k-folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max-length', type=int, default=384)
    parser.add_argument('--output-dir', default='ranker/results/kfold_cv')
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    all_rows = read_jsonl(ROOT / args.all_file)
    bug_ids = sorted(set(r['bug_id'] for r in all_rows))

    folds = split_folds(bug_ids, args.k_folds, args.seed)

    summary = {
        'k_folds': args.k_folds,
        'seed': args.seed,
        'total_bugs': len(bug_ids),
        'total_pairs': len(all_rows),
        'folds': [],
    }

    for fold_idx, (train_ids, test_ids) in enumerate(folds):
        print(f'\n{"="*60}')
        print(f'Fold {fold_idx + 1}/{args.k_folds}: train={len(train_ids)}, test={len(test_ids)}')
        print(f'{"="*60}')

        fold_dir = output_dir / f'fold_{fold_idx}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_set = set(train_ids)
        test_set = set(test_ids)
        train_rows = [r for r in all_rows if r['bug_id'] in train_set]
        test_rows = [r for r in all_rows if r['bug_id'] in test_set]

        write_jsonl(fold_dir / 'train.jsonl', train_rows)
        write_jsonl(fold_dir / 'test.jsonl', test_rows)

        train_pos = sum(1 for r in train_rows if r['label'] > 0.5)
        test_pos = sum(1 for r in test_rows if r['label'] > 0.5)
        test_pos_bugs = len(set(r['bug_id'] for r in test_rows if r['label'] > 0.5))
        print(f'  Train: {len(train_rows)} pairs ({train_pos} pos), Test: {len(test_rows)} pairs ({test_pos} pos, {test_pos_bugs} bugs with positive)')

        summary['folds'].append({
            'fold': fold_idx,
            'train_bugs': len(train_ids),
            'test_bugs': len(test_ids),
            'train_pairs': len(train_rows),
            'test_pairs': len(test_rows),
            'test_pos_bugs': test_pos_bugs,
        })

    (output_dir / 'fold_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'\nFold splits saved to {output_dir}')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
