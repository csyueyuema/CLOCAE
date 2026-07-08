#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild statement-level 5-fold data with per-fold file scores (no leakage)."""

from __future__ import annotations
import json, sys, os, re, torch
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ranker'))

import numpy as np
from train_ranker_gate import RankingDataset, WideDeepRanker, collate_fn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

NUMERICAL_PATTERNS = [
    (r'jacobian', 'jacobian'), (r'singular\s*(matrix|system|block)', 'singular'),
    (r'(?:^|[^a-z])(?:nan|NaN)(?:[^a-z]|$)', 'nan'), (r'(?:^|[^a-z])(?:inf|Inf|infinity)(?:[^a-z]|$)', 'inf'),
    (r'floating\s*point\s*(exception|error)', 'floating_point'), (r'residual\s*(diverge|diverged|divergence)', 'residual_diverge'),
    (r'ill[\s-]*condition', 'ill_condition'), (r'rank[\s-]*deficient', 'rank_deficient'),
    (r'(factorization|decompose)\s*(fail|error)', 'factorization_fail'), (r'pivot', 'pivot'),
    (r'(cholesky|lu|qr)\s*(fail|error|singular)', 'decomposition_fail'),
    (r'(nan|inf)\s*(detected|found|encountered)', 'nan_inf_detected'),
    (r'determinant\s*(?:is\s*)?(?:zero|singular|inf|nan)', 'determinant_zero'),
]
SOLVER_PATTERNS = [
    (r'(?:nonlinear|non-linear)\s*solve', 'nonlinear_solve'), (r'linear\s*solve', 'linear_solve'),
    (r'assembly', 'assembly'), (r'time\s*step', 'time_step'),
    (r'(?:convergence|converge)\s*(check|fail|failed|criterion)', 'convergence'),
    (r'(?:newton|picard)\s*(iteration|method|fail)', 'newton'), (r'(?:line[\s-]*search|backtrack)', 'line_search'),
    (r'(?:precondition|precond)', 'preconditioner'), (r'(?:krylov|gmres|cg\b|bicg|fgmres)', 'krylov'),
    (r'(?:update|update\s*solution)', 'update'), (r'(?:setup|initializ|init)\s*(?:solver|precondition|ksp)', 'solver_setup'),
]
PHYSICS_PATTERNS = [
    (r'(?:mesh|grid)\s*(quality|distort|degenerat|invert|invalid)', 'mesh_quality'),
    (r'(?:negative|zero)\s*(?:volume|area|length|jacobian)', 'negative_volume'),
    (r'(?:element|cell)\s*(invert|degenerat|distort)', 'element_invert'),
    (r'(?:boundary|bc)\s*(condition|error|mis)', 'boundary_error'),
    (r'(?:material|constitutive)\s*(model|error|fail)', 'material_error'),
    (r'(?:geometr|cad)\s*(error|fail|invalid)', 'geometry_error'),
    (r'(?:dof|degree[\s-]*of[\s-]*freedom)\s*(error|mis)', 'dof_error'),
    (r'(?:interpolat|quadrat|shape[\s-]*function)', 'interpolation'),
]
SYSTEM_PATTERNS = [
    (r'(?:mpi|MPI)\s*(error|fail|abort|deadlock|hang)', 'mpi_error'),
    (r'(?:mpi|MPI)\s*(?:rank|process|comm)', 'mpi_mention'),
    (r'(?:segmentation\s*fault|segfault|sigsegv)', 'segfault'),
    (r'(?:out\s*of\s*memory|oom|alloc.*fail|bad_alloc)', 'memory_error'),
    (r'(?:stack\s*(overflow|exhaust))', 'stack_overflow'),
    (r'(?:dlopen|symbol\s*not\s*found|undefined\s*symbol)', 'linker_error'),
    (r'(?:file\s*not\s*found|cannot\s*open|no\s*such\s*file)', 'io_error'),
    (r'(?:permission\s*denied|access\s*denied)', 'permission_error'),
    (r'(?:assert(?:ion)?\s*(?:fail|error|abort))', 'assert_fail'),
    (r'(?:timeout|timed?\s*out)', 'timeout'),
    (r'(?:deadlock|race\s*condition|thread)', 'concurrency'),
]
ALL_PATTERNS = NUMERICAL_PATTERNS + SOLVER_PATTERNS + PHYSICS_PATTERNS + SYSTEM_PATTERNS

def count_patterns(text, patterns):
    counts = {}
    for p, name in patterns:
        counts[name] = len(re.findall(p, text, re.IGNORECASE))
    return counts

def extract_cae_from_text(text):
    text = text[:50000]
    n = count_patterns(text, NUMERICAL_PATTERNS)
    s = count_patterns(text, SOLVER_PATTERNS)
    p = count_patterns(text, PHYSICS_PATTERNS)
    sy = count_patterns(text, SYSTEM_PATTERNS)
    feats = {}
    for name, c in n.items(): feats[f'num_{name}'] = min(float(c), 5.0)
    for name, c in s.items(): feats[f'solver_{name}'] = min(float(c), 5.0)
    for name, c in p.items(): feats[f'phys_{name}'] = min(float(c), 5.0)
    for name, c in sy.items(): feats[f'sys_{name}'] = min(float(c), 5.0)
    return feats

def extract_line_cae(line_text, context_text):
    combined = (line_text + ' ' + context_text).lower()
    feats = {}
    for p, name in ALL_PATTERNS:
        feats[f'line_{name}'] = 1.0 if re.findall(p, combined, re.IGNORECASE) else 0.0
    return feats

def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line: rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')

def predict_file_scores_for_bugs(model_dir, file_data_path, bug_ids_set):
    ckpt = torch.load(model_dir / 'ranker.pt', map_location='cpu')
    feature_keys = ckpt['feature_keys']
    args = ckpt['args']
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'encoder')
    device = torch.device('cuda')
    model = WideDeepRanker(
        model_name=str(model_dir / 'encoder'), wide_dim=len(feature_keys),
        deep_hidden=args.get('deep_hidden', 256), cae_gate=True, cae_start_idx=7
    ).to(device)
    model.load_state_dict(ckpt['state_dict']); model.eval()

    all_file_rows = read_jsonl(file_data_path)
    filtered = [r for r in all_file_rows if r['bug_id'] in bug_ids_set]
    if not filtered:
        return {}

    ds = RankingDataset.__new__(RankingDataset)
    ds.rows = filtered
    ds.feature_keys = feature_keys
    loader = DataLoader(ds, batch_size=4, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.get('max_length', 384), True))

    scores = {}
    with torch.no_grad():
        for batch in loader:
            bug_ids = batch.pop('bug_ids')
            files = batch.pop('files')
            logits = model(input_ids=batch['input_ids'].to(device),
                          attention_mask=batch['attention_mask'].to(device),
                          wide_features=batch['wide_features'].to(device))
            s = torch.sigmoid(logits).cpu().tolist()
            for bid, f, score in zip(bug_ids, files, s):
                scores[(bid, f)] = float(score)
    return scores

def main():
    output_dir = ROOT / 'ranker' / 'data' / 'manual_line_stage3_cae_folded'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original statement data (without file_rank_score)
    stmt_all = read_jsonl(ROOT / 'ranker' / 'data' / 'manual_line_stage3_light' / 'all.jsonl')
    file_all_path = ROOT / 'ranker' / 'data' / 'manual_stage2_deep' / 'all.jsonl'

    # Fold assignment
    file_rows_for_ids = read_jsonl(file_all_path)
    bug_ids = sorted(set(r['bug_id'] for r in file_rows_for_ids))
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(bug_ids))
    k = 5; fold_size = len(bug_ids) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else len(bug_ids)
        test_ids = set(indices[start:end])
        train_ids = set(range(len(bug_ids))) - test_ids
        folds.append(([bug_ids[j] for j in train_ids], [bug_ids[j] for j in test_ids]))

    bug_to_fold = {}
    for fold_idx, (_, test_ids) in enumerate(folds):
        for bid in test_ids:
            bug_to_fold[bid] = fold_idx

    for fold_idx in range(k):
        fold_dir = output_dir / f'fold_{fold_idx}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        _, test_ids = folds[fold_idx]
        test_id_set = set(test_ids)
        train_id_set = set(folds[fold_idx][0])

        # Use corresponding file-level model to compute file scores
        file_model_dir = ROOT / 'ranker' / 'results' / 'kfold_gate' / f'fold_{fold_idx}' / 'model_gate'
        print(f'Fold {fold_idx+1}: computing file scores from {file_model_dir}...')
        file_scores = predict_file_scores_for_bugs(file_model_dir, file_all_path, test_id_set | train_id_set)

        # Rebuild statement features with correct file scores
        train_rows = []
        test_rows = []
        for row in stmt_all:
            bug_id = row['bug_id']
            file_path = row['file'].rsplit(':', 1)[0] if ':' in row['file'] else row['file']
            bug_text = row.get('bug_text', '')
            context_text = row.get('file_text', '')

            # CAE features from bug text
            bug_cae = extract_cae_from_text(bug_text)
            line_cae = extract_line_cae('', context_text)

            # File score from per-fold model
            fs = file_scores.get((bug_id, file_path), 0.0)
            fr = 1.0  # Will be computed from ranking

            # Build features
            old_feats = row.get('features', {})
            new_feats = {
                'file_rank_score': round(fs, 6),
                'file_rank_position': old_feats.get('file_rank_position', 0.0),
                'stack_file_match': old_feats.get('stack_file_match', 0.0),
                'line_no_norm': old_feats.get('line_no_norm', 0.0),
                'line_length': old_feats.get('line_length', 0.0),
                'line_token_overlap': old_feats.get('line_token_overlap', 0.0),
                'context_token_overlap': old_feats.get('context_token_overlap', 0.0),
                'is_comment_line': old_feats.get('is_comment_line', 0.0),
            }
            for k2, v in bug_cae.items():
                new_feats[k2] = v
            for k2, v in line_cae.items():
                new_feats[k2] = v

            new_row = dict(row)
            new_row['features'] = new_feats

            if bug_id in test_id_set:
                test_rows.append(new_row)
            else:
                train_rows.append(new_row)

        write_jsonl(fold_dir / 'train.jsonl', train_rows)
        write_jsonl(fold_dir / 'test.jsonl', test_rows)
        print(f'  train={len(train_rows)} pairs, test={len(test_rows)} pairs, features={len(train_rows[0]["features"])}')

    print(f'\n[OK] Saved to {output_dir}')

if __name__ == '__main__':
    main()
