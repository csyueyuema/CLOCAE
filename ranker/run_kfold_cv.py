#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run all methods on each fold and collect per-category results."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

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
        'top1': int(top1), 'top3': int(top3), 'top5': int(top5), 'top10': int(top10),
        'mrr': rr / valid, 'mfr': fr / valid, 'mar': ar / valid, 'bugs': valid,
    }


def run(cmd, cwd=None):
    print(f'  [RUN] {" ".join(cmd[:6])}...', flush=True)
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        print(f'  [WARN] exit code {result.returncode}')
        if result.stderr:
            print(f'  stderr: {result.stderr[:500]}')
    return result.returncode


def main():
    k_folds = 5
    output_dir = ROOT / 'ranker' / 'results' / 'kfold_cv'
    python = sys.executable

    all_rows = read_jsonl(ROOT / 'ranker' / 'data' / 'manual_stage2_deep' / 'all.jsonl')
    proj_map = {r['bug_id']: r.get('project', '') for r in all_rows}

    fold_metrics = {
        'CrashLocCAE': [],
        'LLM-Only': [],
        'CrashLocator': [],
        'Scaffle': [],
        'w/o CAE Features': [],
    }

    for fold_idx in range(k_folds):
        fold_dir = output_dir / f'fold_{fold_idx}'
        train_file = fold_dir / 'train.jsonl'
        test_file = fold_dir / 'test.jsonl'

        if not train_file.exists():
            print(f'[SKIP] Fold {fold_idx}: no data')
            continue

        print(f'\n{"="*60}')
        print(f'FOLD {fold_idx + 1}/{k_folds}')
        print(f'{"="*60}')

        # 1. Train CrashLocCAE
        model_dir = fold_dir / 'model'
        print(f'\n--- Training CrashLocCAE ---')
        run([
            python, 'ranker/train_ranker_paper.py',
            '--train-file', str(train_file),
            '--valid-file', str(test_file),
            '--output-dir', str(model_dir),
            '--model-name', 'distilroberta-base',
            '--epochs', '4', '--batch-size', '4', '--lr', '2e-5', '--max-length', '384',
        ])

        # 2. Evaluate CrashLocCAE
        print(f'\n--- Evaluating CrashLocCAE ---')
        preds_file = fold_dir / 'crashloccae_preds.jsonl'
        run([
            python, 'ranker/eval_ranker_paper.py',
            '--model-dir', str(model_dir),
            '--test-file', str(test_file),
            '--predictions-file', str(preds_file),
        ])

        # 3. Build no-CAE data for w/o CAE Features
        cae_prefixes = ('num_', 'solver_', 'phys_', 'sys_')
        no_cae_rows = []
        for r in read_jsonl(train_file):
            r2 = dict(r)
            r2['features'] = {k: v for k, v in r['features'].items() if not any(k.startswith(p) for p in cae_prefixes)}
            no_cae_rows.append(r2)
        no_cae_train = fold_dir / 'train_no_cae.jsonl'
        with open(no_cae_train, 'w') as f:
            for r in no_cae_rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        no_cae_test_rows = []
        for r in read_jsonl(test_file):
            r2 = dict(r)
            r2['features'] = {k: v for k, v in r['features'].items() if not any(k.startswith(p) for p in cae_prefixes)}
            no_cae_test_rows.append(r2)
        no_cae_test = fold_dir / 'test_no_cae.jsonl'
        with open(no_cae_test, 'w') as f:
            for r in no_cae_test_rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        no_cae_model = fold_dir / 'model_no_cae'
        print(f'\n--- Training w/o CAE Features ---')
        run([
            python, 'ranker/train_ranker_paper.py',
            '--train-file', str(no_cae_train),
            '--valid-file', str(no_cae_test),
            '--output-dir', str(no_cae_model),
            '--model-name', 'distilroberta-base',
            '--epochs', '4', '--batch-size', '4', '--lr', '2e-5', '--max-length', '384',
        ])

        no_cae_preds = fold_dir / 'no_cae_preds.jsonl'
        run([
            python, 'ranker/eval_ranker_paper.py',
            '--model-dir', str(no_cae_model),
            '--test-file', str(no_cae_test),
            '--predictions-file', str(no_cae_preds),
        ])

        # 4. Load predictions and compute metrics per category
        test_rows = read_jsonl(test_file)
        test_pos_bugs = set(r['bug_id'] for r in test_rows if r['label'] > 0.5)

        def load_preds(path):
            if not path.exists():
                return []
            return read_jsonl(path)

        def llm_only_preds(test_rows):
            rankings_path = ROOT / 'ranker' / 'results' / 'predictions' / 'file_llm_only_rankings.jsonl'
            if not rankings_path.exists():
                return []
            rankings = {r['bug_id']: r for r in read_jsonl(rankings_path)}
            scored = []
            for row in test_rows:
                pred = rankings.get(row['bug_id'])
                if not pred:
                    continue
                ranking = pred.get('ranking', [])
                sm = {item: len(ranking) - i for i, item in enumerate(ranking)}
                scored.append({
                    'bug_id': row['bug_id'], 'file': row['file'],
                    'label': float(row['label']), 'score': float(sm.get(row['file'], 0)),
                })
            return scored

        def crashlocator_preds(test_rows):
            from eval_crashlocator_enhanced import EnhancedCrashLocator
            cl = EnhancedCrashLocator(all_rows, ROOT / 'repos', ROOT / 'ranker' / 'cache' / 'enhanced_crashlocator')
            scored = []
            for row in test_rows:
                score = cl.score(row)
                scored.append({
                    'bug_id': row['bug_id'], 'file': row['file'],
                    'label': float(row['label']), 'score': float(score),
                })
            return scored

        def scaffle_preds(test_rows):
            from eval_comparison_methods import BM25Index, scaffle_style, read_jsonl as cl_read
            bm25 = BM25Index(test_rows)
            cache = {}
            scored = []
            for row in test_rows:
                score = scaffle_style(row, bm25, cache)
                scored.append({
                    'bug_id': row['bug_id'], 'file': row['file'],
                    'label': float(row['label']), 'score': float(score),
                })
            return scored

        cae_preds = load_preds(preds_file)
        no_cae_preds_loaded = load_preds(no_cae_preds)
        llm_preds = llm_only_preds(test_rows)
        cl_preds = crashlocator_preds(test_rows)
        sf_preds = scaffle_preds(test_rows)

        methods = {
            'CrashLocCAE': cae_preds,
            'LLM-Only': llm_preds,
            'CrashLocator': cl_preds,
            'Scaffle': sf_preds,
            'w/o CAE Features': no_cae_preds_loaded,
        }

        print(f'\n--- Fold {fold_idx + 1} Results ---')
        for method_name, preds in methods.items():
            if not preds:
                continue
            cat_results = {}
            for cat_id in ['1', '2', '3', '4']:
                cat_bugs = {bid for bid in test_pos_bugs if cat_for(proj_map.get(bid, '')) == cat_id}
                cat_preds = [r for r in preds if r['bug_id'] in cat_bugs]
                m = compute_metrics(cat_preds)
                cat_results[cat_id] = m

            all_m = compute_metrics(preds)
            cat_results['overall'] = all_m
            fold_metrics[method_name].append(cat_results)

            top1_str = ' '.join(f'{cat_results[c]["top1"]:>3}' for c in ['1', '2', '3', '4'])
            print(f'  {method_name:<20} per_cat_top1=[{top1_str}] overall_top1={all_m["top1"]:>3} mar={all_m["mar"]:.2f}')

    # Collect mean±std
    print(f'\n{"="*80}')
    print('5-fold CV results: mean +/- std')
    print(f'{"="*80}')

    summary = {}
    for method_name, fold_list in fold_metrics.items():
        if not fold_list:
            continue
        summary[method_name] = {}
        for cat_id in ['1', '2', '3', '4', 'overall']:
            key = cat_id if cat_id != 'overall' else 'overall'
            for metric in ['top1', 'top5', 'top10', 'mar', 'mfr', 'bugs']:
                vals = [f.get(key, {}).get(metric, 0) for f in fold_list]
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                summary[method_name][f'{key}_{metric}_mean'] = float(mean_val)
                summary[method_name][f'{key}_{metric}_std'] = float(std_val)

    # Print table
    header = f'{"Method":<20} {"Top-1":>12} {"Top-5":>12} {"Top-10":>13} {"MAR":>14} {"MFR":>14} {"Bugs":>6}'
    for cat_id, cat_name in CAT_NAMES.items():
        print(f'\n--- {cat_id}. {cat_name} ---')
        print(header)
        for method_name in fold_metrics:
            if not fold_metrics[method_name]:
                continue
            d = summary[method_name]
            t1m = d.get(f'{cat_id}_top1_mean', 0); t1s = d.get(f'{cat_id}_top1_std', 0)
            t5m = d.get(f'{cat_id}_top5_mean', 0); t5s = d.get(f'{cat_id}_top5_std', 0)
            t10m = d.get(f'{cat_id}_top10_mean', 0); t10s = d.get(f'{cat_id}_top10_std', 0)
            marm = d.get(f'{cat_id}_mar_mean', 0); mars = d.get(f'{cat_id}_mar_std', 0)
            mfrm = d.get(f'{cat_id}_mfr_mean', 0); mfrs = d.get(f'{cat_id}_mfr_std', 0)
            bugs = d.get(f'{cat_id}_bugs_mean', 0)
            print(f'{method_name:<20} {t1m:>5.1f}±{t1s:<4.1f} {t5m:>5.1f}±{t5s:<4.1f} {t10m:>5.1f}±{t10s:<5.1f} {marm:>6.2f}±{mars:<5.2f} {mfrm:>6.2f}±{mfrs:<5.2f} {bugs:>5.0f}')

    print(f'\n--- Overall ---')
    print(header)
    for method_name in fold_metrics:
        if not fold_metrics[method_name]:
            continue
        d = summary[method_name]
        t1m = d.get('overall_top1_mean', 0); t1s = d.get('overall_top1_std', 0)
        t5m = d.get('overall_top5_mean', 0); t5s = d.get('overall_top5_std', 0)
        t10m = d.get('overall_top10_mean', 0); t10s = d.get('overall_top10_std', 0)
        marm = d.get('overall_mar_mean', 0); mars = d.get('overall_mar_std', 0)
        mfrm = d.get('overall_mfr_mean', 0); mfrs = d.get('overall_mfr_std', 0)
        bugs = d.get('overall_bugs_mean', 0)
        print(f'{method_name:<20} {t1m:>5.1f}±{t1s:<4.1f} {t5m:>5.1f}±{t5s:<4.1f} {t10m:>5.1f}±{t10s:<5.1f} {marm:>6.2f}±{mars:<5.2f} {mfrm:>6.2f}±{mfrs:<5.2f} {bugs:>5.0f}')

    out_path = output_dir / 'kfold_results.json'
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'\n[OK] Results saved to {out_path}')


if __name__ == '__main__':
    main()
