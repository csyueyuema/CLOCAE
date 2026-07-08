#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unified experiment runner for the current CrashLocCAE experiment suite.

Runs:
1. File-level current model training + evaluation
2. Statement-level current model training + evaluation
3. Ablation studies (no_stack, no_structure, no_domain, wide_only, deep_only)
4. Comparison methods (CrashLocator, Scaffle, LLM-Only)
5. K-fold cross-validation
6. Collects all results into result tables
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd=None):
    print(f'\n[RUN] {" ".join(cmd)}', flush=True)
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        print(f'[WARN] Command exited with code {result.returncode}')
    return result.returncode


def main():
    root = Path('.').resolve()
    python = sys.executable

    file_data = 'ranker/data/manual_stage2_deep'
    line_data = 'ranker/data/manual_line_stage3_light'

    file_out = 'ranker/outputs/wide_deep'
    line_out = 'ranker/outputs/line_ranker'

    file_ablation_dir = 'ranker/outputs/ablations'
    line_ablation_dir = 'ranker/outputs/line_experiments'

    print('=' * 60)
    print('CrashLocCAE - Full Experiment Suite')
    print('=' * 60)

    # =========================================================
    # 1. File-level current model (full)
    # =========================================================
    print('\n' + '=' * 60)
    print('1. File-level current model (full)')
    print('=' * 60)
    run([
        python, 'ranker/train_ranker_current.py',
        '--train-file', f'{file_data}/train.jsonl',
        '--valid-file', f'{file_data}/valid.jsonl',
        '--output-dir', file_out,
        '--model-name', 'microsoft/codebert-base',
        '--epochs', '6',
        '--batch-size', '4',
        '--grad-accum-steps', '2',
        '--lr', '1e-5',
        '--max-length', '384',
        '--dropout', '0.2',
        '--pooling', 'mean',
    ])

    run([
        python, 'ranker/eval_ranker_current.py',
        '--model-dir', file_out,
        '--test-file', f'{file_data}/test.jsonl',
        '--output-file', f'{file_out}/test_metrics.json',
        '--predictions-file', f'{file_out}/predictions.jsonl',
    ])

    # =========================================================
    # 2. Statement-level current model (full)
    # =========================================================
    print('\n' + '=' * 60)
    print('2. Statement-level current model (full)')
    print('=' * 60)
    run([
        python, 'ranker/train_line_ranker_current.py',
        '--train-file', f'{line_data}/train.jsonl',
        '--valid-file', f'{line_data}/valid.jsonl',
        '--output-dir', line_out,
        '--model-name', 'microsoft/codebert-base',
        '--epochs', '4',
        '--batch-size', '8',
        '--grad-accum-steps', '2',
        '--lr', '1e-5',
        '--max-length', '256',
        '--dropout', '0.2',
        '--pooling', 'mean',
    ])

    run([
        python, 'ranker/eval_line_ranker_current.py',
        '--model-dir', line_out,
        '--test-file', f'{line_data}/test.jsonl',
        '--output-file', f'{line_out}/test_metrics.json',
        '--predictions-file', f'{line_out}/predictions.jsonl',
    ])

    # =========================================================
    # 3. File-level ablation studies
    # =========================================================
    print('\n' + '=' * 60)
    print('3. File-level ablation studies')
    print('=' * 60)

    file_ablation_configs = [
        ('wide_only', 'Wide Only', ['--disable-deep']),
        ('deep_only', 'Deep Only', ['--disable-wide']),
        ('no_stack', 'w/o Stack', ['--feature-groups', 'structure', 'domain']),
        ('no_structure', 'w/o Structure', ['--feature-groups', 'stack', 'domain']),
        ('no_domain', 'w/o Domain', ['--feature-groups', 'stack', 'structure']),
    ]

    for name, label, extra_args in file_ablation_configs:
        model_dir = f'{file_ablation_dir}/{name}'
        print(f'\n--- Ablation: {label} ---')
        run([
            python, 'ranker/train_ranker_current.py',
            '--train-file', f'{file_data}/train.jsonl',
            '--valid-file', f'{file_data}/valid.jsonl',
            '--output-dir', model_dir,
            '--model-name', 'microsoft/codebert-base',
            '--epochs', '6',
            '--batch-size', '4',
            '--grad-accum-steps', '2',
            '--lr', '1e-5',
            '--max-length', '384',
            '--dropout', '0.2',
            '--pooling', 'mean',
        ] + extra_args)

        run([
            python, 'ranker/eval_ranker_current.py',
            '--model-dir', model_dir,
            '--test-file', f'{file_data}/test.jsonl',
            '--output-file', f'{model_dir}/test_metrics.json',
        ])

    # =========================================================
    # 4. Statement-level ablation studies
    # =========================================================
    print('\n' + '=' * 60)
    print('4. Statement-level ablation studies')
    print('=' * 60)

    line_ablation_configs = [
        ('wide_only', 'Wide Only', ['--disable-deep']),
        ('deep_only', 'Deep Only', ['--disable-wide']),
    ]

    for name, label, extra_args in line_ablation_configs:
        model_dir = f'{line_ablation_dir}/{name}'
        print(f'\n--- Line Ablation: {label} ---')
        run([
            python, 'ranker/train_line_ranker_current.py',
            '--train-file', f'{line_data}/train.jsonl',
            '--valid-file', f'{line_data}/valid.jsonl',
            '--output-dir', model_dir,
            '--model-name', 'microsoft/codebert-base',
            '--epochs', '4',
            '--batch-size', '8',
            '--grad-accum-steps', '2',
            '--lr', '1e-5',
            '--max-length', '256',
            '--dropout', '0.2',
            '--pooling', 'mean',
        ] + extra_args)

        run([
            python, 'ranker/eval_line_ranker_current.py',
            '--model-dir', model_dir,
            '--test-file', f'{line_data}/test.jsonl',
            '--output-file', f'{model_dir}/test_metrics.json',
        ])

    # =========================================================
    # 5. Collect results
    # =========================================================
    print('\n' + '=' * 60)
    print('5. Collecting all results')
    print('=' * 60)

    results = {
        'file_level': {},
        'statement_level': {},
    }

    # File-level main
    try:
        with open(f'{file_out}/test_metrics.json') as f:
            results['file_level']['CrashLocCAE'] = json.load(f)['metrics']
    except FileNotFoundError:
        print('[WARN] File-level current metrics not found')

    # File-level ablations
    for name, label, _ in file_ablation_configs:
        try:
            with open(f'{file_ablation_dir}/{name}/test_metrics.json') as f:
                results['file_level'][label] = json.load(f)['metrics']
        except FileNotFoundError:
            print(f'[WARN] File ablation {name} metrics not found')

    # Statement-level main
    try:
        with open(f'{line_out}/test_metrics.json') as f:
            results['statement_level']['CrashLocCAE'] = json.load(f)['metrics']
    except FileNotFoundError:
        print('[WARN] Statement-level current metrics not found')

    # Statement-level ablations
    for name, label, _ in line_ablation_configs:
        try:
            with open(f'{line_ablation_dir}/{name}/test_metrics.json') as f:
                results['statement_level'][label] = json.load(f)['metrics']
        except FileNotFoundError:
            print(f'[WARN] Line ablation {name} metrics not found')

    # Comparison methods (use existing eval scripts)
    print('\n--- File-level comparison methods ---')
    run([
        python, 'ranker/eval_comparison_methods.py',
        '--test-file', f'{file_data}/test.jsonl',
        '--output-file', 'ranker/results/tables/comparison_methods.json',
    ])

    print('\n--- Statement-level comparison methods ---')
    run([
        python, 'ranker/eval_line_comparison_methods.py',
        '--test-file', f'{line_data}/test.jsonl',
        '--output-file', 'ranker/results/tables/statement_comparison_methods.json',
    ])

    # Load comparison results
    try:
        with open('ranker/results/tables/comparison_methods.json') as f:
            for item in json.load(f):
                results['file_level'][item['method']] = item
    except FileNotFoundError:
        print('[WARN] File comparison methods not found')

    try:
        with open('ranker/results/tables/statement_comparison_methods.json') as f:
            for item in json.load(f):
                results['statement_level'][item['method']] = item
    except FileNotFoundError:
        print('[WARN] Statement comparison methods not found')

    # Save collected results
    out_path = Path('ranker/results/all_results.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nAll results saved to {out_path}')
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
