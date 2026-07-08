#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild statement-level dataset with CAE rule-based features."""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]

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
    (r'(?:file\s*not\s*found|cannot\s*open|no\such\s*file)', 'io_error'),
    (r'(?:permission\s*denied|access\s*denied)', 'permission_error'),
    (r'(?:assert(?:ion)?\s*(?:fail|error|abort))', 'assert_fail'),
    (r'(?:timeout|timed?\s*out)', 'timeout'),
    (r'(?:deadlock|race\s*condition|thread)', 'concurrency'),
]

ALL_PATTERNS = NUMERICAL_PATTERNS + SOLVER_PATTERNS + PHYSICS_PATTERNS + SYSTEM_PATTERNS

def count_patterns(text: str, patterns: List[Tuple[str, str]]) -> Dict[str, int]:
    counts = {}
    for pattern, name in patterns:
        counts[name] = len(re.findall(pattern, text, re.IGNORECASE))
    return counts

def extract_cae_from_text(text: str) -> Dict[str, float]:
    text = text[:50000]
    numerical = count_patterns(text, NUMERICAL_PATTERNS)
    solver = count_patterns(text, SOLVER_PATTERNS)
    physics = count_patterns(text, PHYSICS_PATTERNS)
    system = count_patterns(text, SYSTEM_PATTERNS)
    feats = {}
    for name, count in numerical.items(): feats[f'num_{name}'] = min(float(count), 5.0)
    for name, count in solver.items(): feats[f'solver_{name}'] = min(float(count), 5.0)
    for name, count in physics.items(): feats[f'phys_{name}'] = min(float(count), 5.0)
    for name, count in system.items(): feats[f'sys_{name}'] = min(float(count), 5.0)
    return feats

def extract_line_cae(line_text: str, context_text: str) -> Dict[str, float]:
    combined = (line_text + ' ' + context_text).lower()
    feats = {}
    for pattern, name in ALL_PATTERNS:
        matches = re.findall(pattern, combined, re.IGNORECASE)
        feats[f'line_{name}'] = 1.0 if matches else 0.0
    return feats

def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line: rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='ranker/data/manual_line_stage3_light')
    parser.add_argument('--output-dir', default='ranker/data/manual_line_stage3_cae')
    args = parser.parse_args()

    input_dir = ROOT / args.input_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'valid', 'test']:
        rows = read_jsonl(input_dir / f'{split}.jsonl')
        updated = 0
        for row in rows:
            bug_text = row.get('bug_text', '')
            line_text = ''
            context_text = row.get('file_text', '')

            bug_cae = extract_cae_from_text(bug_text)
            line_cae = extract_line_cae(line_text, context_text)

            new_features = {}
            for k, v in row['features'].items():
                if k == 'domain_keyword_overlap':
                    continue
                new_features[k] = v
            for k, v in bug_cae.items():
                new_features[k] = v
            for k, v in line_cae.items():
                new_features[k] = v

            row['features'] = new_features
            updated += 1

        write_jsonl(output_dir / f'{split}.jsonl', rows)
        feat_count = len(rows[0]['features']) if rows else 0
        print(f'{split}: {len(rows)} rows, {feat_count} features -> {output_dir / split}.jsonl')

    # Write all.jsonl
    all_rows = []
    for split in ['train', 'valid', 'test']:
        all_rows.extend(read_jsonl(output_dir / f'{split}.jsonl'))
    write_jsonl(output_dir / 'all.jsonl', all_rows)
    print(f'all: {len(all_rows)} rows')

if __name__ == '__main__':
    main()
