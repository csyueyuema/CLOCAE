#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Re-extract CAE domain features following the paper's original design.

Paper Section 4.2.2 Table 2:
- CAE domain features are rule-based indicators extracted from crash reports
- Binary or count-based indicators for:
  1. Numerical anomalies: Jacobian failure, singular matrix, NaN/Inf, residual divergence
  2. Solver-stage indicators: assembly, nonlinear solve, linear solve, update, time step
  3. Physics/geometry-related errors
  4. System-level anomalies: MPI, memory, I/O failures
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]

NUMERICAL_PATTERNS = [
    (r'jacobian', 'jacobian'),
    (r'singular\s*(matrix|system|block)', 'singular'),
    (r'(?:^|[^a-z])(?:nan|NaN)(?:[^a-z]|$)', 'nan'),
    (r'(?:^|[^a-z])(?:inf|Inf|infinity)(?:[^a-z]|$)', 'inf'),
    (r'floating\s*point\s*(exception|error)', 'floating_point'),
    (r'residual\s*(diverge|diverged|divergence)', 'residual_diverge'),
    (r'ill[\s-]*condition', 'ill_condition'),
    (r'rank[\s-]*deficient', 'rank_deficient'),
    (r'(factorization|decompose)\s*(fail|error)', 'factorization_fail'),
    (r'pivot', 'pivot'),
    (r'(cholesky|lu|qr)\s*(fail|error|singular)', 'decomposition_fail'),
    (r'(nan|inf)\s*(detected|found|encountered)', 'nan_inf_detected'),
    (r'determinant\s*(?:is\s*)?(?:zero|singular|inf|nan)', 'determinant_zero'),
]

SOLVER_PATTERNS = [
    (r'(?:nonlinear|non-linear)\s*solve', 'nonlinear_solve'),
    (r'linear\s*solve', 'linear_solve'),
    (r'assembly', 'assembly'),
    (r'time\s*step', 'time_step'),
    (r'(?:convergence|converge)\s*(check|fail|failed|criterion)', 'convergence'),
    (r'(?:newton|picard)\s*(iteration|method|fail)', 'newton'),
    (r'(?:line[\s-]*search|backtrack)', 'line_search'),
    (r'(?:precondition|precond)', 'preconditioner'),
    (r'(?:krylov|gmres|cg\b|bicg|fgmres)', 'krylov'),
    (r'(?:update|update\s*solution)', 'update'),
    (r'(?:setup|initializ|init)\s*(?:solver|precondition|ksp)', 'solver_setup'),
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


def count_pattern_matches(text: str, patterns: List[Tuple[str, str]]) -> Dict[str, int]:
    counts = {}
    for pattern, name in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        counts[name] = len(matches)
    return counts


def extract_cae_indicators(bug_text: str) -> Dict[str, float]:
    text = bug_text[:50000]

    numerical = count_pattern_matches(text, NUMERICAL_PATTERNS)
    solver = count_pattern_matches(text, SOLVER_PATTERNS)
    physics = count_pattern_matches(text, PHYSICS_PATTERNS)
    system = count_pattern_matches(text, SYSTEM_PATTERNS)

    features = {}
    for name, count in numerical.items():
        features[f'num_{name}'] = min(float(count), 5.0)
    for name, count in solver.items():
        features[f'solver_{name}'] = min(float(count), 5.0)
    for name, count in physics.items():
        features[f'phys_{name}'] = min(float(count), 5.0)
    for name, count in system.items():
        features[f'sys_{name}'] = min(float(count), 5.0)

    features['num_anomaly_count'] = sum(numerical.values())
    features['solver_stage_count'] = sum(solver.values())
    features['physics_error_count'] = sum(physics.values())
    features['system_anomaly_count'] = sum(system.values())

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--output-file', required=True)
    args = parser.parse_args()

    root = ROOT
    in_path = root / args.input_file
    out_path = root / args.output_file

    rows = []
    with open(in_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    updated = 0
    for row in rows:
        bug_text = row.get('bug_text', '')
        cae_feats = extract_cae_indicators(bug_text)

        old_features = row['features']
        new_features = {}
        for key, val in old_features.items():
            if key in ('symptom_overlap_count', 'domain_keyword_overlap', 'teacher_module_prob'):
                continue
            new_features[key] = val
        for key, val in cae_feats.items():
            new_features[key] = val
        row['features'] = new_features

        if 'debug' in row:
            row['debug']['cae_indicators'] = cae_feats
        updated += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    num_keys = len([k for k in rows[0]['features'] if k.startswith('num_')])
    solver_keys = len([k for k in rows[0]['features'] if k.startswith('solver_')])
    phys_keys = len([k for k in rows[0]['features'] if k.startswith('phys_')])
    sys_keys = len([k for k in rows[0]['features'] if k.startswith('sys_')])
    print(f'[OK] Updated {updated} rows -> {out_path}')
    print(f'  Features: {len(rows[0]["features"])} total ({num_keys} numerical, {solver_keys} solver, {phys_keys} physics, {sys_keys} system)')


if __name__ == '__main__':
    main()
