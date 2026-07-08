#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Re-extract CAE domain features at file-level granularity.

Problem: the original build_ranking_dataset.py computes:
  - symptom_overlap_count: bug text vs MODULE symptom keywords (module-level constant)
  - domain_keyword_overlap: bug domain tokens vs file path/name/dep tokens (too generic)
  - teacher_module_prob: teacher prob for TARGET module (module-level constant)

Fix: recompute as:
  - symptom_file_score: bug keywords vs FILE content, weighted by symptom importance
  - domain_file_overlap: bug domain keywords vs FILE actual code tokens (not just path/name)
  - teacher_file_match: teacher prob for FILE's actual module (per-file varying)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'module_map'))
sys.path.append(str(ROOT / 'scripts'))

from build_ranking_dataset import (
    DEFAULT_MAX_BYTES,
    BareRepoModuleIndex,
    build_bug_tokens,
    git_show_text,
    keyword_hits,
    read_json,
    read_jsonl,
    write_jsonl,
)
from classify_repo import normalize_tokens


RE_TOKEN = re.compile(r"[a-z0-9]{3,}")


def file_content_tokens(file_text: str, max_chars: int = 50000) -> Set[str]:
    text = file_text[:max_chars].lower()
    return set(RE_TOKEN.findall(text))


class CAEFeatureExtractor:
    def __init__(
        self,
        modules_kb_path: Path,
        symptoms_path: Path,
        teacher_path: Optional[Path] = None,
    ):
        self.modules_kb = read_json(modules_kb_path)
        self.symptoms_map = read_json(symptoms_path)
        self.teacher_map = self._load_teacher(teacher_path)
        self.domain_keywords = self._build_domain_keywords()
        self._generic_keywords = self._compute_generic_keywords()

    def _load_teacher(self, path: Optional[Path]) -> Dict[str, Dict]:
        if not path or not path.exists():
            return {}
        rows = read_jsonl(path)
        return {row['bug_id']: row for row in rows if 'bug_id' in row}

    def _build_domain_keywords(self) -> Set[str]:
        kws = set()
        for mod_data in self.modules_kb.values():
            sym = mod_data.get('symptoms', {})
            for group in sym.values():
                if isinstance(group, list):
                    kws.update(x.lower() for x in group if len(x) >= 3)
            for group in ['rootcause_trigger', 'confusion']:
                if isinstance(mod_data.get(group), list):
                    for item in mod_data[group]:
                        kws.update(x for x in normalize_tokens(item) if len(x) >= 3)
        for mod_map in self.symptoms_map.values():
            kws.update(k.lower() for k in mod_map.keys() if len(k) >= 3)
        return kws

    def _compute_generic_keywords(self, threshold: float = 0.3) -> Set[str]:
        module_count = defaultdict(int)
        total_modules = max(len(self.modules_kb), 1)
        for mod_data in self.modules_kb.values():
            sym = mod_data.get('symptoms', {})
            for group in sym.values():
                if isinstance(group, list):
                    for kw in group:
                        module_count[kw.lower()] += 1
            for group in ['rootcause_trigger', 'confusion']:
                if isinstance(mod_data.get(group), list):
                    for item in mod_data[group]:
                        for tok in normalize_tokens(item):
                            if len(tok) >= 3:
                                module_count[tok] += 1
        for mod_map in self.symptoms_map.values():
            for kw in mod_map:
                module_count[kw.lower()] += 1
        generic = {kw for kw, cnt in module_count.items() if cnt / total_modules >= threshold}
        return generic

    def symptom_file_score(
        self,
        bug_text: str,
        file_path: str,
        file_content: str,
        file_module: str,
    ) -> float:
        module_symptoms = self.symptoms_map.get(file_module, {})
        if not module_symptoms:
            return 0.0

        file_tokens = file_content_tokens(file_content)
        file_tokens.update(t.lower() for t in normalize_tokens(Path(file_path).stem))
        for part in Path(file_path).parts:
            file_tokens.update(t.lower() for t in normalize_tokens(part))

        score = 0.0
        bug_lower = bug_text.lower()
        for keyword, weight in module_symptoms.items():
            kw = keyword.lower()
            if kw in bug_lower and kw in file_tokens:
                score += float(weight)
        return score

    def domain_file_overlap(
        self,
        bug_text: str,
        file_content: str,
        file_path: str,
    ) -> float:
        bug_domain = set()
        bug_lower = bug_text.lower()
        for kw in self.domain_keywords:
            if kw in bug_lower:
                bug_domain.add(kw)
        if not bug_domain:
            return 0.0

        file_tokens = file_content_tokens(file_content)
        file_tokens.update(t.lower() for t in normalize_tokens(Path(file_path).stem))
        for part in Path(file_path).parts:
            file_tokens.update(t.lower() for t in normalize_tokens(part))

        effective = bug_domain - self._generic_keywords
        if not effective:
            effective = bug_domain

        return float(len(effective & file_tokens))

    def teacher_file_match(
        self,
        bug_id: str,
        file_module: str,
    ) -> float:
        teacher = self.teacher_map.get(bug_id, {})
        probs_map = teacher.get('teacher_probs_map', {})
        if not probs_map:
            return 0.0
        try:
            return float(probs_map.get(file_module, 0.0))
        except Exception:
            return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True, help='Original dataset JSONL')
    parser.add_argument('--output-file', required=True, help='Output JSONL with updated features')
    parser.add_argument('--modules-kb', default='module_map/configs/modules_kb.json')
    parser.add_argument('--symptoms', default='module_map/kb/module_symptoms.json')
    parser.add_argument('--teacher-file', default='distill/data/manual_seed/teacher_labels.jsonl')
    parser.add_argument('--repos-dir', default='repos')
    parser.add_argument('--cache-dir', default='ranker/cache/module_index')
    parser.add_argument('--config', default='module_map/configs/config_base.json')
    args = parser.parse_args()

    root = ROOT
    extractor = CAEFeatureExtractor(
        modules_kb_path=root / args.modules_kb,
        symptoms_path=root / args.symptoms,
        teacher_path=root / args.teacher_file if args.teacher_file else None,
    )

    rows = read_jsonl(root / args.input_file)
    repo_indices: Dict[str, BareRepoModuleIndex] = {}
    config = read_json(root / args.config)
    repos_dir = root / args.repos_dir
    cache_dir = root / args.cache_dir

    updated = 0
    out_rows = []
    for row in rows:
        project = row.get('project', '')
        file_path = row['file']
        bug_id = row['bug_id']
        file_module = row.get('file_module', row.get('features', {}).get('file_module', ''))

        if project not in repo_indices:
            repo_dir = repos_dir / project
            if repo_dir.exists():
                repo_indices[project] = BareRepoModuleIndex(repo_dir, config, cache_dir)

        index = repo_indices.get(project)
        raw_content = ''
        if index:
            ref = row.get('code_ref', 'HEAD') or 'HEAD'
            raw_content = git_show_text(index.repo_git_dir, file_path, max_bytes=DEFAULT_MAX_BYTES, ref=ref)

        bug_text = row.get('bug_text', '')

        new_symptom = extractor.symptom_file_score(bug_text, file_path, raw_content, file_module)
        new_domain = extractor.domain_file_overlap(bug_text, raw_content, file_path)
        new_teacher = extractor.teacher_file_match(bug_id, file_module)

        old_s = row['features'].get('symptom_overlap_count', 0)
        old_d = row['features'].get('domain_keyword_overlap', 0)
        old_t = row['features'].get('teacher_module_prob', 0)

        row['features']['symptom_overlap_count'] = round(new_symptom, 4)
        row['features']['domain_keyword_overlap'] = round(new_domain, 4)
        row['features']['teacher_module_prob'] = round(new_teacher, 6)

        if new_symptom != old_s or new_domain != old_d or new_teacher != old_t:
            updated += 1

        if 'debug' in row:
            row['debug']['symptom_file_score'] = round(new_symptom, 4)
            row['debug']['domain_file_overlap'] = round(new_domain, 4)
            row['debug']['teacher_file_match'] = round(new_teacher, 6)

        out_rows.append(row)

    out_path = root / args.output_file
    write_jsonl(out_path, out_rows)
    print(f'[OK] Updated {updated}/{len(rows)} rows -> {out_path}')


if __name__ == '__main__':
    main()
