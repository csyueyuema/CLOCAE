#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'module_map'))
sys.path.append(str(ROOT / 'scripts'))

from classify_repo import (  # type: ignore
    normalize_tokens,
    extract_deps,
    extract_api_hits,
    score_tokens,
    score_deps_tokens,
    weighted_sum,
    pick,
    init_scores,
)
from crash_analyze import extract_files_from_log  # type: ignore

RE_PRIMARY = re.compile(r"^- Primary:\s*`([^`]+)`", re.MULTILINE)
RE_MODIFIED_SECTION = re.compile(r"^### Modified Files\s*$", re.MULTILINE)
RE_MODIFIED_LINE = re.compile(r"^-\s+(.+?)(?::\s*.+)?$")
RE_DIFF_FILE = re.compile(r"^diff --git a/(.+?) b/(.+)$", re.MULTILINE)
RE_BUGGY_SHA = re.compile(r"^- Buggy SHA \(parent1\):\s*`([a-f0-9]{7,40})`", re.MULTILINE)
RE_SPLIT = re.compile(r"[^a-z0-9_./+-]+")
DEFAULT_MAX_BYTES = 200_000
DEEP_TEXT_MAX_CHARS = 4000
BUG_TEXT_MAX_CHARS = 5000
EXTRA_EXTENSIONS = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.xml', '.txt'}
CAE_SOURCE_EXTENSIONS = {'.jl', '.mo', '.tpl', '.edp'}
INITIAL_REPORT_MARKERS = [
    '## 【Discussion】',
    '## Discussion',
    '### Discussion',
    '## Comments',
    '## Conversation',
    '## Fault Fix Ground Truth',
    '## Manual Module Label',
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def compact_text(text: str, max_chars: int = DEEP_TEXT_MAX_CHARS) -> str:
    text = text.replace("\r", "\n")
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        lines.append(line)
    compact = "\n".join(lines)
    return compact[:max_chars]


def build_file_text(rel_path: str, text: str) -> str:
    path_text = rel_path.replace("/", " ")
    if not text:
        return path_text[:DEEP_TEXT_MAX_CHARS]
    compact = compact_text(text, max_chars=DEEP_TEXT_MAX_CHARS)
    merged = f"PATH: {path_text}\nCONTENT:\n{compact}"
    return merged[:DEEP_TEXT_MAX_CHARS]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_extension_list(value: str) -> List[str]:
    extensions: List[str] = []
    for item in value.split(','):
        ext = item.strip()
        if not ext:
            continue
        if not ext.startswith('.'):
            ext = f'.{ext}'
        extensions.append(ext)
    return sorted(dict.fromkeys(extensions))


def parse_csv_list(value: str) -> List[str]:
    return sorted(dict.fromkeys(item.strip() for item in value.split(',') if item.strip()))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def run_git(repo_git_dir: Path, args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ['git', f'--git-dir={repo_git_dir}'] + args,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore',
        check=False,
    )


def git_tree_files(repo_git_dir: Path, ref: str = 'HEAD') -> List[str]:
    res = run_git(repo_git_dir, ['ls-tree', '-r', '--name-only', ref])
    if res.returncode != 0:
        raise RuntimeError(f'git ls-tree failed for {repo_git_dir}@{ref}: {res.stderr[:200]}')
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def git_show_text(repo_git_dir: Path, rel_path: str, max_bytes: int = DEFAULT_MAX_BYTES, ref: str = 'HEAD') -> str:
    res = run_git(repo_git_dir, ['show', f'{ref}:{rel_path}'])
    if res.returncode != 0:
        return ''
    return res.stdout[:max_bytes]


def initial_report_text(text: str) -> str:
    cut_points = [text.find(marker) for marker in INITIAL_REPORT_MARKERS]
    cut_points = [x for x in cut_points if x != -1]
    if not cut_points:
        return text
    return text[: min(cut_points)].strip()


def section_before_ground_truth(text: str) -> str:
    cut_points = [
        text.find('## Fault Fix Ground Truth'),
        text.find('## Manual Module Label'),
    ]
    cut_points = [x for x in cut_points if x != -1]
    if not cut_points:
        return text
    return text[: min(cut_points)].strip()


def parse_buggy_sha(text: str) -> Optional[str]:
    m = RE_BUGGY_SHA.search(text)
    return m.group(1).strip() if m else None


def cache_safe_ref(ref: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', ref)[:80]


def parse_primary_module(text: str) -> Optional[str]:
    m = RE_PRIMARY.search(text)
    return m.group(1).strip() if m else None


def parse_modified_files(text: str) -> List[str]:
    modified = []
    m = RE_MODIFIED_SECTION.search(text)
    if m:
        tail = text[m.end():].split('## Manual Module Label', 1)[0]
        for line in tail.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith('## '):
                break
            mm = RE_MODIFIED_LINE.match(line)
            if mm:
                modified.append(mm.group(1).strip())
    if modified:
        return sorted(dict.fromkeys(x.replace('\\', '/') for x in modified))

    for a_path, b_path in RE_DIFF_FILE.findall(text):
        if a_path == b_path:
            modified.append(a_path.strip())
        else:
            modified.append(b_path.strip())
    return sorted(dict.fromkeys(x.replace('\\', '/') for x in modified))


def build_bug_tokens(text: str) -> List[str]:
    return [t for t in RE_SPLIT.split(text.lower()) if len(t) >= 2]


def suffix_match_score(file_path: str, stack_files: List[str]) -> float:
    best = 0.0
    for sf in stack_files:
        if file_path == sf:
            return 1.0
        if file_path.endswith(sf) or sf.endswith(file_path):
            shorter = min(len(file_path), len(sf))
            longer = max(len(file_path), len(sf))
            best = max(best, shorter / max(longer, 1))
    return best


def keyword_hits(text: str, keywords: List[str]) -> List[str]:
    tl = text.lower()
    hits = []
    for kw in keywords:
        if kw and kw.lower() in tl:
            hits.append(kw)
    return sorted(set(hits))


class BareRepoModuleIndex:
    def __init__(self, repo_git_dir: Path, config: Dict[str, Any], cache_dir: Path, ref: str = 'HEAD'):
        self.repo_git_dir = repo_git_dir
        self.config = config
        self.ref = ref
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_dir / f'{repo_git_dir.name}_{cache_safe_ref(ref)}_module_index.jsonl'
        self.records = self._load_or_build()
        self.file_text_cache: Dict[str, str] = {}
        self.by_module: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.by_file: Dict[str, Dict[str, Any]] = {}
        for row in self.records:
            self.by_module[row['module']].append(row)
            self.by_file[row['file']] = row

    def _supported_extensions(self) -> set[str]:
        exts = set(self.config.get('extensions', []))
        exts.update(EXTRA_EXTENSIONS)
        return exts

    def _supported_path_patterns(self) -> List[str]:
        return [str(pattern) for pattern in self.config.get('extra_path_patterns', [])]

    def _is_supported_path(self, rel_path: str) -> bool:
        if Path(rel_path).suffix in self._supported_extensions():
            return True
        return any(fnmatch.fnmatch(rel_path, pattern) for pattern in self._supported_path_patterns())

    def _classify_virtual_file(self, rel_path: str, text: str) -> Dict[str, Any]:
        modules = self.config['modules']
        weights = self.config['weights']
        tau = float(self.config.get('tau', 0.15))
        fallback = self.config.get('fallback_module', 'M1_CORE')
        path_hints = self.config.get('path_hints', {})
        name_lexicon = self.config.get('name_lexicon', {})
        api_lexicon = self.config.get('api_lexicon', {})

        parts = Path(rel_path).parts
        path_tokens: List[str] = []
        for part in parts[:-1]:
            path_tokens.extend(normalize_tokens(part))
        name_tokens = normalize_tokens(Path(rel_path).stem)

        ext = Path(rel_path).suffix
        deps = extract_deps(Path(rel_path), text) if text and ext not in EXTRA_EXTENSIONS else []
        api_hits = extract_api_hits(text, api_lexicon) if text else []

        scores = {
            'path': score_tokens(path_tokens, modules, path_hints),
            'name': score_tokens(name_tokens, modules, name_lexicon),
            'dep': score_deps_tokens(deps, modules, name_lexicon, path_hints),
            'api': init_scores(modules),
        }
        for hit in api_hits:
            for mod, w in api_lexicon.get(hit, {}).items():
                scores['api'][mod] += float(w)

        total = weighted_sum(scores, weights, modules)
        module, margin = pick(total, tau, fallback)
        dep_tokens = []
        for dep in deps:
            dep_tokens.extend(normalize_tokens(Path(dep).stem))

        return {
            'file': rel_path,
            'module': module,
            'module_name': modules[module],
            'margin': margin,
            'path_tokens': sorted(set(path_tokens)),
            'name_tokens': sorted(set(name_tokens)),
            'dep_tokens': sorted(set(dep_tokens)),
            'api_hits': api_hits,
        }

    def get_file_text(self, rel_path: str) -> str:
        if rel_path not in self.file_text_cache:
            raw = git_show_text(self.repo_git_dir, rel_path, max_bytes=DEFAULT_MAX_BYTES, ref=self.ref)
            self.file_text_cache[rel_path] = build_file_text(rel_path, raw)
        return self.file_text_cache[rel_path]

    def _load_or_build(self) -> List[Dict[str, Any]]:
        if self.cache_path.exists():
            return read_jsonl(self.cache_path)

        rows: List[Dict[str, Any]] = []
        supported = self._supported_extensions()
        for rel_path in git_tree_files(self.repo_git_dir, self.ref):
            path_obj = Path(rel_path)
            if path_obj.suffix not in supported and not self._is_supported_path(rel_path):
                continue
            text = git_show_text(self.repo_git_dir, rel_path, ref=self.ref)
            rows.append(self._classify_virtual_file(rel_path, text))
        write_jsonl(self.cache_path, rows)
        return rows


class RankingDatasetBuilder:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = Path(args.root).resolve()
        self.cases_dir = self.root / args.cases_dir
        self.repos_dir = self.root / args.repos_dir
        self.output_dir = self.root / args.output_dir
        self.cache_dir = self.root / args.cache_dir
        self.config = read_json(self.root / args.config)
        self.extra_index_extensions = parse_extension_list(getattr(args, 'extra_index_extensions', ''))
        self.extra_index_path_patterns = parse_csv_list(getattr(args, 'extra_index_path_patterns', ''))
        if self.extra_index_extensions:
            extensions = set(self.config.get('extensions', []))
            extensions.update(self.extra_index_extensions)
            self.config['extensions'] = sorted(extensions)
        if self.extra_index_path_patterns:
            patterns = set(self.config.get('extra_path_patterns', []))
            patterns.update(self.extra_index_path_patterns)
            self.config['extra_path_patterns'] = sorted(patterns)
        self.teacher_map = self._load_teacher_map(self.root / args.teacher_file)
        self.symptoms = read_json(self.root / args.symptoms)
        self.modules_kb = read_json(self.root / args.modules_kb)
        self.domain_keywords = self._build_domain_keywords()
        self.split_map = self._load_split_map(self.root / args.split_dir)
        self.allowed_bug_ids = self._load_bug_id_filter(self.root / args.bug_id_file) if args.bug_id_file else None
        self.repo_indices: Dict[Tuple[str, str], BareRepoModuleIndex] = {}

    def _load_teacher_map(self, path: Path) -> Dict[str, Dict[str, Any]]:
        if not path.exists():
            return {}
        rows = read_jsonl(path)
        return {row['bug_id']: row for row in rows if 'bug_id' in row}

    def _load_split_map(self, split_dir: Path) -> Dict[str, str]:
        split_map: Dict[str, str] = {}
        for split in ['train', 'valid', 'test']:
            p = split_dir / f'{split}.jsonl'
            if not p.exists():
                continue
            for row in read_jsonl(p):
                split_map[row['bug_id']] = split
        return split_map

    def _load_bug_id_filter(self, path: Path) -> set[str]:
        if not path.exists():
            raise FileNotFoundError(f'bug id filter not found: {path}')
        bug_ids = set()
        if path.suffix == '.jsonl':
            for row in read_jsonl(path):
                if 'bug_id' in row:
                    bug_ids.add(str(row['bug_id']))
        else:
            for line in path.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    bug_ids.add(line)
        return bug_ids

    def _build_domain_keywords(self) -> List[str]:
        kws = set()
        for mod_data in self.modules_kb.values():
            sym = mod_data.get('symptoms', {})
            for group in sym.values():
                if isinstance(group, list):
                    kws.update(x.lower() for x in group)
            for group in ['rootcause_trigger', 'confusion']:
                if isinstance(mod_data.get(group), list):
                    for item in mod_data[group]:
                        kws.update(x for x in normalize_tokens(item) if len(x) >= 3)
        for mod_map in self.symptoms.values():
            kws.update(k.lower() for k in mod_map.keys())
        return sorted(kws)

    def _project_index(self, project: str, ref: str = 'HEAD') -> BareRepoModuleIndex:
        key = (project, ref)
        if key not in self.repo_indices:
            repo_dir = self.repos_dir / project
            self.repo_indices[key] = BareRepoModuleIndex(repo_dir, self.config, self.cache_dir, ref=ref)
        return self.repo_indices[key]

    def _case_files(self) -> Iterable[Path]:
        pattern = '*/*.md' if not self.args.project else f'{self.args.project}/*.md'
        yield from sorted(self.cases_dir.glob(pattern))

    def _target_module(self, bug_text: str, bug_id: str) -> Optional[str]:
        if self.args.module_source == 'manual':
            return parse_primary_module(bug_text)
        teacher = self.teacher_map.get(bug_id, {})
        return teacher.get('teacher_primary_module')

    def _teacher_prob(self, bug_id: str, module: str) -> float:
        teacher = self.teacher_map.get(bug_id, {})
        probs_map = teacher.get('teacher_probs_map', {})
        try:
            return float(probs_map.get(module, 0.0))
        except Exception:
            return 0.0

    def _build_rows_for_case(self, case_path: Path) -> List[Dict[str, Any]]:
        text = case_path.read_text(encoding='utf-8')
        bug_text_source = initial_report_text if self.args.bug_text_source == 'initial' else section_before_ground_truth
        bug_text = bug_text_source(text)[:BUG_TEXT_MAX_CHARS]
        bug_id = case_path.stem
        if self.allowed_bug_ids is not None and bug_id not in self.allowed_bug_ids:
            return []
        split = self.split_map.get(bug_id, 'train')
        if self.args.split_filter and split not in set(self.args.split_filter):
            return []
        project = case_path.parent.name
        buggy_sha = parse_buggy_sha(text)
        code_ref = buggy_sha if self.args.code_ref_source == 'buggy' and buggy_sha else 'HEAD'
        index_ref = code_ref if self.args.index_ref_source == 'code' else 'HEAD'
        if self.args.code_ref_source == 'buggy' and not buggy_sha and not self.args.allow_head_fallback:
            return []
        target_module = self._target_module(text, bug_id)
        if not target_module:
            return []

        modified_files = parse_modified_files(text)
        if not modified_files:
            return []

        repo_index = self._project_index(project, index_ref)
        stack_files = extract_files_from_log(bug_text)
        bug_tokens = set(build_bug_tokens(bug_text))
        bug_domain_hits = set(keyword_hits(bug_text, self.domain_keywords))
        candidate_records = list(repo_index.by_module.get(target_module, []))
        positive_records = []
        missing_positive_files = []
        seen_files = {row['file'] for row in candidate_records}
        for mf in modified_files:
            row = repo_index.by_file.get(mf)
            if row is not None:
                if mf not in seen_files:
                    candidate_records.append(row)
                    seen_files.add(mf)
                positive_records.append(row)
            else:
                missing_positive_files.append(mf)

        rows = []
        positive_set = set(modified_files)
        for record in candidate_records:
            file_path = record['file']
            stack_exact = 1.0 if file_path in stack_files else 0.0
            stack_suffix = suffix_match_score(file_path, stack_files)
            path_overlap = len(bug_tokens & set(record['path_tokens']))
            name_overlap = len(bug_tokens & set(record['name_tokens']))
            dep_overlap = len(bug_tokens & set(record['dep_tokens']))
            module_match = 1 if record['module'] == target_module else 0
            symptom_hits = keyword_hits(bug_text, list(self.symptoms.get(record['module'], {}).keys()))
            domain_source = ' '.join(record['path_tokens'] + record['name_tokens'] + record['dep_tokens'] + record['api_hits'])
            domain_hits = bug_domain_hits & set(keyword_hits(domain_source, list(bug_domain_hits)))
            teacher_prob = self._teacher_prob(bug_id, record['module'])
            heuristic = (
                5.0 * stack_exact +
                4.0 * stack_suffix +
                1.5 * path_overlap +
                2.0 * name_overlap +
                1.2 * dep_overlap +
                0.8 * len(symptom_hits) +
                0.8 * len(domain_hits) +
                2.0 * teacher_prob +
                0.5 * module_match +
                0.5 * float(record['margin'])
            )
            raw_file_text = git_show_text(repo_index.repo_git_dir, file_path, max_bytes=DEFAULT_MAX_BYTES, ref=code_ref)
            file_text = build_file_text(file_path, raw_file_text)
            rows.append({
                'bug_id': bug_id,
                'project': project,
                'split': split,
                'selected_module': target_module,
                'file': file_path,
                'file_module': record['module'],
                'label': 1 if file_path in positive_set else 0,
                'bug_text': bug_text,
                'file_path_text': file_path.replace('/', ' '),
                'file_name_text': Path(file_path).name,
                'file_text': file_text,
                'features': {
                    'stack_file_exact_match': stack_exact,
                    'stack_file_suffix_match': round(stack_suffix, 6),
                    'bug_path_token_overlap': path_overlap,
                    'bug_name_token_overlap': name_overlap,
                    'bug_dep_token_overlap': dep_overlap,
                    'module_match_flag': module_match,
                    'module_margin': round(float(record['margin']), 6),
                    'symptom_overlap_count': len(symptom_hits),
                    'domain_keyword_overlap': len(domain_hits),
                    'teacher_module_prob': round(teacher_prob, 6),
                },
                'debug': {
                    'matched_symptoms': symptom_hits,
                    'matched_domain_keywords': sorted(domain_hits),
                    'path_tokens': record['path_tokens'],
                    'name_tokens': record['name_tokens'],
                    'dep_tokens': record['dep_tokens'],
                },
                'heuristic_score': round(heuristic, 6),
                'source_file': str(case_path.relative_to(self.root)).replace('\\', '/'),
                'positive_files': modified_files,
                'missing_positive_files': missing_positive_files,
                'code_ref': code_ref,
                'index_ref': index_ref,
                'buggy_sha': buggy_sha,
            })

        positives = [r for r in rows if r['label'] == 1]
        negatives = [r for r in rows if r['label'] == 0]
        negatives.sort(key=lambda x: (-x['heuristic_score'], x['file']))
        keep_neg = max(self.args.max_candidates_per_bug - len(positives), 0)
        if keep_neg > 0:
            negatives = negatives[:keep_neg]
        else:
            negatives = []
        selected_rows = sorted(positives + negatives, key=lambda x: (-x['label'], -x['heuristic_score'], x['file']))

        bug_meta = {
            'bug_text': bug_text,
            'stack_files': stack_files,
            'modified_files': modified_files,
            'teacher_primary_module': self.teacher_map.get(bug_id, {}).get('teacher_primary_module'),
            'deep_text_ready': True,
            'bug_text_source': self.args.bug_text_source,
            'code_ref': code_ref,
            'index_ref': index_ref,
            'buggy_sha': buggy_sha,
        }
        for row in selected_rows:
            row['bug_meta'] = bug_meta
        return selected_rows

    def build(self) -> Dict[str, Any]:
        all_rows: List[Dict[str, Any]] = []
        bug_counts: Dict[str, int] = {}
        label_counter = Counter()
        module_counter = Counter()

        for case_path in self._case_files():
            rows = self._build_rows_for_case(case_path)
            if not rows:
                continue
            all_rows.extend(rows)
            bug_counts[rows[0]['bug_id']] = len(rows)
            for row in rows:
                label_counter[row['label']] += 1
                module_counter[row['selected_module']] += 1

        split_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in all_rows:
            split_rows[row['split']].append(row)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(self.output_dir / 'all.jsonl', all_rows)
        for split in ['train', 'valid', 'test']:
            write_jsonl(self.output_dir / f'{split}.jsonl', split_rows.get(split, []))

        summary = {
            'bugs': len(bug_counts),
            'pairs': len(all_rows),
            'positive_pairs': label_counter[1],
            'negative_pairs': label_counter[0],
            'avg_candidates_per_bug': round(sum(bug_counts.values()) / max(len(bug_counts), 1), 3),
            'selected_module_distribution': dict(module_counter),
            'split_sizes': {split: len(split_rows.get(split, [])) for split in ['train', 'valid', 'test']},
            'cache_dir': str(self.cache_dir),
            'bug_text_source': self.args.bug_text_source,
            'code_ref_source': self.args.code_ref_source,
            'index_ref_source': self.args.index_ref_source,
            'extra_index_extensions': self.extra_index_extensions,
            'extra_index_path_patterns': self.extra_index_path_patterns,
            'allow_head_fallback': self.args.allow_head_fallback,
            'bug_id_file': self.args.bug_id_file,
        }
        (self.output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        return summary


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--cases-dir', default='labels/manual/cases')
    ap.add_argument('--repos-dir', default='repos')
    ap.add_argument('--config', default='module_map/configs/config_base.json')
    ap.add_argument('--symptoms', default='module_map/kb/module_symptoms.json')
    ap.add_argument('--modules-kb', default='module_map/configs/modules_kb.json')
    ap.add_argument('--teacher-file', default='distill/data/manual_seed/teacher_labels.jsonl')
    ap.add_argument('--split-dir', default='distill/data/manual_seed')
    ap.add_argument('--cache-dir', default='ranker/cache/module_index')
    ap.add_argument('--output-dir', default='ranker/data/manual_seed_oracle')
    ap.add_argument('--module-source', choices=['manual', 'teacher'], default='manual')
    ap.add_argument('--bug-text-source', choices=['initial', 'full'], default='initial')
    ap.add_argument('--code-ref-source', choices=['buggy', 'head'], default='buggy')
    ap.add_argument('--index-ref-source', choices=['head', 'code'], default='head')
    ap.add_argument(
        '--extra-index-extensions',
        default='',
        help=(
            'Comma-separated extensions to add to the repository index. '
            f'For CAE rescue builds, use: {",".join(sorted(CAE_SOURCE_EXTENSIONS))}'
        ),
    )
    ap.add_argument(
        '--extra-index-path-patterns',
        default='',
        help='Comma-separated fnmatch path patterns for supported extensionless files.',
    )
    ap.add_argument('--allow-head-fallback', action='store_true')
    ap.add_argument('--project', default=None)
    ap.add_argument('--split-filter', nargs='+', choices=['train', 'valid', 'test'], default=None)
    ap.add_argument('--bug-id-file', default=None)
    ap.add_argument('--max-candidates-per-bug', type=int, default=200)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    builder = RankingDatasetBuilder(args)
    summary = builder.build()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
