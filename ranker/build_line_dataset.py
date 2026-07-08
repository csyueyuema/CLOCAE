#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'module_map'))
sys.path.append(str(ROOT / 'scripts'))

from build_ranking_dataset import (  # type: ignore
    BUG_TEXT_MAX_CHARS,
    DEEP_TEXT_MAX_CHARS,
    BareRepoModuleIndex,
    build_bug_tokens,
    compact_text,
    extract_files_from_log,
    git_show_text,
    initial_report_text,
    keyword_hits,
    parse_buggy_sha,
    read_json,
    section_before_ground_truth,
)
from eval_ranker import evaluate_model  # type: ignore
from train_ranker import FEATURE_KEYS as FILE_FEATURE_KEYS  # type: ignore
from train_ranker import RankingDataset as FileRankingDataset  # type: ignore
from train_ranker import WideDeepRanker, collate_fn  # type: ignore


RE_MODIFIED_SECTION = re.compile(r"^### Modified Files\s*$", re.MULTILINE)
RE_MODIFIED_LINE_WITH_LINES = re.compile(r"^-\s+(.+?):\s*(.+?)\s*$")
RE_RANGE = re.compile(r"^(\d+)\s*-\s*(\d+)$")
RE_TOKEN = re.compile(r"[^a-z0-9_]+")

LINE_FEATURE_KEYS = [
    'file_rank_score',
    'file_rank_position',
    'stack_file_match',
    'line_no_norm',
    'line_length',
    'line_token_overlap',
    'context_token_overlap',
    'domain_keyword_overlap',
    'is_comment_line',
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def tokenize_text(text: str) -> List[str]:
    return [tok for tok in RE_TOKEN.split(text.lower()) if len(tok) >= 2]


def parse_modified_line_map(text: str) -> Dict[str, Set[int]]:
    line_map: Dict[str, Set[int]] = defaultdict(set)
    match = RE_MODIFIED_SECTION.search(text)
    if not match:
        return line_map
    tail = text[match.end():].split('## Manual Module Label', 1)[0]
    for raw_line in tail.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('## '):
            continue
        mm = RE_MODIFIED_LINE_WITH_LINES.match(line)
        if not mm:
            continue
        file_path = mm.group(1).strip().replace('\\', '/')
        line_spec = mm.group(2).strip()
        nums: Set[int] = set()
        for part in [x.strip() for x in line_spec.split(',') if x.strip()]:
            rm = RE_RANGE.match(part)
            if rm:
                start = int(rm.group(1))
                end = int(rm.group(2))
                if start <= end:
                    nums.update(range(start, end + 1))
            else:
                try:
                    nums.add(int(part))
                except ValueError:
                    continue
        if nums:
            line_map[file_path].update(nums)
    return line_map


def build_statement_text(lines: List[str], idx0: int, window: int = 1) -> str:
    start = max(0, idx0 - window)
    end = min(len(lines), idx0 + window + 1)
    parts = []
    for i in range(start, end):
        prefix = '>>' if i == idx0 else '..'
        parts.append(f'{prefix} L{i+1}: {lines[i]}')
    return compact_text('\n'.join(parts), max_chars=DEEP_TEXT_MAX_CHARS)


def is_comment_line(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return 1 if stripped.startswith(('#', '//', '/*', '*', '!')) else 0


class LineDatasetBuilder:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = Path(args.root).resolve()
        self.cases_dir = self.root / args.cases_dir
        self.repos_dir = self.root / args.repos_dir
        self.file_data_dir = self.root / args.file_data_dir
        self.file_model_dir = self.root / args.file_model_dir
        self.output_dir = self.root / args.output_dir
        self.cache_dir = self.root / args.cache_dir
        self.config = read_json(self.root / args.config)
        self.modules_kb = read_json(self.root / args.modules_kb)
        self.repo_indices: Dict[Tuple[str, str], BareRepoModuleIndex] = {}
        self.domain_keywords = self._build_domain_keywords()
        self.file_predictions = self._predict_file_scores()

    def _build_domain_keywords(self) -> List[str]:
        kws = set()
        for mod_data in self.modules_kb.values():
            sym = mod_data.get('symptoms', {})
            for group in sym.values():
                if isinstance(group, list):
                    kws.update(x.lower() for x in group)
        return sorted(kws)

    def _project_index(self, project: str, ref: str = 'HEAD') -> BareRepoModuleIndex:
        key = (project, ref)
        if key not in self.repo_indices:
            self.repo_indices[key] = BareRepoModuleIndex(self.repos_dir / project, self.config, self.cache_dir, ref=ref)
        return self.repo_indices[key]

    def _predict_file_scores(self) -> Dict[str, List[Dict[str, Any]]]:
        ckpt = torch.load(self.file_model_dir / 'ranker.pt', map_location='cpu')
        model_args = ckpt['args']
        tokenizer = AutoTokenizer.from_pretrained(self.file_model_dir / 'encoder')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = WideDeepRanker(
            model_name=str(self.file_model_dir / 'encoder'),
            wide_dim=len(ckpt.get('feature_keys', FILE_FEATURE_KEYS)),
            deep_hidden=model_args.get('deep_hidden', 256),
            wide_hidden=model_args.get('wide_hidden', 64),
            dropout=model_args.get('dropout', 0.1),
            use_wide=not model_args.get('disable_wide', False),
            use_deep=not model_args.get('disable_deep', False),
        ).to(device)
        model.load_state_dict(ckpt['state_dict'])

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for split in ['train', 'valid', 'test']:
            p = self.file_data_dir / f'{split}.jsonl'
            if not p.exists():
                continue
            ds = FileRankingDataset(p, feature_keys=ckpt.get('feature_keys', FILE_FEATURE_KEYS))
            loader = DataLoader(
                ds,
                batch_size=model_args.get('batch_size', 4),
                shuffle=False,
                collate_fn=lambda batch: collate_fn(batch, tokenizer, model_args.get('max_length', 384), use_deep=not model_args.get('disable_deep', False)),
            )
            _, rows = evaluate_model(model, loader, device)
            for row in rows:
                row['split'] = split
                grouped[row['bug_id']].append(row)
        for bug_id, items in grouped.items():
            items.sort(key=lambda x: (-x['score'], x['file']))
        return grouped

    def _candidate_files(self, bug_id: str, positive_files: Set[str]) -> List[Tuple[str, float, int]]:
        items = self.file_predictions.get(bug_id, [])
        ranked = [(x['file'], float(x['score']), idx + 1) for idx, x in enumerate(items[: self.args.topk_files])]
        seen = {x[0] for x in ranked}
        extra = []
        for idx, row in enumerate(items, start=1):
            if row['file'] in positive_files and row['file'] not in seen:
                extra.append((row['file'], float(row['score']), idx))
                seen.add(row['file'])
        return ranked + extra

    def _score_line(
        self,
        bug_tokens: Set[str],
        bug_domain_hits: Set[str],
        line_text: str,
        context_text: str,
        file_score: float,
        file_rank: int,
        line_no: int,
        total_lines: int,
        stack_file_match: int,
    ) -> Tuple[float, Dict[str, float]]:
        line_tokens = set(tokenize_text(line_text))
        context_tokens = set(tokenize_text(context_text))
        domain_hits = bug_domain_hits & set(keyword_hits((line_text + ' ' + context_text).lower(), list(bug_domain_hits)))
        line_overlap = len(bug_tokens & line_tokens)
        context_overlap = len(bug_tokens & context_tokens)
        feats = {
            'file_rank_score': round(file_score, 6),
            'file_rank_position': round(1.0 / max(file_rank, 1), 6),
            'stack_file_match': float(stack_file_match),
            'line_no_norm': round(line_no / max(total_lines, 1), 6),
            'line_length': float(min(len(line_text.strip()), 200)),
            'line_token_overlap': float(line_overlap),
            'context_token_overlap': float(context_overlap),
            'domain_keyword_overlap': float(len(domain_hits)),
            'is_comment_line': float(is_comment_line(line_text)),
        }
        heuristic = (
            3.0 * feats['file_rank_score'] +
            2.2 * feats['line_token_overlap'] +
            1.6 * feats['context_token_overlap'] +
            1.2 * feats['domain_keyword_overlap'] +
            1.0 * feats['stack_file_match'] +
            0.4 * feats['file_rank_position'] -
            0.2 * feats['is_comment_line']
        )
        return heuristic, feats

    def _build_rows_for_case(self, case_path: Path) -> List[Dict[str, Any]]:
        text = case_path.read_text(encoding='utf-8', errors='ignore')
        bug_id = case_path.stem
        project = case_path.parent.name
        bug_text_source = initial_report_text if self.args.bug_text_source == 'initial' else section_before_ground_truth
        bug_text = bug_text_source(text)[:BUG_TEXT_MAX_CHARS]
        buggy_sha = parse_buggy_sha(text)
        code_ref = buggy_sha if self.args.code_ref_source == 'buggy' and buggy_sha else 'HEAD'
        index_ref = code_ref if self.args.index_ref_source == 'code' else 'HEAD'
        if self.args.code_ref_source == 'buggy' and not buggy_sha and not self.args.allow_head_fallback:
            return []
        line_map = parse_modified_line_map(text)
        positive_files = set(line_map.keys())
        if not positive_files:
            return []

        candidate_files = self._candidate_files(bug_id, positive_files)
        if not candidate_files:
            return []

        repo_index = self._project_index(project, index_ref)
        bug_tokens = set(build_bug_tokens(bug_text))
        bug_domain_hits = set(keyword_hits(bug_text.lower(), self.domain_keywords))
        stack_files = set(extract_files_from_log(bug_text))
        split = candidate_files and self.file_predictions.get(bug_id, [{}])[0].get('split', 'train')

        rows = []
        for file_path, file_score, file_rank in candidate_files:
            raw = git_show_text(repo_index.repo_git_dir, file_path, max_bytes=self.args.max_file_bytes, ref=code_ref)
            if not raw:
                continue
            lines = raw.splitlines()
            if not lines:
                continue
            positive_lines = {ln for ln in line_map.get(file_path, set()) if 1 <= ln <= len(lines)}

            candidate_line_rows = []
            for idx0, line_text in enumerate(lines):
                if not line_text.strip():
                    continue
                line_no = idx0 + 1
                context_text = build_statement_text(lines, idx0, window=self.args.context_window)
                heuristic, feats = self._score_line(
                    bug_tokens=bug_tokens,
                    bug_domain_hits=bug_domain_hits,
                    line_text=line_text,
                    context_text=context_text,
                    file_score=file_score,
                    file_rank=file_rank,
                    line_no=line_no,
                    total_lines=len(lines),
                    stack_file_match=1 if file_path in stack_files else 0,
                )
                candidate_line_rows.append({
                    'bug_id': bug_id,
                    'project': project,
                    'split': split,
                    'selected_file': file_path,
                    'file': f'{file_path}:{line_no}',
                    'file_path': file_path,
                    'line_no': line_no,
                    'label': 1 if line_no in positive_lines else 0,
                    'bug_text': bug_text,
                    'file_path_text': file_path.replace('/', ' '),
                    'file_name_text': f'{Path(file_path).name} line {line_no}',
                    'file_text': context_text,
                    'features': feats,
                    'heuristic_score': round(heuristic, 6),
                    'source_file': str(case_path.relative_to(self.root)).replace('\\', '/'),
                    'positive_file_lines': sorted(positive_lines),
                    'bug_meta': {
                        'stack_files': sorted(stack_files),
                        'positive_files': sorted(positive_files),
                        'granularity': 'line_with_context',
                        'bug_text_source': self.args.bug_text_source,
                        'code_ref': code_ref,
                        'index_ref': index_ref,
                        'buggy_sha': buggy_sha,
                    },
                })

            positives = [r for r in candidate_line_rows if r['label'] == 1]
            negatives = [r for r in candidate_line_rows if r['label'] == 0]
            negatives.sort(key=lambda x: (-x['heuristic_score'], x['line_no']))
            keep_neg = max(self.args.max_lines_per_file - len(positives), 0)
            negatives = negatives[:keep_neg]
            rows.extend(sorted(positives + negatives, key=lambda x: (-x['label'], -x['heuristic_score'], x['line_no'])))

        return rows

    def build(self) -> Dict[str, Any]:
        all_rows: List[Dict[str, Any]] = []
        split_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        bug_counter = Counter()
        label_counter = Counter()
        file_counter = Counter()

        for case_path in sorted(self.cases_dir.glob('*/*.md')):
            rows = self._build_rows_for_case(case_path)
            if not rows:
                continue
            all_rows.extend(rows)
            bug_counter[rows[0]['bug_id']] = len(rows)
            for row in rows:
                split_rows[row['split']].append(row)
                label_counter[row['label']] += 1
                file_counter[row['selected_file']] += 1

        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(self.output_dir / 'all.jsonl', all_rows)
        for split in ['train', 'valid', 'test']:
            write_jsonl(self.output_dir / f'{split}.jsonl', split_rows.get(split, []))

        summary = {
            'bugs': len(bug_counter),
            'pairs': len(all_rows),
            'positive_pairs': label_counter[1],
            'negative_pairs': label_counter[0],
            'avg_candidates_per_bug': round(sum(bug_counter.values()) / max(len(bug_counter), 1), 3),
            'split_sizes': {split: len(split_rows.get(split, [])) for split in ['train', 'valid', 'test']},
            'topk_files': self.args.topk_files,
            'max_lines_per_file': self.args.max_lines_per_file,
            'feature_keys': LINE_FEATURE_KEYS,
            'bug_text_source': self.args.bug_text_source,
            'code_ref_source': self.args.code_ref_source,
            'index_ref_source': self.args.index_ref_source,
            'allow_head_fallback': self.args.allow_head_fallback,
        }
        (self.output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        return summary


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--cases-dir', default='labels/manual/cases')
    ap.add_argument('--repos-dir', default='repos')
    ap.add_argument('--config', default='module_map/configs/config_base.json')
    ap.add_argument('--modules-kb', default='module_map/configs/modules_kb.json')
    ap.add_argument('--cache-dir', default='ranker/cache/module_index')
    ap.add_argument('--file-data-dir', default='ranker/data/manual_oracle_stage2_deep')
    ap.add_argument('--file-model-dir', default='ranker/outputs/wide_deep_seed')
    ap.add_argument('--output-dir', default='ranker/data/manual_line_stage3')
    ap.add_argument('--bug-text-source', choices=['initial', 'full'], default='initial')
    ap.add_argument('--code-ref-source', choices=['buggy', 'head'], default='buggy')
    ap.add_argument('--index-ref-source', choices=['head', 'code'], default='head')
    ap.add_argument('--allow-head-fallback', action='store_true')
    ap.add_argument('--topk-files', type=int, default=5)
    ap.add_argument('--max-lines-per-file', type=int, default=120)
    ap.add_argument('--context-window', type=int, default=1)
    ap.add_argument('--max-file-bytes', type=int, default=250000)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    builder = LineDatasetBuilder(args)
    summary = builder.build()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
