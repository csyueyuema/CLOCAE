#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

FEATURE_KEYS = [
    'stack_file_exact_match',
    'stack_file_suffix_match',
    'bug_path_token_overlap',
    'bug_name_token_overlap',
    'bug_dep_token_overlap',
    'module_match_flag',
    'module_margin',
    'symptom_overlap_count',
    'domain_keyword_overlap',
    'teacher_module_prob',
]

FEATURE_GROUPS = {
    'stack': ['stack_file_exact_match', 'stack_file_suffix_match'],
    'structure': [
        'bug_path_token_overlap',
        'bug_name_token_overlap',
        'bug_dep_token_overlap',
        'module_match_flag',
        'module_margin',
    ],
    'domain': ['symptom_overlap_count', 'domain_keyword_overlap', 'teacher_module_prob'],
}


def resolve_feature_keys(selected_groups: List[str] | None):
    if not selected_groups:
        return FEATURE_KEYS
    keys = []
    for group in selected_groups:
        if group == 'all':
            return FEATURE_KEYS
        if group not in FEATURE_GROUPS:
            raise ValueError(f'Unknown feature group: {group}')
        keys.extend(FEATURE_GROUPS[group])
    # preserve canonical order
    return [key for key in FEATURE_KEYS if key in set(keys)]


class RankingDataset(Dataset):
    def __init__(self, path: Path, feature_keys: List[str] | None = None):
        self.rows = []
        self.feature_keys = feature_keys or FEATURE_KEYS
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        combined_text = (
            f"[BUG] {row['bug_text']}\n"
            f"[FILE_PATH] {row['file_path_text']}\n"
            f"[FILE_NAME] {row['file_name_text']}\n"
            f"[FILE_TEXT] {row['file_text']}"
        )
        return {
            'bug_id': row['bug_id'],
            'file': row['file'],
            'text': combined_text,
            'features': [float(row['features'][k]) for k in self.feature_keys],
            'label': float(row['label']),
        }


class WideDeepRanker(nn.Module):
    def __init__(
        self,
        model_name: str,
        wide_dim: int,
        deep_hidden: int = 256,
        wide_hidden: int = 64,
        dropout: float = 0.1,
        use_wide: bool = True,
        use_deep: bool = True,
    ):
        super().__init__()
        if not use_wide and not use_deep:
            raise ValueError('At least one branch must be enabled.')
        self.use_wide = use_wide
        self.use_deep = use_deep
        hidden_parts = []

        if use_deep:
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size
            self.deep_net = nn.Sequential(
                nn.Linear(hidden_size, deep_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            hidden_parts.append(deep_hidden)
        else:
            self.encoder = None
            self.deep_net = None

        if use_wide:
            self.wide_net = nn.Sequential(
                nn.Linear(wide_dim, wide_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            hidden_parts.append(wide_hidden)
        else:
            self.wide_net = None

        self.classifier = nn.Sequential(
            nn.Linear(sum(hidden_parts), deep_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(deep_hidden, 1),
        )

    def forward(self, input_ids, attention_mask, wide_features):
        parts = []
        if self.use_deep:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, 'last_hidden_state'):
                pooled = outputs.last_hidden_state[:, 0, :]
            else:
                pooled = outputs[0][:, 0, :]
            parts.append(self.deep_net(pooled))
        if self.use_wide:
            parts.append(self.wide_net(wide_features))
        logits = self.classifier(torch.cat(parts, dim=-1)).squeeze(-1)
        return logits


def collate_fn(batch, tokenizer, max_length: int, use_deep: bool = True):
    if use_deep:
        texts = [item['text'] for item in batch]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    else:
        batch_size = len(batch)
        enc = {
            'input_ids': torch.zeros((batch_size, 1), dtype=torch.long),
            'attention_mask': torch.zeros((batch_size, 1), dtype=torch.long),
        }
    enc['wide_features'] = torch.tensor([item['features'] for item in batch], dtype=torch.float)
    enc['labels'] = torch.tensor([item['label'] for item in batch], dtype=torch.float)
    enc['bug_ids'] = [item['bug_id'] for item in batch]
    enc['files'] = [item['file'] for item in batch]
    return enc


def compute_ranking_metrics(rows: List[Dict[str, float]]):
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


def evaluate(model, dataloader, device, pos_weight):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    total_loss = 0.0
    all_rows = []

    with torch.no_grad():
        for batch in dataloader:
            bug_ids = batch.pop('bug_ids')
            files = batch.pop('files')
            labels = batch['labels'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            wide_features = batch['wide_features'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, wide_features=wide_features)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            scores = torch.sigmoid(logits).detach().cpu().tolist()
            labels_cpu = labels.detach().cpu().tolist()
            for bug_id, file_path, score, label in zip(bug_ids, files, scores, labels_cpu):
                all_rows.append({'bug_id': bug_id, 'file': file_path, 'score': float(score), 'label': float(label)})

    metrics = compute_ranking_metrics(all_rows)
    metrics['loss'] = total_loss / max(len(dataloader), 1)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', default='ranker/data/manual_oracle_stage2_deep/train.jsonl')
    parser.add_argument('--valid-file', default='ranker/data/manual_oracle_stage2_deep/valid.jsonl')
    parser.add_argument('--model-name', default='distilroberta-base')
    parser.add_argument('--output-dir', default='ranker/outputs/wide_deep_seed')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max-length', type=int, default=384)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--deep-hidden', type=int, default=256)
    parser.add_argument('--wide-hidden', type=int, default=64)
    parser.add_argument('--feature-groups', nargs='+', default=['all'], help='Choose from: all, stack, structure, domain')
    parser.add_argument('--disable-wide', action='store_true')
    parser.add_argument('--disable-deep', action='store_true')
    args = parser.parse_args()

    train_path = Path(args.train_file).resolve()
    valid_path = Path(args.valid_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_feature_keys = resolve_feature_keys(args.feature_groups)
    use_wide = not args.disable_wide
    use_deep = not args.disable_deep

    train_ds = RankingDataset(train_path, feature_keys=selected_feature_keys)
    valid_ds = RankingDataset(valid_path, feature_keys=selected_feature_keys)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name) if use_deep else None
    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, args.max_length, use_deep=use_deep),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, args.max_length, use_deep=use_deep),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WideDeepRanker(
        model_name=args.model_name,
        wide_dim=len(selected_feature_keys),
        deep_hidden=args.deep_hidden,
        wide_hidden=args.wide_hidden,
        dropout=args.dropout,
        use_wide=use_wide,
        use_deep=use_deep,
    ).to(device)

    pos_count = sum(1 for row in train_ds.rows if float(row['label']) > 0.5)
    neg_count = max(len(train_ds.rows) - pos_count, 1)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(total_steps // 10, 1),
        num_training_steps=max(total_steps, 1),
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_metric = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            labels = batch['labels'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            wide_features = batch['wide_features'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask, wide_features=wide_features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        metrics = evaluate(model, valid_loader, device, pos_weight)
        metrics['epoch'] = epoch
        metrics['train_loss'] = total_loss / max(len(train_loader), 1)
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))

        if metrics['mrr'] > best_metric:
            best_metric = metrics['mrr']
            if model.encoder is not None:
                model.encoder.save_pretrained(output_dir / 'encoder')
                tokenizer.save_pretrained(output_dir / 'encoder')
            torch.save({
                'state_dict': model.state_dict(),
                'feature_keys': selected_feature_keys,
                'args': vars(args),
            }, output_dir / 'ranker.pt')

    (output_dir / 'train_history.json').write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[OK] Saved best ranker to {output_dir}')


if __name__ == '__main__':
    main()
