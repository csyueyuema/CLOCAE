#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

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


def resolve_feature_keys(selected_groups: Optional[List[str]]):
    if not selected_groups:
        return FEATURE_KEYS
    keys = []
    for group in selected_groups:
        if group == 'all':
            return FEATURE_KEYS
        if group not in FEATURE_GROUPS:
            raise ValueError(f'Unknown feature group: {group}')
        keys.extend(FEATURE_GROUPS[group])
    return [key for key in FEATURE_KEYS if key in set(keys)]


class FeatureNormalizer:
    def __init__(self, feature_keys: List[str]):
        self.feature_keys = feature_keys
        self.mean = np.zeros(len(feature_keys), dtype=np.float32)
        self.std = np.ones(len(feature_keys), dtype=np.float32)
        self._fitted = False

    def fit(self, rows: List[dict]):
        values = []
        for row in rows:
            values.append([float(row['features'][k]) for k in self.feature_keys])
        arr = np.array(values, dtype=np.float32)
        self.mean = arr.mean(axis=0)
        self.std = arr.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        self._fitted = True

    def transform(self, features: List[float]) -> List[float]:
        if not self._fitted:
            return features
        arr = np.array(features, dtype=np.float32)
        normed = (arr - self.mean) / self.std
        return normed.tolist()

    def state_dict(self) -> dict:
        return {
            'feature_keys': self.feature_keys,
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
        }

    def load_state_dict(self, d: dict):
        self.feature_keys = d['feature_keys']
        self.mean = np.array(d['mean'], dtype=np.float32)
        self.std = np.array(d['std'], dtype=np.float32)
        self._fitted = True


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        p = torch.sigmoid(logits)
        pt = p * labels + (1 - p) * (1 - labels)
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
        weight = alpha_t * focal_weight
        if self.pos_weight != 1.0:
            weight = weight * (self.pos_weight * labels + (1 - labels))
        return (weight * bce).mean()


class RankingDataset(Dataset):
    def __init__(self, path: Path, feature_keys: Optional[List[str]] = None, normalizer: Optional[FeatureNormalizer] = None):
        self.rows = []
        self.feature_keys = feature_keys or FEATURE_KEYS
        self.normalizer = normalizer
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
        raw_features = [float(row['features'][k]) for k in self.feature_keys]
        if self.normalizer is not None:
            features = self.normalizer.transform(raw_features)
        else:
            features = raw_features
        return {
            'bug_id': row['bug_id'],
            'file': row['file'],
            'text': combined_text,
            'features': features,
            'label': float(row['label']),
        }


class WideDeepRankerV2(nn.Module):
    def __init__(
        self,
        model_name: str,
        wide_dim: int,
        deep_hidden: int = 256,
        wide_hidden: int = 64,
        dropout: float = 0.2,
        use_wide: bool = True,
        use_deep: bool = True,
        pooling: str = 'mean',
    ):
        super().__init__()
        if not use_wide and not use_deep:
            raise ValueError('At least one branch must be enabled.')
        self.use_wide = use_wide
        self.use_deep = use_deep
        self.pooling = pooling
        hidden_parts = []

        if use_deep:
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size
            self.deep_net = nn.Sequential(
                nn.Linear(hidden_size, deep_hidden),
                nn.LayerNorm(deep_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            hidden_parts.append(deep_hidden)
        else:
            self.encoder = None
            self.deep_net = None

        if use_wide:
            self.wide_net = nn.Sequential(
                nn.Linear(wide_dim, wide_hidden),
                nn.LayerNorm(wide_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            hidden_parts.append(wide_hidden)
        else:
            self.wide_net = None

        total_hidden = sum(hidden_parts)
        self.classifier = nn.Sequential(
            nn.Linear(total_hidden, deep_hidden),
            nn.LayerNorm(deep_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(deep_hidden, 1),
        )

    def _pool(self, outputs, attention_mask):
        if self.pooling == 'mean':
            hidden = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            sum_hidden = (hidden * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_hidden / count
        else:
            return outputs.last_hidden_state[:, 0, :]

    def forward(self, input_ids, attention_mask, wide_features):
        parts = []
        if self.use_deep:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = self._pool(outputs, attention_mask)
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


def evaluate(model, dataloader, device, loss_fn):
    model.eval()
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
    parser.add_argument('--train-file', default='ranker/data/manual_stage2_deep/train.jsonl')
    parser.add_argument('--valid-file', default='ranker/data/manual_stage2_deep/valid.jsonl')
    parser.add_argument('--model-name', default='microsoft/codebert-base')
    parser.add_argument('--output-dir', default='ranker/outputs/wide_deep')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--grad-accum-steps', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max-length', type=int, default=384)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--deep-hidden', type=int, default=256)
    parser.add_argument('--wide-hidden', type=int, default=64)
    parser.add_argument('--pooling', choices=['cls', 'mean'], default='mean')
    parser.add_argument('--focal-alpha', type=float, default=0.25)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--feature-groups', nargs='+', default=['all'], help='Choose from: all, stack, structure, domain')
    parser.add_argument('--disable-wide', action='store_true')
    parser.add_argument('--disable-deep', action='store_true')
    parser.add_argument('--no-normalize', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_path = Path(args.train_file).resolve()
    valid_path = Path(args.valid_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_feature_keys = resolve_feature_keys(args.feature_groups)
    use_wide = not args.disable_wide
    use_deep = not args.disable_deep

    normalizer = None if args.no_normalize else FeatureNormalizer(selected_feature_keys)

    train_ds_raw = RankingDataset(train_path, feature_keys=selected_feature_keys)
    if normalizer is not None:
        normalizer.fit(train_ds_raw.rows)
    train_ds = RankingDataset(train_path, feature_keys=selected_feature_keys, normalizer=normalizer)
    valid_ds = RankingDataset(valid_path, feature_keys=selected_feature_keys, normalizer=normalizer)

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
    model = WideDeepRankerV2(
        model_name=args.model_name,
        wide_dim=len(selected_feature_keys),
        deep_hidden=args.deep_hidden,
        wide_hidden=args.wide_hidden,
        dropout=args.dropout,
        use_wide=use_wide,
        use_deep=use_deep,
        pooling=args.pooling,
    ).to(device)

    pos_count = sum(1 for row in train_ds.rows if float(row['label']) > 0.5)
    neg_count = max(len(train_ds.rows) - pos_count, 1)
    pos_weight_val = neg_count / max(pos_count, 1)

    loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, pos_weight=pos_weight_val)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(total_steps // 5, 1),
        num_training_steps=max(total_steps, 1),
    )

    best_metric = -1.0
    best_epoch = 0
    history = []
    patience = 3
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            labels = batch['labels'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            wide_features = batch['wide_features'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask, wide_features=wide_features)
            loss = loss_fn(logits, labels)
            loss = loss / args.grad_accum_steps
            loss.backward()
            total_loss += loss.item() * args.grad_accum_steps

            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        metrics = evaluate(model, valid_loader, device, loss_fn)
        metrics['epoch'] = epoch
        metrics['train_loss'] = total_loss / max(len(train_loader), 1)
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))

        if metrics['mrr'] > best_metric:
            best_metric = metrics['mrr']
            best_epoch = epoch
            patience_counter = 0
            if model.encoder is not None:
                model.encoder.save_pretrained(output_dir / 'encoder')
                tokenizer.save_pretrained(output_dir / 'encoder')
            torch.save({
                'state_dict': model.state_dict(),
                'feature_keys': selected_feature_keys,
                'args': vars(args),
                'normalizer': normalizer.state_dict() if normalizer is not None else None,
            }, output_dir / 'ranker.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'[EARLY STOP] No improvement for {patience} epochs. Best at epoch {best_epoch}.')
                break

    (output_dir / 'train_history.json').write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[OK] Saved best ranker to {output_dir} (best epoch: {best_epoch}, MRR: {best_metric:.4f})')


if __name__ == '__main__':
    main()
