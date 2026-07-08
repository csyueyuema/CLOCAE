#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

from train_ranker_current import FeatureNormalizer, FocalLoss, compute_ranking_metrics

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


class LineRankingDataset(Dataset):
    def __init__(self, path: Path, feature_keys: Optional[List[str]] = None, normalizer: Optional[FeatureNormalizer] = None):
        self.rows = []
        self.feature_keys = feature_keys or LINE_FEATURE_KEYS
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
            f"[LINE] {row['file_name_text']}\n"
            f"[CONTEXT] {row['file_text']}"
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


class GatedFusion(nn.Module):
    def __init__(self, wide_dim: int, deep_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(wide_dim + deep_dim, wide_dim + deep_dim),
            nn.Sigmoid(),
        )
        self.wide_dim = wide_dim
        self.deep_dim = deep_dim

    def forward(self, wide_out: torch.Tensor, deep_out: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([wide_out, deep_out], dim=-1)
        gate_values = self.gate(concat)
        wide_gate = gate_values[:, :self.wide_dim]
        deep_gate = gate_values[:, self.wide_dim:]
        return torch.cat([wide_out * wide_gate, deep_out * deep_gate], dim=-1)


class WideDeepLineRankerV2(nn.Module):
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

        if use_deep:
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size
            self.deep_net = nn.Sequential(
                nn.Linear(hidden_size, deep_hidden),
                nn.LayerNorm(deep_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
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
        else:
            self.wide_net = None

        if use_wide and use_deep:
            self.fusion = GatedFusion(wide_hidden, deep_hidden)
            total_hidden = wide_hidden + deep_hidden
        elif use_deep:
            total_hidden = deep_hidden
        else:
            total_hidden = wide_hidden

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
        deep_out = None
        wide_out = None
        if self.use_deep:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = self._pool(outputs, attention_mask)
            deep_out = self.deep_net(pooled)
        if self.use_wide:
            wide_out = self.wide_net(wide_features)

        if self.use_wide and self.use_deep:
            fused = self.fusion(wide_out, deep_out)
        elif self.use_deep:
            fused = deep_out
        else:
            fused = wide_out

        return self.classifier(fused).squeeze(-1)


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


def evaluate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    rows = []
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
                rows.append({'bug_id': bug_id, 'file': file_path, 'score': float(score), 'label': float(label)})
    metrics = compute_ranking_metrics(rows)
    metrics['loss'] = total_loss / max(len(dataloader), 1)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', default='ranker/data/manual_line_stage3_light/train.jsonl')
    parser.add_argument('--valid-file', default='ranker/data/manual_line_stage3_light/valid.jsonl')
    parser.add_argument('--model-name', default='microsoft/codebert-base')
    parser.add_argument('--output-dir', default='ranker/outputs/line_ranker')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--grad-accum-steps', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--deep-hidden', type=int, default=256)
    parser.add_argument('--wide-hidden', type=int, default=64)
    parser.add_argument('--pooling', choices=['cls', 'mean'], default='mean')
    parser.add_argument('--focal-alpha', type=float, default=0.25)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--feature-keys', nargs='+', default=['all'])
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

    selected_feature_keys = LINE_FEATURE_KEYS if args.feature_keys == ['all'] else args.feature_keys
    unknown = [key for key in selected_feature_keys if key not in LINE_FEATURE_KEYS]
    if unknown:
        raise ValueError(f'Unknown line feature keys: {unknown}')

    use_wide = not args.disable_wide
    use_deep = not args.disable_deep

    normalizer = None if args.no_normalize else FeatureNormalizer(selected_feature_keys)

    train_ds_raw = LineRankingDataset(train_path, feature_keys=selected_feature_keys)
    if normalizer is not None:
        normalizer.fit(train_ds_raw.rows)
    train_ds = LineRankingDataset(train_path, feature_keys=selected_feature_keys, normalizer=normalizer)
    valid_ds = LineRankingDataset(valid_path, feature_keys=selected_feature_keys, normalizer=normalizer)

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
    model = WideDeepLineRankerV2(
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
    print(f'[OK] Saved best line ranker to {output_dir} (best epoch: {best_epoch}, MRR: {best_metric:.4f})')


if __name__ == '__main__':
    main()
