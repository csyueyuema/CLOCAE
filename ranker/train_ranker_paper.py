#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Paper-faithful Wide & Deep ranker (Section 4.2.3).

Architecture:
  Wide: h_wide = W_wide · ϕ_wide + b_wide
  Deep: h_deep = MLP(ϕ_deep)  [2-3 FC layers with ReLU]
  Fusion: y = sigmoid(W_out · [h_wide; h_deep] + b_out)
  Loss: BCE
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

FEATURE_GROUPS = {
    'stack': ['stack_file_exact_match', 'stack_file_suffix_match'],
    'structure': [
        'bug_path_token_overlap', 'bug_name_token_overlap',
        'bug_dep_token_overlap', 'module_match_flag', 'module_margin',
    ],
    'domain': [],  # dynamically populated
}


def get_feature_keys() -> List[str]:
    base = [
        'stack_file_exact_match', 'stack_file_suffix_match',
        'bug_path_token_overlap', 'bug_name_token_overlap',
        'bug_dep_token_overlap', 'module_match_flag', 'module_margin',
    ]
    return base


class RankingDataset(Dataset):
    def __init__(self, path: Path, feature_keys: Optional[List[str]] = None):
        self.rows = []
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if feature_keys is None:
            sample_feats = self.rows[0]['features']
            self.feature_keys = [k for k in sample_feats.keys()]
        else:
            self.feature_keys = feature_keys

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        text = (
            f"[BUG] {row['bug_text']}\n"
            f"[FILE_PATH] {row['file_path_text']}\n"
            f"[FILE_NAME] {row['file_name_text']}\n"
            f"[FILE_TEXT] {row['file_text']}"
        )
        features = [float(row['features'].get(k, 0.0)) for k in self.feature_keys]
        return {
            'bug_id': row['bug_id'],
            'file': row['file'],
            'text': text,
            'features': features,
            'label': float(row['label']),
        }


class WideDeepRanker(nn.Module):
    def __init__(self, model_name: str, wide_dim: int, deep_hidden: int = 256, use_wide: bool = True, use_deep: bool = True):
        super().__init__()
        self.use_wide = use_wide
        self.use_deep = use_deep
        hidden_parts = []

        if use_deep:
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size
            self.deep_net = nn.Sequential(
                nn.Linear(hidden_size, deep_hidden),
                nn.ReLU(),
                nn.Linear(deep_hidden, deep_hidden),
                nn.ReLU(),
            )
            hidden_parts.append(deep_hidden)
        else:
            self.encoder = None

        if use_wide:
            self.wide_linear = nn.Linear(wide_dim, wide_dim)
            hidden_parts.append(wide_dim)

        total = sum(hidden_parts)
        self.output = nn.Linear(total, 1)

    def forward(self, input_ids, attention_mask, wide_features):
        parts = []
        if self.use_deep:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_vec = outputs.last_hidden_state[:, 0, :]
            parts.append(self.deep_net(cls_vec))
        if self.use_wide:
            parts.append(self.wide_linear(wide_features))
        h = torch.cat(parts, dim=-1)
        return self.output(h).squeeze(-1)


def collate_fn(batch, tokenizer, max_length: int, use_deep: bool = True):
    if use_deep:
        texts = [item['text'] for item in batch]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    else:
        bs = len(batch)
        enc = {'input_ids': torch.zeros((bs, 1), dtype=torch.long), 'attention_mask': torch.zeros((bs, 1), dtype=torch.long)}
    enc['wide_features'] = torch.tensor([item['features'] for item in batch], dtype=torch.float)
    enc['labels'] = torch.tensor([item['label'] for item in batch], dtype=torch.float)
    enc['bug_ids'] = [item['bug_id'] for item in batch]
    enc['files'] = [item['file'] for item in batch]
    return enc


def compute_ranking_metrics(rows: List[Dict]) -> Dict[str, float]:
    groups: Dict[str, List] = {}
    for row in rows:
        groups.setdefault(row['bug_id'], []).append(row)
    top1 = top3 = top5 = top10 = 0.0
    rr = fr = ar = 0.0
    valid = 0
    for bug_id, items in groups.items():
        pos = sum(1 for x in items if x['label'] > 0.5)
        if pos == 0:
            continue
        valid += 1
        ranked = sorted(items, key=lambda x: (-x['score'], x['file']))
        pos_ranks = [i + 1 for i, x in enumerate(ranked) if x['label'] > 0.5]
        best = min(pos_ranks)
        avg = sum(pos_ranks) / len(pos_ranks)
        top1 += 1.0 if best <= 1 else 0.0
        top3 += 1.0 if best <= 3 else 0.0
        top5 += 1.0 if best <= 5 else 0.0
        top10 += 1.0 if best <= 10 else 0.0
        rr += 1.0 / best
        fr += best
        ar += avg
    if valid == 0:
        return {'top1': 0, 'top3': 0, 'top5': 0, 'top10': 0, 'mrr': 0, 'mfr': 0, 'mar': 0}
    return {
        'top1': top1 / valid, 'top3': top3 / valid, 'top5': top5 / valid,
        'top10': top10 / valid, 'mrr': rr / valid, 'mfr': fr / valid, 'mar': ar / valid,
    }


def evaluate(model, dataloader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in dataloader:
            bug_ids = batch.pop('bug_ids')
            files = batch.pop('files')
            labels = batch['labels']
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            wide_features = batch['wide_features'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, wide_features=wide_features)
            scores = torch.sigmoid(logits).cpu().tolist()
            for bug_id, f, s, l in zip(bug_ids, files, scores, labels.tolist()):
                rows.append({'bug_id': bug_id, 'file': f, 'score': float(s), 'label': float(l)})
    return compute_ranking_metrics(rows), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', default='ranker/data/manual_stage2_deep/train.jsonl')
    parser.add_argument('--valid-file', default='ranker/data/manual_stage2_deep/valid.jsonl')
    parser.add_argument('--model-name', default='distilroberta-base')
    parser.add_argument('--output-dir', default='ranker/outputs/paper_faithful')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max-length', type=int, default=384)
    parser.add_argument('--deep-hidden', type=int, default=256)
    parser.add_argument('--disable-wide', action='store_true')
    parser.add_argument('--disable-deep', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_path = Path(args.train_file).resolve()
    valid_path = Path(args.valid_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    use_wide = not args.disable_wide
    use_deep = not args.disable_deep

    train_ds = RankingDataset(train_path)
    valid_ds = RankingDataset(valid_path, feature_keys=train_ds.feature_keys)
    feature_keys = train_ds.feature_keys

    tokenizer = AutoTokenizer.from_pretrained(args.model_name) if use_deep else None
    if tokenizer and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length, use_deep),
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length, use_deep),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WideDeepRanker(
        model_name=args.model_name,
        wide_dim=len(feature_keys),
        deep_hidden=args.deep_hidden,
        use_wide=use_wide,
        use_deep=use_deep,
    ).to(device)

    pos_count = sum(1 for r in train_ds.rows if r['label'] > 0.5)
    neg_count = len(train_ds.rows) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {'params': [p for n, p in model.named_parameters() if not any(d in n for d in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(d in n for d in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    best_mrr = -1
    best_epoch = 0
    patience = 3
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            labels = batch['labels'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            wide_features = batch['wide_features'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask, wide_features=wide_features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        metrics, _ = evaluate(model, valid_loader, device)
        metrics['epoch'] = epoch
        metrics['train_loss'] = total_loss / max(len(train_loader), 1)
        print(json.dumps(metrics, ensure_ascii=False))

        if metrics['mrr'] > best_mrr:
            best_mrr = metrics['mrr']
            best_epoch = epoch
            patience_counter = 0
            if model.encoder is not None:
                model.encoder.save_pretrained(output_dir / 'encoder')
                tokenizer.save_pretrained(output_dir / 'encoder')
            torch.save({
                'state_dict': model.state_dict(),
                'feature_keys': feature_keys,
                'args': vars(args),
            }, output_dir / 'ranker.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'[EARLY STOP] Best at epoch {best_epoch}')
                break

    print(f'[OK] Saved to {output_dir} (best epoch: {best_epoch}, MRR: {best_mrr:.4f})')


if __name__ == '__main__':
    main()
