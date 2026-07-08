#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train_line_ranker import LINE_FEATURE_KEYS, LineRankingDataset, WideDeepLineRanker, collate_fn, compute_ranking_metrics


def evaluate_model(model, dataloader, device):
    model.eval()
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
            scores = torch.sigmoid(logits).detach().cpu().tolist()
            labels_cpu = labels.detach().cpu().tolist()
            for bug_id, file_path, score, label in zip(bug_ids, files, scores, labels_cpu):
                rows.append({'bug_id': bug_id, 'file': file_path, 'score': float(score), 'label': float(label)})
    return compute_ranking_metrics(rows), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='ranker/outputs/line_ranker_seed')
    parser.add_argument('--test-file', default='ranker/data/manual_line_stage3/test.jsonl')
    parser.add_argument('--output-file', default='ranker/outputs/line_ranker_seed/test_metrics.json')
    parser.add_argument('--predictions-file', default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    ckpt = torch.load(model_dir / 'ranker.pt', map_location='cpu')
    model_args = ckpt['args']
    use_wide = not model_args.get('disable_wide', False)
    use_deep = not model_args.get('disable_deep', False)

    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'encoder') if use_deep else None
    feature_keys = ckpt.get('feature_keys', LINE_FEATURE_KEYS)
    test_ds = LineRankingDataset(Path(args.test_file).resolve(), feature_keys=feature_keys)
    test_loader = DataLoader(
        test_ds,
        batch_size=model_args.get('batch_size', 8),
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, model_args.get('max_length', 256), use_deep=use_deep),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WideDeepLineRanker(
        model_name=str(model_dir / 'encoder') if use_deep else model_args.get('model_name', 'distilroberta-base'),
        wide_dim=len(feature_keys),
        deep_hidden=model_args.get('deep_hidden', 256),
        wide_hidden=model_args.get('wide_hidden', 64),
        dropout=model_args.get('dropout', 0.1),
        use_wide=use_wide,
        use_deep=use_deep,
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])

    metrics, rows = evaluate_model(model, test_loader, device)
    out = {
        'metrics': metrics,
        'num_pairs': len(rows),
        'num_bugs': len({r['bug_id'] for r in rows}),
    }
    Path(args.output_file).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.predictions_file:
        pred_path = Path(args.predictions_file)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
