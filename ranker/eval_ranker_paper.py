#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from train_ranker_paper import RankingDataset, WideDeepRanker, collate_fn, compute_ranking_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='ranker/outputs/paper_faithful')
    parser.add_argument('--test-file', default='ranker/data/manual_stage2_deep/test.jsonl')
    parser.add_argument('--output-file', default=None)
    parser.add_argument('--predictions-file', default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    ckpt = torch.load(model_dir / 'ranker.pt', map_location='cpu')
    model_args = ckpt['args']
    feature_keys = ckpt['feature_keys']
    use_wide = not model_args.get('disable_wide', False)
    use_deep = not model_args.get('disable_deep', False)

    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'encoder') if use_deep else None
    test_ds = RankingDataset(Path(args.test_file).resolve(), feature_keys=feature_keys)
    test_loader = DataLoader(
        test_ds, batch_size=model_args.get('batch_size', 4), shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, model_args.get('max_length', 384), use_deep),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WideDeepRanker(
        model_name=str(model_dir / 'encoder') if use_deep else model_args.get('model_name', 'distilroberta-base'),
        wide_dim=len(feature_keys),
        deep_hidden=model_args.get('deep_hidden', 256),
        use_wide=use_wide, use_deep=use_deep,
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])

    model.eval()
    rows = []
    with torch.no_grad():
        for batch in test_loader:
            bug_ids = batch.pop('bug_ids')
            files = batch.pop('files')
            labels = batch['labels']
            logits = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                wide_features=batch['wide_features'].to(device),
            )
            scores = torch.sigmoid(logits).cpu().tolist()
            for bug_id, f, s, l in zip(bug_ids, files, scores, labels.tolist()):
                rows.append({'bug_id': bug_id, 'file': f, 'score': float(s), 'label': float(l)})

    metrics = compute_ranking_metrics(rows)
    out = {'metrics': metrics, 'num_pairs': len(rows), 'num_bugs': len({r['bug_id'] for r in rows})}
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.output_file:
        Path(args.output_file).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.predictions_file:
        p = Path(args.predictions_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
