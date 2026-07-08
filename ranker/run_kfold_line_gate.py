#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Statement-level 5-fold CV with Focal Loss + CAE Gate."""

from __future__ import annotations
import json, sys, os, subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line: rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')

def metrics(rows):
    groups = defaultdict(list)
    for r in rows: groups[r['bug_id']].append(r)
    t1=t5=t10=t20=t50=0; fr=ar=0.0; valid=0
    for bid, items in groups.items():
        pos = sum(1 for x in items if x['label']>0.5)
        if pos==0: continue
        valid+=1
        ranked = sorted(items, key=lambda x:(-x['score'],x['file']))
        pr = [i+1 for i,x in enumerate(ranked) if x['label']>0.5]
        best=min(pr); avg=sum(pr)/len(pr)
        t1+=1 if best<=1 else 0; t5+=1 if best<=5 else 0
        t10+=1 if best<=10 else 0; t20+=1 if best<=20 else 0; t50+=1 if best<=50 else 0
        fr+=best; ar+=avg
    if valid==0: return {'top1':0,'top5':0,'top10':0,'top20':0,'top50':0,'mar':0,'mfr':0,'bugs':0}
    return {'top1':int(t1),'top5':int(t5),'top10':int(t10),'top20':int(t20),'top50':int(t50),
            'mar':ar/valid,'mfr':fr/valid,'bugs':valid}

def eval_model(model_dir, test_file, preds_file):
    import torch
    sys.path.insert(0, str(ROOT/'ranker'))
    from train_ranker_gate import RankingDataset, WideDeepRanker, collate_fn, compute_ranking_metrics
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    ckpt = torch.load(model_dir / 'ranker.pt', map_location='cpu')
    feature_keys = ckpt['feature_keys']
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'encoder')
    test_ds = RankingDataset(test_file, feature_keys=feature_keys)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, 256, True))
    device = torch.device('cuda')
    model = WideDeepRanker(model_name=str(model_dir/'encoder'), wide_dim=len(feature_keys),
        deep_hidden=256, cae_gate=True, cae_start_idx=7).to(device)
    model.load_state_dict(ckpt['state_dict']); model.eval()
    rows = []
    with torch.no_grad():
        for batch in test_loader:
            bug_ids=batch.pop('bug_ids'); files=batch.pop('files'); labels=batch['labels']
            logits=model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), wide_features=batch['wide_features'].to(device))
            scores=torch.sigmoid(logits).cpu().tolist()
            for bid,f,s,l in zip(bug_ids,files,scores,labels.tolist()):
                rows.append({'bug_id':bid,'file':f,'score':float(s),'label':float(l)})
    write_jsonl(preds_file, rows)
    return compute_ranking_metrics(rows)

def main():
    output_dir = ROOT / 'ranker' / 'results' / 'kfold_line_gate'
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / 'ranker' / 'data' / 'manual_line_stage3_cae'

    all_rows = read_jsonl(data_dir / 'all.jsonl')
    bug_ids = sorted(set(r['bug_id'] for r in all_rows))
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(bug_ids))
    k=5; fold_size=len(bug_ids)//k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k-1 else len(bug_ids)
        test_idx = set(indices[start:end])
        test_ids = [bug_ids[j] for j in test_idx]
        train_ids = [bug_ids[j] for j in range(len(bug_ids)) if j not in test_idx]
        folds.append((train_ids, test_ids))

    all_fold_results = {m: [] for m in ['CrashLocCAE','LLM-Only','CrashLocator','Scaffle','w/o CAE Features']}

    for fold_idx, (train_ids, test_ids) in enumerate(folds):
        fold_dir = output_dir / f'fold_{fold_idx}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_set = set(train_ids); test_set = set(test_ids)
        train_rows = [r for r in all_rows if r['bug_id'] in train_set]
        test_rows = [r for r in all_rows if r['bug_id'] in test_set]
        train_file = fold_dir / 'train.jsonl'
        test_file = fold_dir / 'test.jsonl'
        write_jsonl(train_file, train_rows)
        write_jsonl(test_file, test_rows)
        test_pos_bugs = set(r['bug_id'] for r in test_rows if r['label']>0.5)

        print(f'\n{"="*60}')
        print(f'FOLD {fold_idx+1}/5: train={len(train_rows)} pairs, test={len(test_rows)} pairs ({len(test_pos_bugs)} pos bugs)')
        print(f'{"="*60}')

        # Train
        model_dir = fold_dir / 'model'
        if not (model_dir / 'ranker.pt').exists():
            print(f'\n--- Train ---')
            subprocess.run([PYTHON, 'ranker/train_ranker_gate.py',
                '--train-file', str(train_file), '--valid-file', str(test_file),
                '--output-dir', str(model_dir), '--model-name', 'distilroberta-base',
                '--epochs', '4', '--batch-size', '8', '--lr', '2e-5', '--max-length', '256',
                '--focal-alpha', '0.25', '--focal-gamma', '2.0'],
                env={**os.environ, 'MKL_SERVICE_FORCE_INTEL':'1'}, timeout=1800)

        # Evaluate
        preds_file = fold_dir / 'preds.jsonl'
        if not preds_file.exists():
            print(f'\n--- Evaluate ---')
            m = eval_model(model_dir, test_file, preds_file)
            print(f'  gate: top1={m["top1"]} top10={m["top10"]} mar={m["mar"]:.1f}')

        # w/o CAE
        no_cae_model = fold_dir / 'model_no_cae'
        no_cae_preds = fold_dir / 'preds_no_cae.jsonl'
        if not (no_cae_model / 'ranker.pt').exists():
            print(f'\n--- Train w/o CAE ---')
            cae_pfx = ('num_','solver_','phys_','sys_','line_')
            def strip_cae(in_path, out_path):
                rows = read_jsonl(in_path)
                for r in rows:
                    r['features'] = {k:v for k,v in r['features'].items() if not any(k.startswith(p) for p in cae_pfx)}
                write_jsonl(out_path, rows)
            strip_cae(train_file, fold_dir / 'train_no_cae.jsonl')
            strip_cae(test_file, fold_dir / 'test_no_cae.jsonl')
            subprocess.run([PYTHON, 'ranker/train_ranker_paper.py',
                '--train-file', str(fold_dir / 'train_no_cae.jsonl'),
                '--valid-file', str(fold_dir / 'test_no_cae.jsonl'),
                '--output-dir', str(no_cae_model), '--model-name', 'distilroberta-base',
                '--epochs', '4', '--batch-size', '8', '--lr', '2e-5', '--max-length', '256'],
                env={**os.environ, 'MKL_SERVICE_FORCE_INTEL':'1'}, timeout=1800)
        if not no_cae_preds.exists():
            subprocess.run([PYTHON, 'ranker/eval_ranker_paper.py',
                '--model-dir', str(no_cae_model), '--test-file', str(test_file),
                '--predictions-file', str(no_cae_preds)],
                env={**os.environ, 'MKL_SERVICE_FORCE_INTEL':'1'}, timeout=300)

        # File-level scores for baselines (inherit to statement level)
        file_preds = read_jsonl(Path(f'ranker/results/kfold_gate/fold_{fold_idx}/crashloccae_gate_preds.jsonl'))
        file_scores = {}
        for r in file_preds:
            file_scores[(r['bug_id'], r['file'])] = r['score']

        nc_file_preds = read_jsonl(Path(f'ranker/results/kfold_gate/fold_{fold_idx}/no_cae_preds.jsonl'))
        nc_file_scores = {}
        for r in nc_file_preds:
            nc_file_scores[(r['bug_id'], r['file'])] = r['score']

        # CrashLocator
        sys.path.insert(0, str(ROOT/'ranker'))
        from eval_crashlocator_enhanced import EnhancedCrashLocator
        all_file_rows = read_jsonl(ROOT / 'ranker' / 'data' / 'manual_stage2_deep' / 'all.jsonl')
        cl = EnhancedCrashLocator(all_file_rows, ROOT/'repos', ROOT/'ranker'/'cache'/'enhanced_crashlocator')
        cl_scores = {}
        for r in test_rows:
            fp = r['file'].rsplit(':',1)[0] if ':' in r['file'] else r['file']
            key = (r['bug_id'], fp)
            if key not in cl_scores:
                row_data = {'bug_id':r['bug_id'],'file':fp,'bug_text':r.get('bug_text',''),
                           'features':r.get('features',{}),'bug_meta':r.get('bug_meta',{}),'project':r.get('project','')}
                cl_scores[key] = cl.score(row_data)

        # Scaffle
        from eval_comparison_methods import BM25Index, scaffle_style
        file_test = read_jsonl(ROOT / 'ranker' / 'data' / 'manual_stage2_deep' / 'test.jsonl')
        bm25 = BM25Index(file_test); cache={}
        sf_scores = {}
        for r in file_test:
            sf_scores[(r['bug_id'], r['file'])] = scaffle_style(r, bm25, cache)

        # LLM-Only
        llm_scores = {}
        with open(ROOT / 'ranker' / 'results' / 'predictions' / 'file_llm_only_rankings.jsonl') as f:
            for line in f:
                line=line.strip()
                if line:
                    r2=json.loads(line)
                    ranking=r2.get('ranking',[])
                    for idx,fname in enumerate(ranking):
                        llm_scores[(r2['bug_id'],fname)]=len(ranking)-idx

        def to_rows(file_scores_dict):
            rows = []
            for r in test_rows:
                fp = r['file'].rsplit(':',1)[0] if ':' in r['file'] else r['file']
                score = file_scores_dict.get((r['bug_id'],fp), 0.0)
                rows.append({'bug_id':r['bug_id'],'file':r['file'],'label':float(r['label']),'score':float(score)})
            return rows

        gate_preds = read_jsonl(preds_file)
        nc_preds = read_jsonl(no_cae_preds) if no_cae_preds.exists() else []

        methods = {
            'CrashLocCAE': gate_preds,
            'LLM-Only': to_rows(llm_scores),
            'CrashLocator': to_rows(cl_scores),
            'Scaffle': to_rows(sf_scores),
            'w/o CAE Features': nc_preds,
        }

        print(f'\n--- Fold {fold_idx+1} Results ---')
        for mname, preds in methods.items():
            if not preds: continue
            m = metrics(preds)
            all_fold_results[mname].append(m)
            print(f'  {mname:<20} t1={m["top1"]:>3} t5={m["top5"]:>3} t10={m["top10"]:>3} t20={m["top20"]:>3} t50={m["top50"]:>3} mar={m["mar"]:.1f} mfr={m["mfr"]:.1f}')

    # Summary
    print(f'\n{"="*80}')
    print('STATEMENT-LEVEL 5-FOLD CV RESULTS (Focal + CAE Gate)')
    print(f'{"="*80}')
    summary = {}
    for mname, fold_list in all_fold_results.items():
        if not fold_list: continue
        summary[mname] = {}
        for m in ['top1','top5','top10','top20','top50','mar','mfr','bugs']:
            vals = [f.get(m,0) for f in fold_list]
            summary[mname][f'{m}_mean'] = float(np.mean(vals))
            summary[mname][f'{m}_std'] = float(np.std(vals))

    methods_order = ['CrashLocCAE','LLM-Only','CrashLocator','Scaffle','w/o CAE Features']
    def f(m,s): return f'{m:.1f}+/-{s:.1f}'
    print(f'\n{"Method":<20} {"Top-1":>10} {"Top-5":>10} {"Top-10":>11} {"Top-20":>11} {"Top-50":>11} {"MAR":>12} {"MFR":>12}')
    for mname in methods_order:
        d = summary.get(mname, {})
        print(f'{mname:<20} {f(d.get("top1_mean",0),d.get("top1_std",0)):>10} {f(d.get("top5_mean",0),d.get("top5_std",0)):>10} {f(d.get("top10_mean",0),d.get("top10_std",0)):>11} {f(d.get("top20_mean",0),d.get("top20_std",0)):>11} {f(d.get("top50_mean",0),d.get("top50_std",0)):>11} {f(d.get("mar_mean",0),d.get("mar_std",0)):>12} {f(d.get("mfr_mean",0),d.get("mfr_std",0)):>12}')

    (output_dir / 'results.json').write_text(json.dumps(summary, indent=2))
    print(f'\n[OK] Saved to {output_dir / "results.json"}')

if __name__ == '__main__':
    main()
