#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Complete 5-fold CV with the CAE-gated model. Trains all folds, evaluates all methods, collects RQ1-RQ4."""

from __future__ import annotations
import json, sys, os, subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

CATEGORY_MAP = {
    'CalculiX-Examples': '1', 'Calculix': '1', 'calculix': '1',
    'deal.II': '1', 'dealii': '1', 'FireDrake': '1', 'firedrake': '1',
    'FreeFem': '1', 'FreeFEM': '1', 'gridap': '1', 'Gridap': '1',
    'MFEM': '1', 'mfem': '1', 'SfePy': '1', 'sfepy': '1',
    'Sparselizard': '1', 'sparselizard': '1',
    'code': '2', 'Code Saturne': '2', 'Code_Saturne': '2', 'Code-Saturne': '2',
    'coolfluid': '2', 'COOLFluiD': '2', 'FDS': '2', 'fds': '2',
    'Fluidity': '2', 'Nek5000': '2', 'SU2': '2', 'su2': '2',
    'xcompact3d': '2', 'Xcompact3d': '2',
    'Goma': '3', 'goma': '3', 'Kratos': '3', 'kratos': '3',
    'OpenModelica': '4', 'openmodelica': '4', 'ROSS': '4', 'ross': '4',
}
CAT_NAMES = {'1': 'FEM', '2': 'CFD', '3': 'Multiphysics', '4': 'Modeling'}

def cat_for(proj):
    if proj in CATEGORY_MAP: return CATEGORY_MAP[proj]
    n = proj.lower().replace('_','').replace('-','').replace('.','').replace(' ','')
    for name, cat in CATEGORY_MAP.items():
        if n == name.lower().replace('_','').replace('-','').replace('.','').replace(' ',''):
            return cat
    return 'unknown'

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
    t1=t3=t5=t10=0; rr=fr=ar=0.0; valid=0
    for bid, items in groups.items():
        pos = sum(1 for x in items if x['label']>0.5)
        if pos==0: continue
        valid += 1
        ranked = sorted(items, key=lambda x:(-x['score'],x['file']))
        pr = [i+1 for i,x in enumerate(ranked) if x['label']>0.5]
        best=min(pr); avg=sum(pr)/len(pr)
        t1+=1 if best<=1 else 0; t3+=1 if best<=3 else 0
        t5+=1 if best<=5 else 0; t10+=1 if best<=10 else 0
        rr+=1/best; fr+=best; ar+=avg
    if valid==0: return {'top1':0,'top3':0,'top5':0,'top10':0,'mrr':0,'mfr':0,'mar':0,'bugs':0}
    return {'top1':int(t1),'top3':int(t3),'top5':int(t5),'top10':int(t10),'mrr':rr/valid,'mfr':fr/valid,'mar':ar/valid,'bugs':valid}

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
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, 384, True))
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
    m = compute_ranking_metrics(rows)
    return m

def eval_no_cae(fold_dir, train_file, test_file):
    import torch
    sys.path.insert(0, str(ROOT/'ranker'))
    from train_ranker_gate import RankingDataset, WideDeepRanker, collate_fn, compute_ranking_metrics, FocalLoss
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    cae_pfx = ('num_','solver_','phys_','sys_')
    def strip_cae(in_path, out_path):
        rows = read_jsonl(in_path)
        for r in rows:
            r['features'] = {k:v for k,v in r['features'].items() if not any(k.startswith(p) for p in cae_pfx)}
        write_jsonl(out_path, rows)

    no_cae_train = fold_dir / 'train_no_cae.jsonl'
    no_cae_test = fold_dir / 'test_no_cae.jsonl'
    no_cae_model = fold_dir / 'model_no_cae'
    no_cae_preds = fold_dir / 'no_cae_preds.jsonl'

    if not (no_cae_model / 'ranker.pt').exists():
        strip_cae(train_file, no_cae_train)
        strip_cae(test_file, no_cae_test)
        subprocess.run([PYTHON, 'ranker/train_ranker_paper.py',
            '--train-file', str(no_cae_train), '--valid-file', str(no_cae_test),
            '--output-dir', str(no_cae_model), '--model-name', 'distilroberta-base',
            '--epochs', '4', '--batch-size', '4', '--lr', '2e-5', '--max-length', '384'],
            env={**os.environ, 'MKL_SERVICE_FORCE_INTEL':'1'}, timeout=1200)

    if not no_cae_preds.exists():
        subprocess.run([PYTHON, 'ranker/eval_ranker_paper.py',
            '--model-dir', str(no_cae_model), '--test-file', str(no_cae_test if (fold_dir/'test_no_cae.jsonl').exists() else test_file),
            '--predictions-file', str(no_cae_preds)],
            env={**os.environ, 'MKL_SERVICE_FORCE_INTEL':'1'}, timeout=300)

    return read_jsonl(no_cae_preds) if no_cae_preds.exists() else []

def main():
    output_dir = ROOT / 'ranker' / 'results' / 'kfold_gate'
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = read_jsonl(ROOT / 'ranker' / 'data' / 'manual_stage2_deep' / 'all.jsonl')
    proj_map = {r['bug_id']: r.get('project','') for r in all_rows}
    bug_ids = sorted(set(r['bug_id'] for r in all_rows))

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(bug_ids))
    k = 5; fold_size = len(bug_ids) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k-1 else len(bug_ids)
        test_idx = set(indices[start:end])
        test_ids = [bug_ids[j] for j in test_idx]
        train_ids = [bug_ids[j] for j in range(len(bug_ids)) if j not in test_idx]
        folds.append((train_ids, test_ids))

    sys.path.insert(0, str(ROOT/'ranker'))
    from eval_crashlocator_enhanced import EnhancedCrashLocator
    from eval_comparison_methods import BM25Index, scaffle_style

    cl = EnhancedCrashLocator(all_rows, ROOT/'repos', ROOT/'ranker'/'cache'/'enhanced_crashlocator')
    llm_rankings = {}
    with open(ROOT/'ranker'/'results'/'predictions'/'file_llm_only_rankings.jsonl') as f:
        for line in f:
            line = line.strip()
            if line: r = json.loads(line); llm_rankings[r['bug_id']] = r

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
        print(f'FOLD {fold_idx+1}/5: train={len(train_ids)}, test={len(test_ids)} ({len(test_pos_bugs)} with positive)')
        print(f'{"="*60}')

        # Train gated model
        model_dir = fold_dir / 'model_gate'
        if not (model_dir / 'ranker.pt').exists():
            print(f'\n--- Train gated model ---')
            subprocess.run([PYTHON, 'ranker/train_ranker_gate.py',
                '--train-file', str(train_file), '--valid-file', str(test_file),
                '--output-dir', str(model_dir), '--model-name', 'distilroberta-base',
                '--epochs', '4', '--batch-size', '4', '--lr', '2e-5', '--max-length', '384',
                '--focal-alpha', '0.25', '--focal-gamma', '2.0'],
                env={**os.environ, 'MKL_SERVICE_FORCE_INTEL':'1'}, timeout=1800)

        # Evaluate gated model
        preds_file = fold_dir / 'crashloccae_gate_preds.jsonl'
        if not preds_file.exists():
            print(f'\n--- Evaluate gated model ---')
            m = eval_model(model_dir, test_file, preds_file)
            print(f'  gate: top1={m["top1"]} mar={m["mar"]:.2f}')

        # w/o CAE
        print(f'\n--- w/o CAE ---')
        nc_preds = eval_no_cae(fold_dir, train_file, test_file)

        # Other methods
        gate_preds = read_jsonl(preds_file)
        cl_preds = [{'bug_id':r['bug_id'],'file':r['file'],'label':float(r['label']),'score':float(cl.score(r))} for r in test_rows]
        bm25 = BM25Index(test_rows); cache={}
        sf_preds = [{'bug_id':r['bug_id'],'file':r['file'],'label':float(r['label']),'score':float(scaffle_style(r,bm25,cache))} for r in test_rows]
        llm_preds = []
        for row in test_rows:
            pred = llm_rankings.get(row['bug_id'])
            if not pred: continue
            ranking = pred.get('ranking',[]); sm = {item:len(ranking)-i for i,item in enumerate(ranking)}
            llm_preds.append({'bug_id':row['bug_id'],'file':row['file'],'label':float(row['label']),'score':float(sm.get(row['file'],0))})

        methods = {'CrashLocCAE':gate_preds, 'LLM-Only':llm_preds, 'CrashLocator':cl_preds, 'Scaffle':sf_preds, 'w/o CAE Features':nc_preds}

        print(f'\n--- Fold {fold_idx+1} Results ---')
        for mname, preds in methods.items():
            if not preds: continue
            cat_m = {}
            for cid in ['1','2','3','4']:
                cat_bugs = {b for b in test_pos_bugs if cat_for(proj_map.get(b,''))==cid}
                cat_preds = [r for r in preds if r['bug_id'] in cat_bugs]
                cat_m[cid] = metrics(cat_preds)
            cat_m['overall'] = metrics(preds)
            all_fold_results[mname].append(cat_m)
            t1s = ' '.join(f'{cat_m[c]["top1"]:>3}' for c in ['1','2','3','4'])
            print(f'  {mname:<20} cat_top1=[{t1s}] overall={cat_m["overall"]["top1"]:>3} mar={cat_m["overall"]["mar"]:.2f}')

    # Summary
    print(f'\n{"="*80}')
    print('5-FOLD CV RESULTS (Focal + CAE Gate)')
    print(f'{"="*80}')

    summary = {}
    for mname, fold_list in all_fold_results.items():
        if not fold_list: continue
        summary[mname] = {}
        for cid in ['1','2','3','4','overall']:
            for m in ['top1','top3','top5','top10','mar','mfr','bugs']:
                vals = [f.get(cid,{}).get(m,0) for f in fold_list]
                summary[mname][f'{cid}_{m}_mean'] = float(np.mean(vals))
                summary[mname][f'{cid}_{m}_std'] = float(np.std(vals))

    def fmt(m, s): return f'{m:.1f}+/-{s:.1f}'
    def fmt2(m, s): return f'{m:.2f}+/-{s:.2f}'
    methods_order = ['CrashLocCAE','LLM-Only','CrashLocator','Scaffle','w/o CAE Features']
    header = f'{"Method":<20} {"Top-1":>12} {"Top-5":>12} {"Top-10":>13} {"MAR":>14} {"MFR":>14}'

    for cid, cname in CAT_NAMES.items():
        print(f'\n--- {cid}. {cname} ---')
        print(header)
        for mname in methods_order:
            d = summary.get(mname, {})
            print(f'{mname:<20} {fmt(d.get(f"{cid}_top1_mean",0), d.get(f"{cid}_top1_std",0)):>12} {fmt(d.get(f"{cid}_top5_mean",0), d.get(f"{cid}_top5_std",0)):>12} {fmt(d.get(f"{cid}_top10_mean",0), d.get(f"{cid}_top10_std",0)):>13} {fmt2(d.get(f"{cid}_mar_mean",0), d.get(f"{cid}_mar_std",0)):>14} {fmt2(d.get(f"{cid}_mfr_mean",0), d.get(f"{cid}_mfr_std",0)):>14}')

    print(f'\n--- Overall ---')
    print(header)
    for mname in methods_order:
        d = summary.get(mname, {})
        print(f'{mname:<20} {fmt(d.get("overall_top1_mean",0), d.get("overall_top1_std",0)):>12} {fmt(d.get("overall_top5_mean",0), d.get("overall_top5_std",0)):>12} {fmt(d.get("overall_top10_mean",0), d.get("overall_top10_std",0)):>13} {fmt2(d.get("overall_mar_mean",0), d.get("overall_mar_std",0)):>14} {fmt2(d.get("overall_mfr_mean",0), d.get("overall_mfr_std",0)):>14}')

    (output_dir / 'kfold_gate_results.json').write_text(json.dumps(summary, indent=2))
    print(f'\n[OK] Saved to {output_dir / "kfold_gate_results.json"}')

if __name__ == '__main__':
    main()
