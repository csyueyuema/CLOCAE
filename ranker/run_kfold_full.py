#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Complete 5-fold CV: train, evaluate, collect per-category mean±std."""

from __future__ import annotations
import json, subprocess, sys, os
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
    return {'top1':int(t1),'top3':int(t3),'top5':int(t5),'top10':int(t10),
            'mrr':rr/valid,'mfr':fr/valid,'mar':ar/valid,'bugs':valid}

def run(cmd):
    print(f'  RUN: {cmd[0].split("/")[-1]} {" ".join(cmd[1:5])}...', flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                       env={**os.environ, 'MKL_SERVICE_FORCE_INTEL':'1'})
    if r.returncode != 0:
        print(f'  WARN exit {r.returncode}: {r.stderr[:300]}')
    return r.returncode

def build_no_cae(in_path, out_path):
    rows = read_jsonl(in_path)
    cae_pfx = ('num_','solver_','phys_','sys_')
    for r in rows:
        r['features'] = {k:v for k,v in r['features'].items() if not any(k.startswith(p) for p in cae_pfx)}
    write_jsonl(out_path, rows)
    return len(rows)

def load_predictions(path):
    if not path.exists(): return []
    return read_jsonl(path)

def llm_only_scored(test_rows):
    rankings = {r['bug_id']:r for r in read_jsonl(ROOT/'ranker'/'results'/'predictions'/'file_llm_only_rankings.jsonl')}
    scored = []
    for row in test_rows:
        pred = rankings.get(row['bug_id'])
        if not pred: continue
        ranking = pred.get('ranking',[])
        sm = {item:len(ranking)-i for i,item in enumerate(ranking)}
        scored.append({'bug_id':row['bug_id'],'file':row['file'],'label':float(row['label']),'score':float(sm.get(row['file'],0))})
    return scored

def crashlocator_scored(test_rows, all_rows, repos_dir, cache_dir):
    sys.path.insert(0, str(ROOT/'ranker'))
    from eval_crashlocator_enhanced import EnhancedCrashLocator
    cl = EnhancedCrashLocator(all_rows, repos_dir, cache_dir)
    return [{'bug_id':r['bug_id'],'file':r['file'],'label':float(r['label']),'score':float(cl.score(r))} for r in test_rows]

def scaffle_scored(test_rows):
    sys.path.insert(0, str(ROOT/'ranker'))
    from eval_comparison_methods import BM25Index, scaffle_style
    bm25 = BM25Index(test_rows)
    cache = {}
    return [{'bug_id':r['bug_id'],'file':r['file'],'label':float(r['label']),'score':float(scaffle_style(r,bm25,cache))} for r in test_rows]


def main():
    output_dir = ROOT / 'ranker' / 'results' / 'kfold_cv'
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = read_jsonl(ROOT / 'ranker' / 'data' / 'manual_stage2_deep' / 'all.jsonl')
    proj_map = {r['bug_id']: r.get('project','') for r in all_rows}
    bug_ids = sorted(set(r['bug_id'] for r in all_rows))

    # Split folds
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(bug_ids))
    k = 5
    fold_size = len(bug_ids) // k
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
        train_set = set(train_ids)
        test_set = set(test_ids)
        train_rows = [r for r in all_rows if r['bug_id'] in train_set]
        test_rows = [r for r in all_rows if r['bug_id'] in test_set]

        train_file = fold_dir / 'train.jsonl'
        test_file = fold_dir / 'test.jsonl'
        write_jsonl(train_file, train_rows)
        write_jsonl(test_file, test_rows)

        test_pos_bugs = set(r['bug_id'] for r in test_rows if r['label']>0.5)
        print(f'\n{"="*60}')
        print(f'FOLD {fold_idx+1}/5: train={len(train_ids)} bugs, test={len(test_ids)} bugs ({len(test_pos_bugs)} with positive)')
        print(f'{"="*60}')

        # 1. Train CrashLocCAE
        model_dir = fold_dir / 'model'
        if not (model_dir / 'ranker.pt').exists():
            print('\n--- Train CrashLocCAE ---')
            run([PYTHON, 'ranker/train_ranker_paper.py',
                 '--train-file', str(train_file), '--valid-file', str(test_file),
                 '--output-dir', str(model_dir),
                 '--model-name', 'distilroberta-base',
                 '--epochs', '4', '--batch-size', '4', '--lr', '2e-5', '--max-length', '384'])

        # 2. Evaluate CrashLocCAE
        preds_file = fold_dir / 'crashloccae_preds.jsonl'
        if not preds_file.exists():
            print('\n--- Evaluate CrashLocCAE ---')
            run([PYTHON, 'ranker/eval_ranker_paper.py',
                 '--model-dir', str(model_dir), '--test-file', str(test_file),
                 '--predictions-file', str(preds_file)])

        # 3. Build and train w/o CAE
        no_cae_train = fold_dir / 'train_no_cae.jsonl'
        no_cae_test = fold_dir / 'test_no_cae.jsonl'
        no_cae_model = fold_dir / 'model_no_cae'
        no_cae_preds = fold_dir / 'no_cae_preds.jsonl'

        build_no_cae(train_file, no_cae_train)
        build_no_cae(test_file, no_cae_test)

        if not (no_cae_model / 'ranker.pt').exists():
            print('\n--- Train w/o CAE ---')
            run([PYTHON, 'ranker/train_ranker_paper.py',
                 '--train-file', str(no_cae_train), '--valid-file', str(no_cae_test),
                 '--output-dir', str(no_cae_model),
                 '--model-name', 'distilroberta-base',
                 '--epochs', '4', '--batch-size', '4', '--lr', '2e-5', '--max-length', '384'])

        if not no_cae_preds.exists():
            print('\n--- Evaluate w/o CAE ---')
            run([PYTHON, 'ranker/eval_ranker_paper.py',
                 '--model-dir', str(no_cae_model), '--test-file', str(no_cae_test),
                 '--predictions-file', str(no_cae_preds)])

        # 4. Load all predictions
        cae_preds = load_predictions(preds_file)
        nc_preds = load_predictions(no_cae_preds)
        llm_preds = llm_only_scored(test_rows)
        cl_preds = crashlocator_scored(test_rows, all_rows, ROOT/'repos', ROOT/'ranker'/'cache'/'enhanced_crashlocator')
        sf_preds = scaffle_scored(test_rows)

        methods = {
            'CrashLocCAE': cae_preds,
            'LLM-Only': llm_preds,
            'CrashLocator': cl_preds,
            'Scaffle': sf_preds,
            'w/o CAE Features': nc_preds,
        }

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
    print('5-FOLD CV RESULTS (mean +/- std)')
    print(f'{"="*80}')

    summary = {}
    for mname, fold_list in all_fold_results.items():
        if not fold_list: continue
        summary[mname] = {}
        for cid in ['1','2','3','4','overall']:
            for m in ['top1','top5','top10','mar','mfr','bugs']:
                vals = [f.get(cid,{}).get(m,0) for f in fold_list]
                summary[mname][f'{cid}_{m}_mean'] = float(np.mean(vals))
                summary[mname][f'{cid}_{m}_std'] = float(np.std(vals))

    header = f'{"Method":<20} {"Top-1":>12} {"Top-5":>12} {"Top-10":>13} {"MAR":>14} {"MFR":>14} {"Bugs":>6}'
    for cid, cname in CAT_NAMES.items():
        print(f'\n--- {cid}. {cname} ---')
        print(header)
        for mname in all_fold_results:
            if not all_fold_results[mname]: continue
            d = summary[mname]
            t1m=d.get(f'{cid}_top1_mean',0); t1s=d.get(f'{cid}_top1_std',0)
            t5m=d.get(f'{cid}_top5_mean',0); t5s=d.get(f'{cid}_top5_std',0)
            t10m=d.get(f'{cid}_top10_mean',0); t10s=d.get(f'{cid}_top10_std',0)
            marm=d.get(f'{cid}_mar_mean',0); mars=d.get(f'{cid}_mar_std',0)
            mfrm=d.get(f'{cid}_mfr_mean',0); mfrs=d.get(f'{cid}_mfr_std',0)
            bugs=d.get(f'{cid}_bugs_mean',0)
            print(f'{mname:<20} {t1m:>5.1f}±{t1s:<4.1f} {t5m:>5.1f}±{t5s:<4.1f} {t10m:>5.1f}±{t10s:<5.1f} {marm:>6.2f}±{mars:<5.2f} {mfrm:>6.2f}±{mfrs:<5.2f} {bugs:>5.0f}')

    print(f'\n--- Overall ---')
    print(header)
    for mname in all_fold_results:
        if not all_fold_results[mname]: continue
        d = summary[mname]
        t1m=d.get('overall_top1_mean',0); t1s=d.get('overall_top1_std',0)
        t5m=d.get('overall_top5_mean',0); t5s=d.get('overall_top5_std',0)
        t10m=d.get('overall_top10_mean',0); t10s=d.get('overall_top10_std',0)
        marm=d.get('overall_mar_mean',0); mars=d.get('overall_mar_std',0)
        mfrm=d.get('overall_mfr_mean',0); mfrs=d.get('overall_mfr_std',0)
        bugs=d.get('overall_bugs_mean',0)
        print(f'{mname:<20} {t1m:>5.1f}±{t1s:<4.1f} {t5m:>5.1f}±{t5s:<4.1f} {t10m:>5.1f}±{t10s:<5.1f} {marm:>6.2f}±{mars:<5.2f} {mfrm:>6.2f}±{mfrs:<5.2f} {bugs:>5.0f}')

    (output_dir / 'kfold_results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'\n[OK] Saved to {output_dir / "kfold_results.json"}')

if __name__ == '__main__':
    main()
