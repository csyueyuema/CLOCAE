# Reproduction Guide

## Environment

Use Python 3.9+ with the dependencies in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

For GPU evaluation or retraining, install a PyTorch build compatible with the
local CUDA runtime before installing the remaining dependencies.

## Unpack Data

Download the companion Figshare archives into the repository root and verify
their checksums:

```bash
sha256sum -c SHA256SUMS
tar --zstd -xf data_core.tar.zst
tar --zstd -xf models.tar.zst
tar --zstd -xf results.tar.zst
tar --zstd -xf metadata.tar.zst
```

## Rebuild Result Tables

```bash
python ranker/collect_results_matrix.py
```

Expected key outputs:

- `ranker/results/ALL_RQ_RESULTS.md`
- `ranker/results/tables/rq_matrix_current.json`
- `ranker/results/tables/all_rq_5x4x2_long.csv`

## Regenerate Paper Figures

```bash
python ranker/plot_recovered_paper_figures.py
```

The script writes figures under `ranker/results/figures/`.

## Optional Training Entrypoints

File-level ranker:

```bash
python ranker/train_ranker_current.py \
  --train-file ranker/data/manual_stage2_deep/train.jsonl \
  --valid-file ranker/data/manual_stage2_deep/valid.jsonl \
  --output-dir ranker/outputs/wide_deep_distil \
  --model-name distilroberta-base \
  --epochs 6 --batch-size 4 --grad-accum-steps 2 --lr 2e-5 \
  --max-length 384 --dropout 0.15 --pooling mean --focal-gamma 1.5
```

Statement-level ranker:

```bash
python ranker/train_line_ranker_current.py \
  --train-file ranker/data/manual_line_stage3_light/train.jsonl \
  --valid-file ranker/data/manual_line_stage3_light/valid.jsonl \
  --output-dir ranker/outputs/line_ranker_distil \
  --model-name distilroberta-base \
  --epochs 4 --batch-size 8 --grad-accum-steps 2 --lr 2e-5 \
  --max-length 256 --dropout 0.15 --pooling mean --focal-gamma 1.5
```
