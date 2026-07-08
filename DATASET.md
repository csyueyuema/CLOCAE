# Companion Figshare Dataset

The companion Figshare item contains all non-code material needed to reproduce
the paper results without publishing the full local 119G worktree.

## Archives

- `data_core.tar.zst`
  - `raw_data/`
  - `issue_corpus/`
  - `labels/manual/`
  - `ranker/data/manual_stage2_deep/`
  - `ranker/data/manual_line_stage3_light/`
  - `ranker/data/manual_line_stage3_no_cae/`
- `models.tar.zst`
  - `ranker/results/models/`
- `results.tar.zst`
  - `ranker/results/tables/`
  - `ranker/results/predictions/`
  - `ranker/results/figures/`
- `metadata.tar.zst`
  - `DATA_MANIFEST.md`
  - `REPRODUCE.md`
  - `SHA256SUMS`

## Verification

Place all archives in the repository root and run:

```bash
sha256sum -c SHA256SUMS
tar --zstd -xf data_core.tar.zst
tar --zstd -xf models.tar.zst
tar --zstd -xf results.tar.zst
tar --zstd -xf metadata.tar.zst
```

The GitHub repository does not include cloned upstream project histories under
`repos/`. The dataset includes issue URLs, fixing files, labels, prepared model
inputs, predictions, and trained model checkpoints used by the submitted paper.
