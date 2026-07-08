# CrashLocCAE Artifact

This repository contains the source code required to rebuild datasets, train and
evaluate the rankers, run comparison baselines, and regenerate the paper tables
and figures. The data, trained models, predictions, and generated results are
distributed separately through the companion Figshare artifact.

## Repository Contents

- `ranker/`: file-level and statement-level ranking models, evaluation scripts,
  comparison baselines, result collection, and plotting utilities.
- `module_map/`: manifestation-aware module partitioning and crash-to-module
  analysis utilities.
- `scripts/`: issue, repository, and ground-truth extraction scripts.
- `config/` and `ranker/config/`: public configuration templates only.
- `REPRODUCE.md`: end-to-end reproduction commands.
- `DATASET.md`: Figshare package layout and expected data locations.

## Data

The GitHub repository intentionally excludes datasets, model checkpoints,
predictions, generated results, raw issue corpora, and local repository clones.
Download the companion Figshare artifact and unpack it into the repository root:

```bash
tar --zstd -xf data_core.tar.zst
tar --zstd -xf models.tar.zst
tar --zstd -xf results.tar.zst
tar --zstd -xf metadata.tar.zst
sha256sum -c SHA256SUMS
```

During anonymous review, use the private Figshare review link supplied in the
paper submission system. After acceptance, replace that link with the public
Figshare DOI.

## Quick Checks

```bash
python -m py_compile ranker/*.py module_map/*.py scripts/*.py
python ranker/collect_results_matrix.py
python ranker/plot_recovered_paper_figures.py
```

## License

Code is released under the MIT License. The companion dataset is intended to be
released under CC BY 4.0 after the anonymous review period.
