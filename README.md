# Feature Selection for Deep Text Classification under Limited Labeled Data

This repository contains the code and released results for the revised empirical study of filter feature selection in deep text classification under limited labeled data.

The study evaluates document frequency (DF), information gain (IG), and the distinguishing feature selector (DFS) across 16 benchmark datasets and 11 neural classifiers. The main experiment uses five-fold cross-validation and fixes the retained vocabulary at 1,000 words. A separate sensitivity experiment evaluates 500, 1,000, 1,500, and 2,000 retained words on four representative datasets.

## Leakage-safe protocol

Feature selection is fitted independently inside every fold:

1. Split a dataset into five outer folds.
2. Fit tokenization, vocabulary construction, and DF/IG/DFS ranking using training data only.
3. Apply the frozen selected vocabulary to both the training and held-out texts.
4. Select the training epoch using an inner validation partition of the outer-training fold.
5. Refit on the complete outer-training fold and evaluate the trained classifier once on the held-out fold.

Test texts and labels never contribute to feature scores, vocabulary construction, model fitting, or epoch selection. See [`docs/experimental_protocol.md`](docs/experimental_protocol.md) for details.

## Experimental scope

| Component | Main experiment |
|---|---|
| Datasets | 16 |
| Outer folds | 5 |
| Feature conditions | None, DF, IG, DFS |
| Retained words | 1,000 for DF/IG/DFS |
| Classic models | Random/FastText × word averaging, CNN, BiLSTM |
| Graph model | FastText residual inductive word GCN |
| BERT models | CLS, masked mean, CNN, BiLSTM-attention |
| Total main evaluations | 3,520 |

The feature-count analysis covers 20NG, Amazon Review Full, Farm Ads, and WOS5736 using the three feature selectors and four vocabulary sizes. It adds 1,440 newly trained configurations and reuses the corresponding no-FS and 1,000-feature formal runs, giving 2,080 validated fold-level rows in the combined sensitivity analysis.

## Repository contents

```text
configs/       Frozen experiment protocols and local configuration templates
data/          Dataset preparation and expected input layout
docs/          Protocol, assets, and reproduction instructions
results/       Released fold-level and aggregate results
scripts/       Auditing, training, orchestration, and summarization commands
src/fs_lrtc/   Reusable Python implementation
tests/         Unit and protocol tests
```

The public repository intentionally excludes pretrained model files, raw benchmark corpora, model checkpoints, and thousands of per-run prediction/history files. The complete numerical tables used by the manuscript are included under [`results/`](results/).

## Installation

Python 3.12 is the frozen environment for the released experiments. On Windows, a short virtual-environment path is recommended because PyTorch contains deeply nested header paths.

```powershell
C:\Python312\python.exe -m venv D:\venvs\fs_lrtc_py312
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& "D:\venvs\fs_lrtc_py312\Scripts\Activate.ps1"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-torch-cu124.txt
python -m pip install -r requirements-experiment.txt
python -m pip install -r requirements-assets.txt
python -m pip install -e .
```

Users without a CUDA 12.4-compatible setup should install the appropriate PyTorch build from the official PyTorch instructions, then install the remaining requirements.

## Verify the release

```powershell
python -m unittest discover -s tests -v
python scripts/verify_published_results.py
```

The second command validates the released row counts, factorial coverage, duplicate keys, and completeness reports without training any model.

## Local configuration and data

Copy the two templates and edit only the copied local files:

```powershell
Copy-Item configs/paths.example.yaml configs/paths.local.yaml
Copy-Item configs/assets.example.yaml configs/assets.local.yaml
```

The local files are ignored by Git because they contain machine-specific paths. Dataset files are not redistributed in this repository. See [`data/README.md`](data/README.md) and [`docs/pretrained_assets.md`](docs/pretrained_assets.md).

## Reproducing the experiments

The shortest reproducible sequence is:

```powershell
python scripts/audit_data.py --paths configs/paths.local.yaml --datasets configs/datasets.yaml
python scripts/prepare_data.py --paths configs/paths.local.yaml --datasets configs/datasets.yaml --config configs/data_preparation.yaml
python scripts/create_splits.py --paths configs/paths.local.yaml --datasets configs/datasets.yaml --preparation-config configs/data_preparation.yaml --split-config configs/splits.yaml

python scripts/run_formal_classic_main.py --paths configs/paths.local.yaml --assets configs/assets.local.yaml
python scripts/run_formal_gcn.py --paths configs/paths.local.yaml --assets configs/assets.local.yaml
python scripts/run_formal_bert.py --paths configs/paths.local.yaml --assets configs/assets.local.yaml
```

These experiments are computationally expensive and support shard selection. Full commands, preflight checks, aggregation steps, and feature-count runs are documented in [`docs/reproduction.md`](docs/reproduction.md).

## Released results

- [`results/main/main_fold_results.csv`](results/main/main_fold_results.csv): all 3,520 main fold-level evaluations.
- [`results/main/main_mean_std.csv`](results/main/main_mean_std.csv): five-fold mean and standard deviation for all 704 dataset–model–condition cells.
- [`results/main/main_significance_tests.csv`](results/main/main_significance_tests.csv): paired statistical tests with multiplicity correction and effect sizes.
- [`results/main/main_runtime_summary.csv`](results/main/main_runtime_summary.csv): runtime summaries.
- [`results/feature_count/`](results/feature_count/): feature-count sensitivity summaries.
- [`results/supplementary/Supplementary_Materials.xlsx`](results/supplementary/Supplementary_Materials.xlsx): Supplementary Tables S1–S3.

## Revision note

The previous repository contained texts that had been feature-filtered before cross-validation. Those files are obsolete and are not part of the revised protocol. The present implementation always estimates supervised feature statistics from the applicable training partition only. See [`RELEASE_NOTES.md`](RELEASE_NOTES.md).

## License and citation

The source-code license and final article citation will be added when the manuscript metadata is finalized. Third-party datasets and pretrained models remain subject to their original licenses.
