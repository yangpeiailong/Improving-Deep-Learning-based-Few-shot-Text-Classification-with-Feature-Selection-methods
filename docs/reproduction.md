# Reproduction guide

Run all commands from the repository root in an activated Python 3.12 environment.

## 1. Configure paths and assets

```powershell
Copy-Item configs/paths.example.yaml configs/paths.local.yaml
Copy-Item configs/assets.example.yaml configs/assets.local.yaml
```

Edit the copied files for the local paper/data root, result root, FastText model, and BERT directory.

## 2. Verify installation and data

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v

python scripts/check_environment.py `
  --paths configs/paths.local.yaml `
  --config configs/environment.yaml

python scripts/audit_data.py `
  --paths configs/paths.local.yaml `
  --datasets configs/datasets.yaml
```

## 3. Prepare frozen inputs

```powershell
python scripts/prepare_data.py `
  --paths configs/paths.local.yaml `
  --datasets configs/datasets.yaml `
  --config configs/data_preparation.yaml

python scripts/create_splits.py `
  --paths configs/paths.local.yaml `
  --datasets configs/datasets.yaml `
  --preparation-config configs/data_preparation.yaml `
  --split-config configs/splits.yaml
```

## 4. Preflight formal experiments

```powershell
python scripts/run_formal_classic_main.py --paths configs/paths.local.yaml --assets configs/assets.local.yaml --preflight-only
python scripts/run_formal_gcn.py --paths configs/paths.local.yaml --assets configs/assets.local.yaml --preflight-only
python scripts/run_formal_bert.py --paths configs/paths.local.yaml --assets configs/assets.local.yaml --preflight-only
```

## 5. Run formal experiments

All shards may be run, or selected shard numbers may be supplied after `--shards`.

```powershell
python scripts/run_formal_classic_main.py --paths configs/paths.local.yaml --assets configs/assets.local.yaml
python scripts/run_formal_gcn.py --paths configs/paths.local.yaml --assets configs/assets.local.yaml
python scripts/run_formal_bert.py --paths configs/paths.local.yaml --assets configs/assets.local.yaml
```

Examples of partial execution:

```powershell
python scripts/run_formal_classic_main.py --paths configs/paths.local.yaml --assets configs/assets.local.yaml --shards 1
python scripts/run_formal_bert.py --paths configs/paths.local.yaml --assets configs/assets.local.yaml --shards 1 2 3 4
```

Completed immutable shards are skipped; an `.inprogress` directory is treated as resumable.

## 6. Aggregate the main experiment

```powershell
python scripts/summarize_formal_classic_main.py --paths configs/paths.local.yaml
python scripts/summarize_formal_gcn.py --paths configs/paths.local.yaml
python scripts/summarize_formal_bert.py --paths configs/paths.local.yaml
python scripts/summarize_main_results.py --paths configs/paths.local.yaml
```

The final combined directory is `04_results/derived/main_fixed1000_v1` unless an alternative output name is explicitly provided.

## 7. Feature-count sensitivity

The sensitivity orchestrator adds only the 500-, 1,500-, and 2,000-word runs. It reuses the no-FS and 1,000-word results from the formal main experiment.

```powershell
python scripts/run_feature_count_sensitivity.py `
  --paths configs/paths.local.yaml `
  --assets configs/assets.local.yaml `
  --preflight-only

python scripts/run_feature_count_sensitivity.py `
  --paths configs/paths.local.yaml `
  --assets configs/assets.local.yaml

python scripts/summarize_feature_count_sensitivity.py `
  --paths configs/paths.local.yaml
```

The validated summary is written to `04_results/derived/feature_count_v1`.
