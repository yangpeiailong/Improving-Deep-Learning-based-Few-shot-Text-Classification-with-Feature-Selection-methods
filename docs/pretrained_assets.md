# Pretrained assets

The experiments use local copies of two external pretrained resources. They are not redistributed in this repository.

## FastText

- Source: English Common Crawl FastText vectors, `cc.en.300.bin`.
- Experimental dimension: 100.
- The asset utility can create a reduced 100-dimensional local model while retaining an auditable manifest.

Example local configuration:

```yaml
fasttext:
  source_path: "D:/fasttext/cc.en.300.bin"
  reduced_path: "D:/fasttext/cc.en.100.bin"
  expected_source_dimension: 300
  target_dimension: 100
```

## BERT

- Identity: `google-bert/bert-base-uncased`.
- Loading mode: local/offline.
- The local directory must include the model configuration, tokenizer vocabulary, and weights.

Example:

```yaml
bert:
  path: "D:/bert-base-uncased"
  identity: "google-bert/bert-base-uncased"
  local_files_only: true
```

## Audit and optional FastText reduction

```powershell
python scripts/manage_pretrained_assets.py `
  --paths configs/paths.local.yaml `
  --assets configs/assets.local.yaml

python scripts/manage_pretrained_assets.py `
  --paths configs/paths.local.yaml `
  --assets configs/assets.local.yaml `
  --reduce-fasttext
```

The generated manifest records identities, paths, dimensions, and SHA-256 hashes. Machine-specific asset paths belong only in `configs/assets.local.yaml`, which is ignored by Git.
