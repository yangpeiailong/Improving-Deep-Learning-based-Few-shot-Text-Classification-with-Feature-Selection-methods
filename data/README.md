# Dataset layout

Benchmark texts are not redistributed in this repository. Download and use each corpus under its original license, then place the frozen study samples in the following line-aligned format:

```text
<paper_root>/03_data/raw/<dataset_id>/texts.txt
<paper_root>/03_data/raw/<dataset_id>/labels.txt
<paper_root>/03_data/raw/<dataset_id>/metadata.yaml
```

Each line of `texts.txt` must correspond to the label on the same line of `labels.txt`. Raw files are treated as immutable. Prepared records, exclusions, audit reports, and deterministic folds are written beneath `03_data/processed`, `03_data/audit`, and `03_data/splits`.

## Dataset identifiers

| Identifier | Dataset |
|---|---|
| `20ng` | 20 Newsgroups |
| `ag_news` | AG News |
| `amazon_cells` | Amazon Cell Phones sentiment |
| `amazon_review_full` | Amazon Review Full |
| `amazon_review_polarity` | Amazon Review Polarity |
| `bbc_sport` | BBC Sport |
| `dbpedia` | DBpedia ontology classification |
| `farm_ads` | Farm Ads |
| `imdb` | IMDb sentiment |
| `pang_lee` | Pang and Lee sentence polarity |
| `reuters8` | Reuters-8 |
| `sentence` | Sentence classification corpus used in the study |
| `sst5` | Stanford Sentiment Treebank, five classes |
| `wos5736` | Web of Science 5736 |
| `yelp_review_full` | Yelp Review Full |
| `yelp_review_polarity` | Yelp Review Polarity |

The registry and manuscript subset originally used by the study are recorded in `configs/datasets.yaml`. The revised formal experiment evaluates all 16 registered datasets.

## Preparation commands

```powershell
python scripts/audit_data.py `
  --paths configs/paths.local.yaml `
  --datasets configs/datasets.yaml

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

Do not replace these steps with the obsolete prefiltered datasets from the original GitHub repository.
