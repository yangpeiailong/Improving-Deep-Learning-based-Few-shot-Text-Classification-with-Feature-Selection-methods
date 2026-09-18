# Released results

## Main experiment

`main/` contains the validated fixed-1,000-word experiment:

- 3,520 fold-level evaluations;
- 16 datasets;
- 11 models in three model families;
- four conditions (`none`, `df`, `ig`, and `dfs`);
- five outer folds.

`main_completeness_report.json` records the expected and observed factorial coverage and the SHA-256 hashes of the three family-level source tables. `main_fold_results.csv` is the complete cross-family fold-level table. The remaining files provide aggregate results, paired effects, selector/family summaries, significance tests, and runtime summaries.

## Feature-count sensitivity

`feature_count/` contains the complete aggregate analysis for four datasets, eight evaluated models, three selectors, and 500/1,000/1,500/2,000 retained words, together with the no-FS baselines. Its completeness report validates 2,080 fold-level source rows.

## Supplementary material

`supplementary/Supplementary_Materials.xlsx` contains:

- Supplementary Table S1: complete main experimental results;
- Supplementary Table S2: feature-count sensitivity results;
- Supplementary Table S3: computational-time results.

Accuracy and F1 values are reported as proportions. Timing variables are in seconds.
