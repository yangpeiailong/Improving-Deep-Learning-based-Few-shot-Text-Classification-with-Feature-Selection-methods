# Experimental protocol

## Outer evaluation

Every dataset uses five fixed outer folds. In turn, four folds form `outer_train` and the remaining fold forms `test`. Exact text groups are kept together where required by the frozen split policy.

## Development stage

An inner validation partition is created from `outer_train`. For every selector and model configuration:

1. token statistics and the candidate vocabulary are computed from `inner_train` only;
2. DF, IG, or DFS is fitted using `inner_train` (and only its labels for supervised selectors);
3. the selected vocabulary transforms `inner_train` and validation texts;
4. training uses `inner_train` and the validation partition selects the best epoch.

The held-out outer test fold is inaccessible during this stage.

## Final stage

After an epoch is selected, all preprocessing and feature-selection objects are refitted on `outer_train`. The frozen outer-training vocabulary is then applied to both `outer_train` and test texts. The model is trained from a fresh initialization for the selected number of epochs and evaluated once on the held-out fold.

The same boundary applies to graph construction: the inductive word graph and positive-PMI edges are computed from the relevant training scope only. Held-out documents are represented using the frozen training graph.

## No-feature-selection condition

`none` does not use a globally constructed vocabulary. Its vocabulary is still learned from the applicable training scope, so held-out-only tokens cannot enter the model.

## Main and sensitivity experiments

The main experiment fixes the retained vocabulary at 1,000 for DF, IG, and DFS. This value is specified before testing and is not selected by test performance. The feature-count analysis evaluates 500, 1,000, 1,500, and 2,000 retained words on four representative datasets and is reported as sensitivity analysis.

## Audit trail

Each run records hashes for the inner-training, validation, outer-training, and test indices as well as the development and final vocabularies. GCN runs additionally record graph hashes. Completeness reports reject missing, duplicated, extra, or hash-conflicting configurations.
