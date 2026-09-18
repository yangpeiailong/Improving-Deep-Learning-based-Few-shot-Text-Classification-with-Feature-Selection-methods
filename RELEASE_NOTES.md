# Release notes

## Revised reproducible release

This release replaces the original repository artifact `dataset_after_feature_selection.zip`.

The original artifact stored feature-filtered texts produced before cross-validation. That representation is unsuitable for the revised evaluation because IG and DFS use class-label statistics, and a vocabulary estimated before splitting may expose held-out information to model development.

The revised release therefore changes the experimental unit from a prefiltered dataset to a fold-scoped pipeline:

- outer folds are frozen before feature selection;
- supervised feature scores use training-fold labels only;
- validation and test texts are transformed with a vocabulary learned from the corresponding training scope;
- validation is used for epoch selection;
- the held-out fold is evaluated only after final refitting;
- run manifests record split and vocabulary hashes;
- final tables report five-fold means and standard deviations;
- paired effects, statistical tests, runtime summaries, and feature-count sensitivity results are released.

The old prefiltered text archive must not be combined with the code or results in this release.
