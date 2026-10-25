# Fork maintenance

This fork tracks `sisinflab/warprec` and keeps small, tested correctness and
reproducibility fixes for downstream use. Experiment configurations, datasets,
checkpoints and analysis results belong in the consuming project.

## Starting point

The fork starts at upstream **2.2.8**, commit
`be5d02b9e6a61a55cbcf0093d9a016df52c50cd8` (23 October 2026).
The older downstream checkout was **1.15.2**, commit
`7ea2511538d5fe19fd871db34011dcbe6a437006`: it had no local commits and was
257 commits behind. Starting from current upstream retains its intervening
loss, metric, ranking, data and checkpoint fixes. A 1.x installation is a
separate historical runtime, not an interchangeable dependency.

## Fixes already inherited from upstream

- [Paired BPR loss](https://github.com/sisinflab/warprec/commit/a85bb126):
  one-dimensional positive and negative scores are compared row by row;
  multiple negatives are compared within their own row. See `tests/test_losses.py`.
- [Undefined fairness metrics](https://github.com/sisinflab/warprec/commit/bbd343d1):
  PopREO returns a metric dictionary containing NaN when a group has no relevant
  items or both group hit rates are zero. Its numerator counts relevant
  recommendations, not every recommendation. That relevant-hit filtering was
  already present in the 1.15.2 checkout; it is not a new change of this fork.
  The golden beyond-accuracy tests check the formula, and this fork adds explicit
  missing-group and no-hit regressions.

## Additional fix

Popularity groups now use a stable descending sort, breaking equal interaction
counts by ascending internal item index. CPU and MPS could previously put
different members of a tied block in the head and tail. This shared fix applies
to ACLT, APLT, PopREO and PopRSP. It preserves the cumulative rule: the head
includes the first item whose prefix strictly exceeds `pop_ratio` of the total.

`tests/test_popularity_partition.py` exercises all four metrics on CPU and,
when available, MPS and CUDA. The fixture has a tied block spanning the cutoff
and explicit expected memberships. Determinism requires identical item mappings
and interaction counts; stable sorting alone does not harmonize different
datasets, mappings, floating-point scores or ranking policies.

## Verification and adoption

Run the focused checks from the repository root:

```bash
python -m pytest tests/test_losses.py tests/test_metrics_golden.py \
  tests/test_metrics_golden_beyond_accuracy.py tests/test_popularity_partition.py \
  tests/test_ranking_masking.py tests/test_sampled_graded_relevance.py \
  tests/test_tie_break.py tests/test_metric_smoke.py tests/test_checkpoint_roundtrip.py -q
```

Pin a consuming project to the exact fork commit. Before switching an existing
analysis, verify split and item mappings, binary versus graded nDCG, seen-item
masking, tie breaking, aggregation and checkpoint compatibility. WarpRec exposes
binary nDCG as `nDCGRendle2020`; `nDCG` uses graded relevance. Use the evaluation
API or `ProxyRecommender` for existing models/lists instead of retraining just
to adopt the framework. Historical results keep their original runtime and
popularity partitions until an explicit new comparison is requested.
