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

## Follow-up audit: training and evaluation paths

The follow-up audit adds fixes on the paths used by BPR and LightGCN:

- Training negatives: canonicalize non-canonical CSR inputs on a copy before
  binary-search membership checks, so unsorted rows cannot return seen items.
  Reject draws with no unseen item of positive probability with `DataError`
  rather than entering an endless rejection loop. Popularity sampling can
  exhaust its support even when unseen zero-popularity items remain. The
  distribution and random stream of feasible, canonical inputs are unchanged.
- Graph inference: preserve the `nn.Module.train()/eval()` return contract,
  invalidate propagated embedding caches after state-dict loads (including
  nested loads) and device/dtype conversion, and match sparse edge-value
  precision to the dense operand. LightGCN now scores and backpropagates in
  float64 instead of crashing or returning an old float32 cache.
- ARP: preserve zero training-interaction counts instead of smoothing them to
  one. ARP performs no division by popularity and needs no such smoothing;
  novelty metrics retain their separate handling.
- Documentation: PopREO's displayed formula now conditions the numerator on
  relevance and sums user-item occurrences, matching the implementation and
  golden tests.

Regressions are in `test_training_negative_edges.py`,
`test_graph_inference_state.py` and the golden beyond-accuracy tests. The audit
also exercises model smoke, training-seed, splitter and evaluation-loader tests
in addition to the checks above. These use synthetic fixtures, not downstream
experiment datasets.

### Remaining evaluation limitations

The evaluator can still count unavailable `-inf` positions when fewer than k
finite candidates remain. A three-item fixture with only one available,
irrelevant item reports `NumRetrieved=1` but `Recall=1`, binary nDCG about
`0.63093` and `ItemCoverage=3`, because a masked relevant item fills a top-k
slot. Do not treat partial/restricted rankings as validated until that path
handles missing slots consistently across accuracy and exposure metrics. The
current audit records the defect without imposing a new partial-list protocol.

`ProxyRecommender` additionally assigns zero to unlisted pairs by documented
design; a listed negative score ranks below every unlisted item. It is not a
lossless importer of arbitrary finite lists. These limitations must be resolved
before downstream migration; shifting scores does not resolve missing-list
semantics. Direct in-place edits to embedding weights during evaluation also
require explicit cache invalidation, for example by calling `eval()` again;
the graph fixes do not intercept arbitrary tensor writes.
