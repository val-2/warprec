# Partial recommendation lists

A ranking cutoff `k` is a maximum list length. A top-k position whose score is
not finite is unavailable, even if its returned index happens to name a relevant
item. Such positions contribute neither hits nor exposure. The same rule applies
to full-catalogue scores and to sampled candidates, including padding IDs.
If `k` exceeds the score-matrix width, ranking returns only that width. Ideal
gain and rank-position vectors use the available width, while nominal-k
normalizations such as Precision keep the configured cutoff.

Let `m_u` be the number of available recommendations in user `u`'s first `k`
positions, and `R_u` the user's relevant evaluation items. Ground truth is not
deleted merely because the model cannot recommend it.

| Metric | Denominator or accumulation | Empty list, with nonempty ground truth |
| --- | --- | --- |
| Precision@k | Nominal `k` | 0 |
| Recall@k | `|R_u|` | 0 |
| nDCG@k | Original ideal gain through `min(k, |R_u|)` | 0 |
| APLT@k | Actual `m_u` | NaN |
| ARP@k | Actual `m_u` | NaN |
| ACLT@k | Count available long-tail recommendations | 0 |
| NumRetrieved@k | `m_u` | 0 |
| ItemCoverage | Distinct available recommended catalogue IDs | 0 |
| Gini | Available recommendation frequencies over the catalogue | NaN when all lists are empty |
| ShannonEntropy | Available recommendation frequencies | 0 when all lists are empty |
| UserCoverage | Evaluated users with a nonempty list | Does not count the empty list |

For per-user averages, an undefined row contributes to neither the sum nor the
row count. A user with no defined row remains NaN. This also applies when one
user has several evaluation contexts: an empty context cannot poison a defined
APLT or ARP in another context. Users with no relevant evaluation item retain
the framework's existing exclusion from user-averaged ranking metrics.

PopREO and REO count only available relevant recommendations. Their denominators
remain the relevant user-item occurrences of each group, including misses;
missing group rates and two zero hit rates retain the existing NaN convention.
PopRSP and RSP divide available recommendations by finite-score opportunities.
Item-group exposure and disparity counters exclude unavailable positions, and
mean top-k score uses the available list length.

Rank-discounted novelty metrics retain their nominal-k discount normalization;
unavailable positions contribute zero gain. Regression metrics on predicted
ratings require rating predictions and should not be interpreted as evaluation
of a recommendation list that omits those predictions.

The evaluator and direct metric fallbacks use the same validity rule.
`TopKMetric.recommendations()` supplies global item indices together with their
validity mask, also under sampled evaluation. An invalid slot has a safe lookup
index only to avoid indexing past the catalogue; consumers must apply its mask.

Complete lists preserve the previous denominators and formulas. No score shift,
extra recommended item or fabricated hit is used to complete a short list.
