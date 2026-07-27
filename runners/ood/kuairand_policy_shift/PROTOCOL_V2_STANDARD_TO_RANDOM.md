# KuaiRand standard-to-random protocol v2

## Status

This protocol supersedes the earlier forced-c3 and full-history-square
protocols. It was fixed after inspecting dataset availability and label
prevalence, but before running any model under this construction. It is
therefore an exploratory, model-blind protocol rather than a preregistration.

The current `KuaiRandNaturalExposureTaskBuilder` does **not** implement this
protocol yet. Do not launch model scoring through that builder until it is
replaced or extended.

## Research question

Can a response model trained or conditioned on interactions collected under
the standard recommendation regime generalize to future randomized exposures
of the same users and source-warm items?

The main target is future randomized exposure. Future standard exposure is a
temporal reference:

| Source | Target | Interpretation |
| --- | --- | --- |
| pre-intervention standard | future standard | temporal reference |
| pre-intervention standard | future random | temporal + exposure-policy OOD |

This is deliberately not a full standard/random history square. Random
interactions are never used as source history or model training data.

## Dataset and semantic metadata

- Dataset: KuaiRand-Pure.
- Scenario: `tab=1`.
- Response: released binary `target_click`, described as positive interaction
  because its UI-specific semantics can be click or valid play.
- Item text: GPT-5.5 English translation of the released Chinese `caption`,
  `show_cover_text`, and hierarchical category nodes.
- Semantic eligibility: an item must have at least one of caption or cover
  text and must have a non-empty category path.
- Both caption and cover text are retained as separate fields. Missing cover
  text is not fabricated.

The translation contains 7,506 eligible items. It is stored outside Git at:

```text
data/processed/kuairand/semantic_translation_gpt55/
```

## Fixed temporal split

| Part | Dates |
| --- | --- |
| Source | 2022-04-08 through 2022-04-21 |
| Buffer | 2022-04-22 |
| Validation | 2022-04-23 through 2022-04-30 |
| Test | 2022-05-01 through 2022-05-08 |

Source contains only `tab=1, exposure_policy=standard`. April 22 is excluded
because the official random-intervention stage began that day and intraday
ordering relative to the first randomized insertions is not established.

Validation and test each contain two target tables, one standard and one
random. They are never added to model training.

## Source history

For an evaluated user, the LLM sees the latest 20 distinct user-item standard
interactions in the source window, ordered chronologically. Each row contains:

- translated caption, when released;
- translated cover text, when released;
- translated category path;
- observed binary positive-interaction label.

The LLM never sees raw user/item IDs, date, policy name, global CTR, popularity,
or target labels.

Classical models use the complete standard source log. Restricting them to 20
rows would turn the LLM context limit into an artificial handicap for the
baselines. An optional H=20 classical sensitivity may be added later, but it
is not a primary result.

## Common-warm target items

Every primary target item must satisfy, in the pre-intervention standard source
only:

```text
source exposure count >= 20
distinct source users >= 5
```

Both gates are label-blind. They do not use source clicks or target outcomes.
They prevent a content-aware LLM advantage from being misreported as
policy-shift generalization when the collaborative methods actually face item
cold start.

Under the English-semantic item filter, the source contains 820,534 raw
standard tab-1 rows, 25,114 users, and 7,158 items. The common-warm gate retains
4,098 items. The distinct source-history table contains 815,034 user-item
pairs, and 13,528 users have at least 20 such rows.

An unrestricted semantic target pool may be reported separately as
`full-pool system robustness`; it must not be mixed with the primary
behavioral-transport result.

## Pair novelty

For a target window beginning at date \(t\), candidate pair \((u,i)\) is kept
only if it does not appear in any released interaction before \(t\), across
all tabs and both exposure regimes. This prevents a target label from being
recoverable from a repeated earlier exposure.

Pair novelty does not replace the item-warm gate: target users and target items
are warm, while the exact evaluated user-item edge is new.

## Natural candidate groups

For each selected user, target policy, split, and candidate seed:

1. take pair-novel, semantic, common-warm logged exposures from the target
   window;
2. collapse repeated user-item rows in that window to their first exposure;
3. order eligible rows by a stable seeded hash that does not include the
   response label;
4. retain \(m=10\) rows.

No negatives are generated and labels are not balanced. `target_click=0` rows
are real observed non-interactions after exposure, not missing catalog items.
The ten rows are separate logged exposures, not a simultaneous production
slate.

Candidate seeds are exactly `0, 1, 2`. They resample candidate rows for the
same fixed users and split; they do not move temporal windows and are not
independent datasets.

## Cohorts

Eligibility is label-blind:

- at least 20 distinct standard source interactions;
- at least 10 eligible future standard target rows;
- at least 10 eligible future random target rows.

With the fixed semantic and common-warm gates, there are 404 validation-eligible
and 910 test-eligible users. Select 200 validation users and 500 test users by
a stable hash with cohort seed 0, with no user overlap between validation and
test.

For each seed this yields:

- validation: 2,000 standard and 2,000 random target rows;
- test: 5,000 standard and 5,000 random target rows.

The same user cohort is used for the standard and random target columns within
a split. The exact candidate rows differ by policy and candidate seed.

## Metrics and zero-positive groups

Primary ranking metrics:

- NDCG@10;
- HR@10;
- optionally NDCG@5 and HR@1 as stricter secondary cutoffs.

Following the common KuaiRand convention, groups with zero positives have
undefined IDCG and are excluded only when computing ranking metrics. They
remain in the saved task and in pointwise metrics. No cohort or candidate
selection conditions on the labels.

Primary probability metrics for probability-producing methods:

- user-macro log-loss;
- user-macro Brier score.

Secondary diagnostics:

- row-level AUROC and average precision;
- UAUC only for users with both labels;
- calibration intercept/slope and predicted versus observed click rate.

The manifest must report the denominator for every metric, including the number
and fraction of zero-positive, all-positive, and mixed-label groups.

In the fixed feasibility sample, random-target zero-positive rates are
approximately 24--35% across split/seed cells; enough groups remain for ranking
evaluation, and every row remains usable for log-loss and Brier score.

## Initial methods

All trained baselines receive the complete pre-intervention standard source:

- global and user positive-interaction rates;
- ItemCTR with explicit smoothing;
- ItemKNN;
- BPR/MF;
- feature logistic regression or FM/NFM over released user/item features;
- text-affinity using the same translated text as the LLM.

The LLM receives only its 20-row textual user history. `No history` is retained
as a cheap candidate/content-prior control. Agent4Rec is a later, separate
simulator baseline; it does not replace no-history because the two answer
different questions.

The permuted-feedback grounding control is an optional ablation and is omitted
from the first full run.

## Valid claims

The random target tests generalization from pre-intervention standard history
to future randomized exposures without user or item cold start. It still
contains temporal shift, which cannot be removed because randomized exposures
start only on April 22.

The standard target is the temporal reference. Raw standard and random metric
levels must not be interpreted as a pure causal policy effect because their
candidate and response distributions differ. The defensible comparisons are:

- methods on byte-identical target rows within a target-policy column;
- each method's degradation relative to strong source-trained references;
- whether an LLM advantage emerges specifically on common-warm random targets.
