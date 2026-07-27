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
distinct positive-response source users >= 1
```

The first two structural gates are label-blind. The final compatibility gate
uses only source outcomes, never validation/test outcomes, and is required
because the existing BPR and ItemKNN scorers construct their item vocabularies
from positive source interactions only. It removes three of the 4,098
exposure-warm items and does not change the current 200/500 cohort feasibility.
Without it, those strict scorers can fail even though an item has many source
exposures.

Together the gates prevent a content-aware LLM advantage from being
misreported as policy-shift generalization when the collaborative methods
actually face item cold start.

Under the English-semantic item filter, the source contains 820,534 raw
standard tab-1 rows, 25,114 users, and 7,158 items. The common-warm gate retains
4,098 items. The distinct source-history table contains 815,034 user-item
pairs, and 13,528 users have at least 20 such rows.

An unrestricted semantic target pool is a separately named secondary
`full-pool system robustness` result. It preserves natural target items and
uses explicitly configured cold fallbacks for classical methods. Its manifest
must report both exposure-cold and positive-graph-cold row fractions.

The predeclared source-item exposure-degree strata for that secondary result
are:

```text
0
1--4
5--19
20--99
100+
```

Positive-source-user degree is reported alongside exposure degree because
BPR/ItemKNN support follows the positive graph. The full-pool result must not
be mixed with the primary common-warm behavioral-transport result. In the
current feasibility sample, approximately 4.6% of random-target rows are
absent from the standard exposure graph and approximately 12% are absent from
the positive-only graph; a single generic "cold item" number would hide this
difference.

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

## Primary LLM task

The headline LLM condition uses the existing
`LLMTargetResponseListwiseRankingScorer`, not the preference prompt and not the
probability JSON scorer. One API request contains:

- the user's 20 standard-source history videos and their observed responses;
- the ten target candidates;
- translated caption, cover text, and category fields;
- the explicit response target: positive interaction after exposure.

The unchanged repository prompt asks the model to return all candidate labels
from most likely to least likely to produce that response:

```text
C7
C2
C10
C1
C5
C4
C8
C3
C9
C6
```

Every label must occur exactly once. The returned order is converted to ordinal
ranking scores. It is one request per user, target policy, split, and candidate
seed.

The probability scorer is not part of the headline ranking result. It may be
run later as a separately labelled calibration experiment; listwise ranks must
never be treated as click probabilities.

## Metrics and zero-positive groups

Primary ranking metrics:

- NDCG@5;
- HR@1.

Secondary ranking metrics:

- NDCG@10;
- HR@5.

HR@10 is not reported because every retained group contains exactly ten
candidates; conditional on at least one positive it would always equal one
regardless of the ranking.

Following the common KuaiRand convention, groups with zero positives have
undefined IDCG and are excluded only when computing ranking metrics. They
remain in the saved task and in pointwise metrics. No cohort or candidate
selection conditions on the labels.

Optional probability metrics, reported only for separately run
probability-producing methods:

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
evaluation. Every row remains usable for log-loss and Brier score only in the
optional probability experiment.

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

Correct-history and no-history LLM conditions are run on all 500 test users,
both target policies, and all three candidate seeds. Controls are not reduced
to a 200-user subset. Validation users are used for protocol/model selection,
not substituted for the fixed test cohort.

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
