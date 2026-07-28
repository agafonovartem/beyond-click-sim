# KuaiRand standard-to-random protocol

## Status

This is the locked specification implemented by the current KuaiRand task
builder and runner. Its cohort sizes and construction parameters were selected
from label-blind feasibility checks and aggregate label-prevalence checks
before running either GPT-5.5 or Pop under this construction.

It is an exploratory, model-blind protocol, not an externally preregistered
confirmatory experiment.

Generated task manifests identify this protocol as
`kuairand-standard-to-random-v2`. Paid inference is only started after task
construction, invariant checks, prompt materialization, and cost preflight.

## Research question

Can a pretrained language model use behavior collected under the ordinary
recommendation regime to rank future randomized exposures of the same users
and source-warm videos better than an item-popularity baseline learned from
that same ordinary source log?

Only one source regime is used:

| Source | Target | Interpretation |
| --- | --- | --- |
| pre-intervention standard | future standard | temporal reference |
| pre-intervention standard | future random | temporal + exposure-regime OOD |

Random exposures are never used as history or training data. This is not a
standard/random history square.

The main comparison is GPT-5.5 versus Pop within the random-target column.
The standard-target column is a temporal reference. Raw metric levels across
the two columns are not a causal estimate of the intervention effect because
the natural candidate and response distributions differ.

## Dataset and response

- Dataset: KuaiRand-Pure.
- Scenario used for source and targets: `tab=1`.
- Target column: released binary `target_click`.
- Paper-facing name: positive interaction after exposure.

The broader name is used because the released signal can correspond to click
or valid play depending on the interface. Long-view, like, follow, comment,
forward, play time, and missing catalog interactions are not targets in this
experiment.

Every candidate is a real logged exposure. `target_click=0` is an observed
non-interaction following an exposure; it is not a sampled missing item.

## English semantic metadata

GPT-5.5 sees the GPT-5.5 English translation of the released:

- Chinese caption;
- cover text;
- hierarchical category path.

Caption and cover text remain separate fields. An absent field remains absent;
one field is not silently substituted for another.

The translation contains 7,506 semantically eligible items and is stored
outside Git:

```text
data/processed/kuairand/semantic_translation_gpt55/
├── kuairand_items_en.csv
├── kuairand_category_glossary_en.csv
└── manifest.json
```

The task loader must verify the committed hashes from
`TRANSLATION_GPT55_20260727.md`, require unique item IDs, and perform a
one-to-one join. The translation is fixed input data; it is not regenerated
during experiment construction.

## Temporal split

| Part | Dates |
| --- | --- |
| Source | 2022-04-08 through 2022-04-21 |
| Buffer | 2022-04-22 |
| Validation | 2022-04-23 through 2022-04-30 |
| Test | 2022-05-01 through 2022-05-08 |

Source contains only standard `tab=1` exposures. April 22 is excluded because
the released random-intervention stage began that day and intraday ordering
relative to the first insertions is not established.

The buffer is excluded from model training and target evaluation, but its
rows participate in the global pair-novelty lookup.

Validation and test each contain a standard target table and a random target
table. Neither is added to model training.

## Source data

After the English-semantic item filter, the pre-intervention standard source
contains:

- 820,534 exposure rows;
- 25,114 users;
- 7,158 items;
- 815,034 distinct user-item pairs.

There are 13,528 users with at least 20 distinct source items.

Pop uses the complete source exposure table, including repeated exposures.
GPT receives only the bounded user history defined below.

## GPT history

For each evaluated user:

1. take all standard `tab=1` source rows from April 8--21;
2. order rows by `timestamp_ms`, with `interaction_id` as the deterministic
   tie-break;
3. for each user-item pair retain its latest source exposure;
4. take the latest 20 resulting distinct items;
5. return those 20 rows in chronological order.

Each visible history row contains:

- English caption, when non-empty;
- English cover text, when non-empty;
- English category path;
- observed binary positive-interaction response.

GPT does not see:

- raw user or item IDs;
- interaction IDs;
- dates or timestamps;
- `tab`;
- exposure-regime names;
- source popularity or CTR;
- validation/test labels.

Exactly 20 rows are visible. Extra source interactions of an evaluated user
remain available to Pop but are not placed in the language-model context.

## Primary common-warm items

Every target item must satisfy both requirements using only the
pre-intervention standard source:

```text
source exposure rows >= 20
distinct source users >= 5
```

These two structural, response-label-blind conditions retain exactly **4,098
primary items**. Source positive-response count is not an eligibility gate:
Pop is well-defined as zero for an exposure-warm item that received no positive
source response.

This common-warm restriction is part of the primary task. A separate
unrestricted cold-item robustness experiment is not part of the initial run.

## Exact pair novelty

A target row is eligible only when it is the first released exposure of its
exact user-item pair.

The earliest-row lookup uses all released interactions:

- every `tab`;
- standard and random regimes;
- source, April 22, validation, and any earlier test rows.

Rows are ordered by `timestamp_ms`, with `interaction_id` as deterministic
tie-break. A candidate remains only if its `interaction_id` equals the global
earliest row for that user-item pair.

Consequently:

- validation cannot repeat source or April 22 pairs;
- test cannot repeat source, April 22, or validation pairs;
- a target cannot repeat an earlier row in its own window;
- if the same pair appears under both regimes, only the globally earliest row
  can survive.

This isolates first-exposure response prediction from familiarity and repeated
exposure effects. Pair novelty is an edge constraint and does not replace item
warmth.

## Natural candidate groups

For each user, split, target regime, and candidate seed:

1. take semantic, common-warm, globally pair-novel natural exposures from the
   fixed target window;
2. compute a stable seeded hash over split, target regime, user, item, and
   interaction identity without including `target_click`;
3. sort by that hash;
4. retain the first \(m=10\) rows.

The candidate seeds are exactly:

```text
0, 1, 2
```

They resample candidate exposures for the same fixed users. They do not move
the temporal windows and are not independent datasets.

No target labels are balanced. No negatives are generated. A group may
naturally contain zero, one, several, or ten positive responses. The ten rows
are separate logged exposures, not one simultaneous production slate.

## Cohorts

User eligibility is label-blind:

- at least 20 distinct standard source items;
- at least ten eligible future standard rows;
- at least ten eligible future random rows.

After semantic filtering, common-warm filtering, and exact pair novelty:

- 402 validation users are eligible;
- 908 test users are eligible.

Cohort seed is `0`.

The deterministic allocation is:

1. order validation-eligible users by the cohort hash and take the first 200;
2. remove those users from the test-eligible set;
3. order the remaining test-eligible users by the same cohort hash and take
   the first 500.

Validation and test therefore contain no shared users. Within a split, the
same user cohort is used for standard and random targets and for every method.

For each candidate seed:

- validation contains 2,000 standard and 2,000 random rows;
- test contains 5,000 standard and 5,000 random rows.

Across all three seeds, the test task contains 30,000 target-row occurrences.

## Method 1: GPT-5.5

- Model: `gpt-5.5`.
- Client: `openai_vk_proxy`.
- Task: existing target-response listwise ranking scorer.
- Generation count: one.
- One API request scores all ten candidates in a group.

The existing target-response prompt is reused unchanged. It receives:

- the twenty source-history videos and their observed responses;
- ten target candidates;
- English caption, cover text, and category fields;
- the explicit target: positive interaction after exposure.

The response must contain every opaque candidate label exactly once, ordered
from most likely to least likely:

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

Item names, raw IDs, explanations, probabilities, and rank numbers are not
requested. The order is converted to unique ordinal ranking scores.

Before the full run, a ten-candidate smoke test must verify that the configured
completion-token cap cannot truncate a valid response. The exact request
configuration and prompt-template hash are frozen in the run manifest.

There is no artificial LLM seed: candidate seeds change the target rows, not
the deterministic model request.

The first suite has two GPT conditions:

- `observed_history`: the twenty source rows specified above;
- `no_history`: the same candidates, candidate order, metadata, target wording,
  model, and request configuration, but no user-history rows.

No-history is a within-model ablation/control that measures candidate/content
prior. It is not presented as another recommender baseline. Candidate-frame
hashes must be identical between the two GPT conditions.

## Method 2: Pop

Pop is the exact positive-response count of an item in the complete standard
source log:

\[
\operatorname{Pop}(i)
=
\sum_{r \in D_{\mathrm{source}}}
\mathbf{1}[i_r=i]\mathbf{1}[y_r=1].
\]

Repeated source exposures contribute separately. There is no normalization,
smoothing, user conditioning, metadata, or target-period information.

Every primary candidate is exposure-warm. An item with no positive source
response receives the exact Pop score zero; this is the defined statistic, not
a cold-item fallback.

Pop and GPT rank byte-identical candidate groups in the same order. Pop is
fitted once because its source table is identical across candidate seeds and
target regimes.

## Metrics

Primary:

- NDCG@5;
- HR@1.

Secondary:

- NDCG@10;
- HR@5.

HR@10 is not reported: for a ten-item group containing any positive response,
it is always one regardless of ranking.

Ranking metrics are computed only for mixed-label groups:

```text
1 <= number of positive responses <= 9
```

All-zero groups have no relevant item. All-positive groups give every ordering
the same perfect ranking metric. Both remain saved in the task and prediction
artifacts but are excluded from ranking aggregation.

The manifest and metric files must report, for every split, target regime, and
seed:

- total groups;
- all-zero groups;
- all-positive groups;
- mixed-label groups;
- exact denominator of every reported metric.

Score ties use the repository's average-tie convention. This matters for Pop;
GPT's parsed complete ranking has unique ordinal scores.

## Statistical comparison

The headline random-target contrasts are:

- observed-history GPT minus Pop;
- observed-history GPT minus no-history GPT.

For each metric:

1. align the compared conditions on the exact
   `seed × user × candidate_group`;
2. keep only groups that are mixed-label and successfully scored by both
   compared conditions;
3. compute the paired condition difference;
4. average available seed differences within each user;
5. bootstrap users, not `user × seed` rows, for the confidence interval.

The report must show:

- number of users;
- number of eligible seed-user pairs;
- minimum, mean, and maximum eligible seeds per user;
- mean paired difference;
- 95% user-bootstrap confidence interval.

The two random-target contrasts evaluated with NDCG@5 and HR@1 form a
four-test headline family and receive Holm correction if null-hypothesis
\(p\)-values are reported.

The same comparisons on standard targets are temporal-reference diagnostics,
not additional headline claims. Candidate seeds are correlated resamples and
must never be presented as three independent datasets.

## Request and artifact volume

The full test run requires:

```text
500 users × 2 target regimes × 3 candidate seeds × 2 GPT conditions
= 6,000 GPT requests
```

Each request contains ten candidates, yielding 60,000 GPT prediction rows
across observed-history and no-history. Pop evaluates the 30,000 unique target
rows without API calls.

A complete paid prompt/cost preflight must materialize all 6,000 requests
before the client is constructed. A one-user/one-seed format smoke makes four
requests: observed-history and no-history for standard and random targets.

Successful requests are saved in content-addressed atomic checkpoints. Reusing
the exact run directory must load compatible checkpoints and make no duplicate
calls. Task, prompt, model, metadata, candidate order, or request changes must
produce cache misses rather than silently reuse predictions.

The current checkpoint identity additionally binds the configured client
provider and explicit scorer/parser format versions. Checkpoints from the
already-paid legacy run remain readable only through an exact reconstruction
of their old content key; that key already binds the task, cell, user, ordered
items, complete prompt, model, and request configuration. A compatible legacy
hit is copied atomically under the current identity key and makes no API call.
The fallback is disabled for every other provider/scorer/parser combination.

A run with any failed GPT groups is not `complete`. It is
`complete_with_errors` when at least one group succeeds and `failed` when no
group succeeds. Combined GPT-versus-Pop analysis is materialized only when
both component runs have status `complete`.

## Required task and run invariants

A real-data audit must pass before any paid request:

- source dates and policies exactly match this document;
- April 22 contributes no source or target rows;
- random source rows equal zero;
- translation hashes and one-to-one item join match;
- every visible GPT history contains exactly 20 distinct items;
- every target group contains exactly ten distinct item rows;
- candidate selection and cohort membership are unchanged after flipping
  validation/test labels;
- all target items pass the response-label-blind 20-exposure/5-user source
  gate;
- every target row is the global earliest released pair exposure;
- validation and test cohort overlap is zero;
- candidate hashes reproduce exactly across builds;
- observed-history GPT, no-history GPT, and Pop target-frame hashes match;
- no hidden audit, ID, policy, timestamp, popularity, or label column reaches
  the GPT scorer.

Every run manifest records:

- dataset and canonical-source hashes;
- translation artifact hashes;
- temporal split and buffer;
- target definition;
- history and candidate construction;
- cohort and candidate seeds;
- user, item, group, and row counts;
- common-warm item-ID hash;
- source/history/target frame hashes;
- label-group denominators;
- model, prompt, parser, and request configuration;
- visible and hidden metadata;
- Pop definition;
- Git commit and dirty status;
- checkpoint, token, error, and cost accounting;
- known caveats.

The full paid run should start from a clean Git state and an explicit immutable
run directory.

## Valid interpretation

A positive GPT-minus-Pop difference on common-warm random targets means that,
under this exact natural ranking construction, GPT's pretrained semantic prior
plus twenty standard-source responses rank future randomized outcomes better
than source positive-count popularity.

Observed-history GPT beating no-history GPT provides the corresponding evidence
that the visible source history adds useful ranking signal beyond the
candidate/content prior. If both GPT conditions beat Pop but observed history
does not beat no history, the result supports a semantic candidate prior, not
behavioral transport.

It does not by itself prove:

- a pure causal effect of the recommendation intervention;
- calibrated click-probability prediction;
- off-policy policy-value estimation;
- trajectory simulation;
- superiority over all collaborative or feature-based recommenders.

The unavoidable temporal shift must be stated explicitly. The standard target
is useful as a temporal reference, but raw standard/random metric differences
remain confounded by different natural exposure distributions.

Agent4Rec, collaborative recommenders, probability calibration, long-view
prediction, permuted-feedback controls, and unrestricted cold-item robustness
are possible later experiments. They are not part of this first locked suite.
