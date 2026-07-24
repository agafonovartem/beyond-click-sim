# KuaiRand natural-exposure policy-shift protocol

## Status

This protocol replaces the exploratory forced-c3 construction for future
KuaiRand policy-shift runs. Its parameters were fixed after a local feasibility
and aggregate label-distribution audit, but before any model outputs or paid
API responses were inspected. It is therefore a stronger exploratory protocol,
not a confirmatory preregistered experiment.

The exploratory c3 artifacts remain useful for diagnosing implementation and
candidate-support problems, but they are not the main policy-shift result.

## Research question

Given feedback history collected under one exposure policy, can a response
model predict later outcomes under another exposure policy for the same users?
Does it transport this signal better than classical and text-only baselines?

The full square is:

| Visible source history | Standard target | Random target |
| --- | --- | --- |
| Standard exposures | matched | transferred |
| Random exposures | transferred | matched |

Within one target-policy column, matched and transferred methods receive
byte-identical target rows. Raw metric levels must not be compared across the
standard-target and random-target columns because the natural exposure and
click distributions differ.

## Fixed construction

- Dataset: KuaiRand-Pure, tab 1 targets, released captions and category paths.
- Source window: 2022-04-22 through 2022-04-30.
- Validation window: 2022-05-01 through 2022-05-03.
- Test window: 2022-05-04 through 2022-05-08.
- Visible history length: exactly \(H=5\) rows from each source policy.
- History selection: five latest unique user-item-policy source pairs. Labels
  do not affect row selection.
- Candidate-group size: exactly \(m=5\) natural exposures from each target
  policy.
- Candidate seeds: 0, 1, and 2.
- Candidate selection: stable seeded hash over eligible exposure identity.
  Click labels do not affect the selected users or candidate rows.
- Pair novelty: every target user-item pair must be absent from every earlier
  released tab under both policies.
- Item warmth: every target item must have at least eight distinct users in
  each full source-policy log. After evaluation users are restricted to their
  visible \(H=5\) rows, the actual collaborative training graph must still
  contain at least five distinct users for every target item under each policy.
- Source clicks are not used for the item-warmth filter. Positive source degree
  is retained only for predeclared support-stratified analysis.
- Users are selected only by having at least five source and five target rows
  under both policies. Users are not required to have a click or non-click in
  either history or target regime.

The target candidates are separately logged exposures, not a simultaneous
slate. This is one-step response prediction under natural logged exposure, not
off-policy policy-value estimation.

## Why H=5, m=5, full-source support 8, and training support 5

The support thresholds remove the previous degree-1 near-cold artifact without
conditioning on clicks. Full-source support 5 was insufficient: after hiding
evaluation users' extra rows, some candidate items fell to training degree
2–4. Full-source support 8 leaves 2,376 common source-warm items and guarantees
an observed training degree of at least 5 in the materialized task.

At this threshold, the label-blind cohort grid is:

| Split | H | m=3 | m=5 | m=8 | m=10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 3 | 1,536 | 515 | 130 | 48 |
| validation | 5 | 1,048 | **369** | 94 | 31 |
| validation | 8 | 545 | 220 | 62 | 20 |
| validation | 10 | 353 | 144 | 42 | 17 |
| test | 3 | 2,206 | 798 | 229 | 114 |
| test | 5 | 1,373 | **540** | 179 | 91 |
| test | 8 | 708 | 333 | 125 | 63 |
| test | 10 | 454 | 221 | 83 | 43 |

\(H=5\) provides multiple feedback observations per policy while preserving
substantially more users than \(H=10\). \(m=5\) provides many more evaluable
users than \(m=8\). Increasing \(m\) raises the fraction of random-policy groups
with a click but loses more total ranking-eligible groups than it gains.

## Label distributions after fixing the task

These labels describe the already label-blind selected rows; they did not
select users or candidates.

| Split | Policy | History CTR | Target CTR range over seeds | Mixed-label groups | All-zero groups |
| --- | --- | ---: | ---: | ---: | ---: |
| validation | standard | .5209 | .4900–.5003 | .8076–.8428 | .0759–.0976 |
| validation | random | .1680 | .1864–.1973 | .5854–.6179 | .3821–.4146 |
| test | standard | .4656 | .4478–.4511 | .7796–.8167 | .1204–.1481 |
| test | random | .1630 | .2011–.2159 | .5704–.6000 | .3963–.4259 |

The standard/random CTR difference is part of the natural policy shift. It is
not removed from the primary estimand. The experiment includes a fixed
Beta(1,1) Laplace-smoothed five-label user CTR so that item-specific transport
can be separated from merely learning a user's average click rate.

## Metrics and controls

Primary probability metrics, computed on every selected natural exposure:

- user-macro log-loss;
- user-macro Brier score.

Secondary probability diagnostics:

- row-level log-loss and Brier score;
- row-level ROC AUC and average precision;
- calibration intercept and slope;
- predicted click-rate mean versus observed CTR.

Secondary ranking diagnostics:

- NDCG@5;
- Hit@1.

Every constructed group, including all-zero and all-one groups, remains in the
saved task and contributes to the probability metrics. Ranking metrics are
defined only for groups containing both labels, so the report must show their
eligible coverage. Listwise-only methods remain conditional ranking baselines
and must not be presented as calibrated response predictors.

Seeds 0, 1, and 2 are correlated candidate resamples for the same users, not
independent datasets. Mixed-label eligibility can vary across seeds. Every
model comparison is paired on the same eligible `seed × user` target group;
available eligible seeds are then averaged inside the user before user-level
bootstrap and sign-flip inference. The analysis reports the minimum, mean, and
maximum number of eligible seeds per user. It does not condition the cohort on
having mixed labels under all three resamples.

Implemented history conditions:

- correct observed history;
- no history for semantic models;
- an association-breaking control that preserves the item rows and per-user
  click multiset but changes the item-to-feedback assignment.

Implemented scalar calibration companion:

- raw user source-policy CTR for transparency;
- fixed Beta(1,1) Laplace-smoothed CTR from the same five visible labels as
  the main scalar-history baseline.

This distinguishes a transferred user-level response propensity from
item-specific preference transport. If the LLM only learns that a user clicks
often, its correct and permuted-history results can remain similar and it need
not improve over smoothed user CTR. If it grounds feedback in the corresponding
video semantics, correct history should beat the permutation that preserves
the same items, labels, history length, and per-user click count but reassigns
which video received which label.

Planned positive-source support strata are descriptive and fixed before model
scoring:

- zero positive users;
- one positive user;
- two to four positive users;
- five or more positive users.

The primary task does not filter on these strata. Every target row now stores
its policy-specific exposure/positive degrees and the derived support stratum
as analysis-only columns hidden from scorers. Aggregating the final paired
comparisons by stratum remains to be added after the main unstratified
baselines. It will test whether any LLM or text advantage is concentrated in
weak collaborative-support regions.

The released source logs contain substantially different numbers of standard
and random exposures. The primary experiment keeps those native regimes
intact. Therefore a matched-versus-transferred gap is a system-robustness
contrast across released exposure regimes, not a source-size-controlled causal
effect of the policy flag. A balanced-source sensitivity run is required
before attributing the whole gap specifically to policy rather than policy
plus source evidence volume.

## Reproduce the audit

```bash
uv run python -m runners.ood.kuairand_policy_shift.natural_exposure_audit \
  --output-dir outputs/ood/kuairand_policy_shift/20260724_natural_exposure_feasibility \
  --history-sizes 3,5,8,10 \
  --candidate-sizes 3,5,8,10 \
  --exposure-supports 1,2,5,8,10,20 \
  --candidate-seeds 0,1,2
```

The command writes `feasibility_audit.json` and `feasibility_report.md`. It
does not construct an API client, fit a model, or make network calls.
