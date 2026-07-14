# First-Candidate Positional Bias in Grouped LLM Preference Prediction

Status: exploratory empirical finding for later paper integration.
Date recorded: 2026-07-14.
Local row-level `predictions.parquet` artifacts are required to reproduce the analysis.

## Finding

The grouped direct yes/no preference protocol exhibits a large and highly
consistent **first-candidate positional bias**. An item shown as `C1` is more
likely to receive a positive prediction than an item shown in a later slot,
even though the ground-truth positive-label rate is approximately
position-balanced by the seeded candidate shuffle.

The effect appears in both evaluated Qwen models, both datasets, all four
negative ratios, and all three seeds:

- Qwen3.6-27B: the `C1` positive-prediction premium is positive in all 24
  dataset/ratio/seed cells and ranges from 5.1 to 9.9 percentage points across
  individual cells.
- Qwen3-8B: the premium is positive in all 24 cells and ranges from 2.2 to
  12.4 percentage points.

This is not a comparison of probabilities that should sum to one. The relevant
quantities are conditional rates over two disjoint sets of evaluation rows:

```text
P(target = 1 | position = C1)
P(target = 1 | position != C1)

P(prediction = yes | position = C1)
P(prediction = yes | position != C1)
```

For example, `P(prediction=yes | C1)=0.531` and
`P(prediction=yes | position!=C1)=0.437` mean that 53.1% of first-slot rows and
43.7% of later-slot rows received `yes`. These are rates for different row
subsets and are not complementary probabilities.

## Protocol Scope

The audit covers the direct grouped preference-prediction runs:

- datasets: MovieLens 1M and Steam canonical `v1`;
- targets: rating at least 4/5 and total playtime at least 120 minutes;
- negative ratios: `m in {1, 2, 3, 9}`;
- seeds: `0, 1, 2`;
- candidate groups: observed preference labels only, at most 10 items;
- prompt output: one hard `Ck: yes/no` decision per candidate;
- Qwen3.6-27B: 428,946/428,946 parsed rows, zero failed groups;
- Qwen3-8B: complete MovieLens coverage and partial Steam coverage because of
  previously documented group-level format failures.

The position is reconstructed from the preserved row order within each
`candidate_group`:

```python
predictions["position"] = (
    predictions.groupby("candidate_group", sort=False).cumcount() + 1
)
```

This row order is the same order used to assign `C1`, `C2`, ..., in the prompt.

## Main Evidence

The table pools rows across the three seeds. `Target delta` is the difference
between the true positive rate at `C1` and the true positive rate in all later
positions. The model columns report the analogous difference in predicted
`yes` rate. All values are percentage points.

| Dataset | Ratio | Target delta | Qwen3.6-27B `C1` premium | Qwen3-8B `C1` premium |
|---|---:|---:|---:|---:|
| MovieLens 1M | 1 | +1.20 | +7.29 | +11.07 |
| MovieLens 1M | 2 | +1.65 | +6.87 | +10.35 |
| MovieLens 1M | 3 | +1.23 | +7.15 | +9.79 |
| MovieLens 1M | 9 | +0.64 | +6.00 | +7.54 |
| Steam | 1 | +0.16 | +9.38 | +6.73 |
| Steam | 2 | -0.34 | +8.31 | +5.94 |
| Steam | 3 | -0.40 | +8.28 | +5.41 |
| Steam | 9 | +0.17 | +6.25 | +2.61 |

Representative Steam `m=1` rates for Qwen3.6-27B are:

```text
P(target = 1 | C1)              = 0.501
P(target = 1 | position != C1)  = 0.500

P(prediction = yes | C1)             = 0.531
P(prediction = yes | position != C1) = 0.437
```

Thus the observed target imbalance is only +0.16 percentage points, whereas
the prediction imbalance is +9.38 points.

The result is not merely caused by small residual differences in the label mix.
For Qwen3.6-27B, conditioning on the ground-truth class still produces a large
first-slot premium across every dataset/ratio slice:

- among actual negatives, the `C1` false-positive-rate premium ranges from
  +6.0 to +9.0 percentage points;
- among actual positives, the `C1` true-positive-rate premium ranges from
  +5.0 to +10.1 percentage points.

The complete per-position profile shows a first-position spike rather than a
simple monotonic decline over the list. For example, in Steam `m=1`, the pooled
Qwen3.6-27B `yes` rates are 0.531 at `C1`, 0.405 at `C2`, and approximately
0.437-0.454 over `C3`-`C10`. The most precise description is therefore
**first-candidate premium**, not a generic assumption that every later position
is progressively penalized.

## Related Co-Candidate Dependence

The preference sampler can reuse an observed negative item in multiple
candidate groups for the same user. These repeated pairs provide an accidental
consistency check. Qwen3.6-27B produced both `yes` and `no` for 16.6% of repeated
MovieLens user-item pairs and 15.6% of repeated Steam pairs.

A concrete MovieLens `m=1, seed=0` example is user `2911` and
`American Pie (1999)`. The real preference label is `0` in both occurrences:

| Candidate group | Position | Qwen3.6-27B prediction |
|---|---:|---|
| `candidate:user:2911:chunk:0` | `C6` | `no` |
| `candidate:user:2911:chunk:1` | `C2` | `yes` |

There are also different answers when the repeated item occupies the same
position. For user `4937`, `Naked (1993)` appears at `C8` in chunks 0 and 3,
with target `0` both times, but receives `no` and `yes`, respectively. This
second observation concerns co-candidate/context sensitivity rather than pure
position bias. Replica-level nondeterminism cannot be ruled out because request
and serving-replica identifiers were not saved.

## Interpretation

The candidate order is deterministically shuffled, and the ground-truth label
rate is nearly flat across positions. Therefore the first-position premium is
not direct label leakage. It should instead be interpreted as strong evidence
that the grouped hard yes/no response is affected by list position and prompt
context.

Consequences depend on the intended scientific object:

- For independent pointwise preference prediction, position dependence is an
  undesirable lack of invariance: the same underlying user-item decision should
  not change merely because the item is renamed from `C8` to `C1` or shown next
  to different candidates.
- For slate-conditioned response simulation, co-candidate dependence may be a
  legitimate behavior rather than an error. The protocol and reported claim
  must then explicitly define the simulated response as conditional on the
  displayed slate.
- Random shuffling makes the position effect approximately label-agnostic in
  aggregate, so it does not automatically invalidate the current comparison.
  It can nevertheless add noise, reduce consistency, and affect any individual
  group or user-item prediction.
- A direct listwise ranking protocol may also be position-sensitive. Ranking
  experiments should therefore consider candidate-order counterbalancing or a
  rank-stability analysis, especially when studying uncertainty.

## Limitation of the Current Evidence

This audit is observational. Seeds change the split, candidate construction,
and candidate order simultaneously. They are not controlled permutations of
the exact same prompt. The consistency across all task/seed cells and the
ground-truth-conditioned analysis make the evidence strong, but a causal
position-effect estimate would require holding the history and candidate set
fixed while rotating candidate positions.

A minimal confirmation experiment would use a fixed subset of groups, route
requests to a fixed replica, and compare the original order with one or more
cyclic rotations. A full rerun is not required for this diagnostic.

## Candidate Paper Formulation

> We observed a pronounced first-candidate positional bias in grouped direct
> yes/no prompting. Across two datasets, four candidate ratios, and three seeds,
> Qwen3.6-27B predicted the first candidate as positive 6.0-9.4 percentage
> points more often than candidates in later positions, while the corresponding
> ground-truth rate differed by only -0.4 to +1.7 points. The effect was positive
> in every task-seed cell and persisted after conditioning on the true label:
> both the true-positive and false-positive rates were higher for the first
> slot. Qwen3-8B exhibited the same directional effect. These results suggest
> that list position and co-candidate context can materially affect LLM-simulated
> responses even when candidate order is randomized.

This wording should remain qualified as an observed association until a fixed-
slate position-rotation experiment is completed.

## Reproducibility Pointers

- Qwen3.6-27B outputs:
  `outputs/in_distribution/preference_prediction/20260713T1242Z_qwen36_27b_full/`
- Qwen3-8B outputs:
  `outputs/in_distribution/preference_prediction/20260713T1030Z_qwen3_8b_full/`
- Grouped prompt construction:
  `src/beyond_click_sim/scorers/history/llm.py`
- Candidate shuffling and observed-preference sampling:
  `src/beyond_click_sim/tasks/samplers.py`
- Preference task registry:
  `runners/in_distribution/preference_prediction/task_builders.py`

Model revisions and serving metadata are recorded in each run manifest.
