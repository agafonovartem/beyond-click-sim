# Policy roster configs

The policy-ranking experiments (`policy_ranking_agreement`,
`policy_ranking_agreement_preference`) build their recommender-policy rosters from
**committed, manually reviewed JSON configs** — not from raw tuning output.

| Piece | Location |
|---|---|
| Config files (committed) | `configs/policies/{setup}.json` (`interaction.json`, `preference.json`) |
| Loader | `src/beyond_click_sim/tasks/policy_config.py` (`load_policy_roster`) |
| Config dir source of truth | `default_policies_config_dir()` in `policy_config.py` |
| Tuning evidence artifacts | `<repo root>/artifacts/` (see `docs/policy_tuning.md`) |
| Skeleton emitter | `runners/in_distribution/policy_tuning/run.py --emit-config` |

## Why configs instead of auto-consumed tuning artifacts

The tuner produces evidence (full trial logs per (dataset, target, seed)); the config is the
**decision record**. A human reviews the artifacts, picks hyperparameters, and commits them.
This keeps model selection deliberate and auditable, and it decouples experiment runs from
the tuner's raw output. `tuned_from` links preserve the connection from each config entry
back to the search log it was chosen from.

## Config schema (schema_version 1)

```json
{
  "schema_version": 1,
  "setup": "preference",
  "datasets": {
    "ml-1m": {
      "default": [
        {"name": "random", "class": "RandomPolicy"},
        {"name": "popularity", "class": "PopularityPolicy",
         "comment": "preference-popularity floor"},
        {"name": "itemknn", "class": "ItemKNNPolicy",
         "params": {"n_neighbors": 10},
         "tuned_from": "artifacts/ml-1m_target_like_ge4_seed0.json"},
        {"name": "als", "class": "ALSPolicy",
         "params": {"n_factors": 64, "regularization": 0.1, "alpha": 10.0},
         "tuned_from": "artifacts/ml-1m_target_like_ge4_seed0.json"},
        {"name": "als_f32", "class": "ALSPolicy",
         "params": {"n_factors": 32},
         "comment": "near-tie variant for the hard sub-roster"}
      ],
      "seed3": [ {"name": "...", "class": "...", "params": {}} ],
      "_review_notes": ["..."]
    }
  }
}
```

- **Entry keys**: `name` (required, unique within the resolved roster — it is the policy's
  identity in the test frame, metrics, and manifests), `class` (required; one of the classes
  in `POLICY_CLASSES`), `params` (model hyperparameters only), optional `tuned_from`,
  optional `comment`. Unknown keys are rejected (typo guard).
- **Resolution**: `datasets[dataset]["seed{N}"]` if present, else `["default"]`. Section
  keys are validated (`default`, `seed<N>`, or `_`-prefixed informational keys only) so a
  typo'd `seed_3` cannot silently fall back to `default`.
- **Injected, never configured**: `k` (task-level `POLICY_K`), `seed` (policy seed, pinned
  0), `positive_feedback_column` (setup-determined), `name`. Any of these inside `params`
  is an error.
- **Validation on load**: file presence, `schema_version`, `setup` match, dataset section,
  unique names, known class, constructor errors wrapped with entry context. All failures
  name the file, section, and entry.

## Workflow

```bash
# 1. tune (writes artifacts/ with full trial logs)
uv run python -m runners.in_distribution.policy_tuning.run [--resume]

# 2. draft configs from artifacts (never overwrites without --overwrite)
uv run python -m runners.in_distribution.policy_tuning.run --emit-config

# 3. MANUAL STEP: review the artifact trial logs and the _review_notes,
#    edit configs/policies/*.json (pick params, add variants, write comments),
#    commit the configs.

# 4. run experiments — task builders load rosters from the configs
uv run python -m runners.in_distribution.policy_ranking_agreement_preference.run ...
```

The skeleton's `default` roster comes from each dataset's lowest-seed artifact;
`_review_notes` per dataset list which policies got *different* params across seeds — that
is the signal for whether per-seed `seed<N>` override sections are warranted.

## Coat is tuned under a different protocol

The `coat` section of `preference.json` does **not** come from the workflow above. It is
produced by `runners/in_distribution/policy_tuning_rerank/run.py`, and the differences are
not cosmetic — do not reason about that section using the ml-1m rules.

| | ml-1m / steam | coat |
| --- | --- | --- |
| tuner | `tasks/policy_tuning.py` | `tasks/policy_tuning_rerank.py` |
| candidates | free retrieval over the catalog | re-rank the user's own 16-item held-out pool |
| validation rows | inner 25% split of outer train | the held-out MCAR half — **also the evaluation set** |
| objective | `macro_by_user_group_mean.ndcg@10` | `mean_hit_rate` (= precision@k, the runner's own utility) |
| `iterations` | pinned at 300 | tuned, like any other hyperparameter |
| grid size | 4-12 points per policy | 11-480; a full pass is a few minutes on 290x300 |
| unlabelled candidates | zero-filled | impossible; the join is strict and raises |
| artifact | `{dataset}_{target}_seed{n}.json` | `{dataset}_{target}_rerank_users{N}_seed{n}.json` |
| config emission | `--emit-config` writes the file | `--print-config` prints a block for hand-merge |

Coat forces all of this. Its train half holds 22.74 rows per user, so a 25% inner split
leaves 5.68 candidates against `POLICY_K = 5` — a top-5 covering 88% of the pool, where no
setting can distinguish itself. Scoring free retrieval instead is worse, not a fallback:
with 5.68 labelled cells among 300 items, the zero-fill would fabricate ~98% of the
objective. The only usable validation labels Coat has are the uniformly-assigned ones, and
those are the labels the experiment reports against.

**So the Coat roster is selected on its own evaluation labels, and its real utilities are
optimistically biased.** The bias is measured rather than asserted: a diagnostic pass
(`--tuning-user-fraction 0.5`) selects on half the users, scores the selection on both
halves, and subtracts the same gap averaged over the policy's whole grid — which removes
the two halves' base-rate difference and leaves the selection bias alone. A policy with
nothing to select reports exactly zero, which is the correct answer and the estimator's
self-check. Numbers live in the artifact's `curse_summary` and in the section's
`_review_notes`. One thing that measurement does *not* cover: the search boxes themselves
were revised once against the diagnostic half, after every policy first selected on a
boundary. Selecting the grid is a layer above selecting within it.

`iterations` is tuned rather than pinned because 300 is a budget chosen for ml-1m, and
carrying it to a matrix a thousand times smaller assumes the answer. It matters: once the
budget was free, BPR and LightGCN both went to the ceiling of `{100, 300, 1000}`, and BPR
then preferred the *smallest* model on offer having wanted the largest while pinned —
capacity and training budget are not separable questions here. ALS is the sharper case: its
`iterations` is absent from the ml-1m grid entirely, so it had been running the class
default of 20 unexamined.

Both artifacts are kept. `emit_configs` must never be pointed at Coat — it rebuilds the
whole config file from artifacts and would drop the hand-maintained ml-1m section.

## Steam's roster is deliberately widened

`interaction.json` holds **15 entries for ml-1m and 18 for steam**, but only 6 policies exist.
The rest are the same classes re-instantiated at *knowingly worse* hyperparameters. This is a
design choice about the measurement, not about the models, and it is worth understanding before
editing either section.

The experiment's statistic is a rank correlation between simulated and real policy utility. With
6 policies there are 15 pairs, so Kendall's tau can only take values 2/15 ≈ 0.133 apart, and its
confidence interval is far too wide to call significant. Worse, the tuned six do not spread out:
on steam the top three sit inside 0.0037–0.0085 across every seed, and ALS and LightGCN swap
places between seeds (6/9 vs 3/9). A ranking built on that is mostly noise. Adding variants at
spaced-out utilities turns 15 pairs into 153, moving the tau lattice to 2/153 ≈ 0.013.

### The `_mean=X` naming convention

Shared by both dataset sections. `X` is the variant's **mean `macro_by_user_group_mean.ndcg@10`
across the tuning seeds** — validation-side evidence, never a test number. Decimals are the
fewest that keep the labels unique within a class, which is why steam's ItemKNN entries need
three (`0.556`/`0.566`/`0.571`) while everything else needs two. The label is a human-readable
handle, not an assertion about the run: the ordering in the actual experiment does differ, and
on ml-1m `ALSPolicy_mean=0.56` finished *above* the tuned `ALSPolicy`.

### How steam's variants were chosen

Every variant is a grid point the tuner actually evaluated, so its objective is measured rather
than assumed. Three per tunable policy, selected under a hard constraint plus an objective:

- **Constraint** — every pair on a policy's ladder, the tuned pick included, must have a paired
  95% CI on the across-seed objective difference that excludes zero *and* order identically on
  all 9 complete seeds.
- **Objective** — among qualifying triples, take the one that maximises the *tightest* rung, so
  the levels spread instead of clumping.

Rungs below are each entry's drop from the row above it, read left to right down the ladder.

| policy | tuned | variants (mean ndcg@10) | rungs (drop from the level above) |
| --- | --- | --- | --- |
| ALSPolicy | 0.6000 | 0.5794, 0.5595, 0.5386 | 0.0205 / 0.0200 / 0.0209 |
| BPRPolicy | 0.4516 | 0.3954, 0.3453, 0.3136 | 0.0562 / 0.0501 / 0.0316 |
| ItemKNNPolicy | 0.5946 | 0.5707, 0.5662, 0.5562 | 0.0238 / **0.0046** / **0.0099** |
| LightGCNTorchPolicy | 0.5995 | 0.5800, 0.5381, 0.5313 | 0.0195 / 0.0419 / **0.0068** |

Popularity and Random have no hyperparameters and therefore no variants.

### Two of the four ladders will produce near-ties in the run

Separability above is measured on the tuning validation split — 57,333 users, ndcg@10. The
experiment scores `mean_hit_rate` on 1,000 eval users, which resolves far less. A paired
bootstrap over the ml-1m 15-policy run (`20260820T020733Z`, 2,000 replicates) separates adjacent
rungs reliably only from about **0.010** upward. **7 of its 14 adjacent pairs overlap zero.** Below
0.010 it is a coin flip: a 0.0057 gap came out separable there while a 0.0069 gap did not.

ItemKNN carries two rungs under that bar and LightGCN one (bolded above). Neither is a bad pick
— both grids are simply too narrow. ItemKNN has 4 points total and three of them fall inside
0.556–0.571; LightGCN's 6 points cluster into 0.53–0.54 and 0.58–0.60 with nothing between. No
alternative triple is wider. Expect those entries to read as ties in the results, and do not
interpret their relative order.

### Two things the steam section does not certify

- **The search boxes were placed by an ml-1m sweep**, as the comment above `SEARCH_SPACES` says
  outright. Steam pushes against them: ItemKNN selects the minimum `n_neighbors` on 9/9 seeds
  with the steepest slope of any policy, BPR selects the minimum `learning_rate` and maximum
  `n_factors` on 9/9 (its `iterations` and `regularization` are single-choice, so not searched
  at all), and LightGCN sits on both its edges on 10/10. The tuned params are the best values
  *inside the box*; for ItemKNN and BPR they are probably not the best values.
- **Seed 9 is a partial artifact** — LightGCN only, from an earlier GPU batch. `--emit-config`
  refuses to draft while any artifact is incomplete, so this section was hand-merged rather than
  emitted. Widening any box changes `search_spaces_sha256` and invalidates all 10 steam *and*
  all 10 ml-1m artifacts, so settle the boxes before re-tuning.

### Cost

Roster size is not cost-neutral and LightGCN dominates it. On the ml-1m run above, its four
entries took 53.5 + 26.1 + 22.7 + 21.1 minutes against ~0.2 minutes for each ALS, ItemKNN, and
Popularity entry — 97% of the 2.11 h fit budget for one seed. Steam is an order of magnitude
larger. Budget a steam sweep around its four LightGCN entries; if that is prohibitive, the
cheapest thing to drop is `LightGCNTorchPolicy_mean=0.53`, whose rung was already below
resolution.

## Caveats (read before trusting a run)

- **The staleness guardrail is weaker than with artifacts.** Artifact identity was
  hash-validated automatically; configs are hand-maintained, so nothing forces re-review
  when the dataset/filter/splitter changes. The config *is* the selection record — treat
  edits like experiment changes (commit message, `tuned_from`, `comment`).
- **Pick from validation-side evidence only.** The tuning artifacts contain inner-val
  metrics; never choose params using test-side outputs. **Coat is the documented
  exception** — it has no third partition, so its validation labels *are* its evaluation
  labels. See the section above; the resulting bias is measured and recorded, and any
  reported number from that roster has to carry it.
- **Name-based identity changes output schemas.** The `policy` column in
  `predictions.parquet`, utility dicts, and manifests carry config names (`als_f32`), not
  class names. Downstream analysis must key on these; outputs produced before this change
  use class names and don't join directly.
- **Variants multiply LLM cost** (~linearly in roster size) and may drift from near-tie
  status across seeds: tuned entries are per-seed, variants are fixed. Define hard pairs
  per seed, or treat variants as absolute fixed points.
- **Missing config = loud failure**: until `configs/policies/{setup}.json` exists with a
  section for the requested (dataset, seed), task builds raise with the file path and the
  missing section named.

## Provenance in outputs

Each task manifest's `policies` block records, per entry: `name`, `class`, full
`hyperparameters`, and `hyperparameter_selection` = `{kind: "manual_config", config,
section, tuned_from, comment}` — enough to trace every deployed policy back to the config
section and the tuning evidence behind it.
