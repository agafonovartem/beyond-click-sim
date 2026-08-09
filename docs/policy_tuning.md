# Policy hyperparameter tuning

Offline Optuna-based hyperparameter search for the recommender **policies** used by the
policy-ranking agreement experiments (`policy_ranking_agreement` and
`policy_ranking_agreement_preference`).

> **Consumption model (updated):** task builders no longer read tuning artifacts directly.
> They build rosters from committed, manually reviewed configs in `configs/policies/` —
> see **`docs/policy_configs.md`**. Tuning artifacts are the *evidence base*; the config is
> the *decision record*. Draft configs from artifacts with
> `runners/in_distribution/policy_tuning/run.py --emit-config`.

| Piece | Location |
|---|---|
| Tuning library | `src/beyond_click_sim/tasks/policy_tuning.py` |
| Tuning runner (CLI) | `runners/in_distribution/policy_tuning/run.py` |
| Artifacts (committed) | `<repo root>/artifacts/{dataset}_{target}_seed{seed}.json` |
| Artifact location source of truth | `default_artifacts_dir()` in `policy_tuning.py` |
| Consumers | `configs/policies/*.json` (via `--emit-config` + manual review) |
| Tests | `tests/test_policy_tuning.py` (+ additions in `test_policies.py`, `test_policy_ranking_task_builder.py`) |

## Motivation

Previously, one hardcoded hyperparameter set was shared across datasets (ml-1m vs steam),
split seeds, and targets (interaction vs preference). Since the policy-ranking experiment
measures simulator agreement against the *real* policy ranking, a misconfigured mid-tier
policy (ALS/BPR/LightGCN/ItemKNN) distorts the ground-truth ranking itself. Tuning the
roster per task configuration removes that confound.

## Architecture and key design decisions

### Offline tuning + committed artifact (not inline tuning)

Tuning runs as a separate step (`policy_tuning/run.py`) that writes one JSON artifact per
job. Task builders read artifacts; they never tune. Rationale: task construction stays cheap
and pure, tuning cost is paid once, artifacts are committed and auditable, and staleness is
detectable via identity validation. Alternatives considered and rejected: lazy tuning inside
the builder with a disk cache (hidden multi-hour cost inside "build a task", fiddly
invalidation) and tune-per-build (12x+ redundant compute, seed-level tuning noise).

### Granularity: per (dataset, target_source_column, split_seed)

2 datasets x 2 setups x N seeds. Axes that change the training distribution get their own
tuning: dataset (ml-1m dense ratings vs steam sparse playtime), target (interaction vs
preference change both the fit frames and the objective), seed (per-seed tuning was a
deliberate choice for maximum hygiene). The eval_users variant (1000 vs full) is test-side
only and shares artifacts.

Consequence to keep in mind when aggregating: cross-seed variance now includes
model-selection variance, not just split variance — seeds may legitimately select different
hyperparameters. Report per-seed; check each task manifest's `hyperparameters`.

### Split discipline (leakage safety)

- The tuner receives **only the outer train frame** (per-user 80% of interactions,
  `MinUserInteractionsFilter(10)` + `RandomFractionSplitter(0.8, 0.0, 0.2, seed)`); test data
  is structurally excluded from `tune_policies_for_split`.
- Inner split: `RandomFractionSplitter(0.75, 0.25, 0.0, seed=split_seed + 1000)` on the outer
  train -> inner train 60% / inner val 20% of all rows (val same expected size as test).
  The +1000 offset matters: the splitter derives per-user randomness as
  `seed + group_position`, so reusing the outer seed would correlate the folds.
- Fit and seen-item exclusion both use the inner train frame, mirroring the deployed
  protocol; val targets are looked up from inner-val rows with the same
  `target_source_column` semantics as the task builder (missing pair -> 0, last-row-wins on
  duplicates).
- After selection, task builders refit the winning config on the full outer train.

### Objective

`macro_by_user_group_mean.ndcg@10` from `evaluate_policy_recommendations`
(`src/beyond_click_sim/evaluation/policy_ranking.py`) — the same ranking-metric machinery
used for the final policy evaluation. Chosen over HR@10 (too coarse, many ties) and mean
target (ignores rank quality).

### Search: Optuna, seeded TPE, small spaces

Standard Optuna workflow: `create_study(direction="maximize", sampler=TPESampler(seed))`
per policy, `study.optimize(n_trials=..., catch=(Exception,))`. Search spaces are declarative
dicts (`SEARCH_SPACES`), hashed (`search_spaces_sha256`) into the artifact identity.

| Policy | Space | n_trials |
|---|---|---|
| ItemKNNPolicy | `n_neighbors` in {10, 20, 50, 100} | 4 |
| ALSPolicy | `n_factors` {16, 32, 64}; `regularization` {0.01, 0.1}; `alpha` {2, 5, 10, 20}; iterations=20 fixed | 24 |
| BPRPolicy | `n_factors` {32, 64}; `learning_rate` {0.001, 0.002, 0.005}; `iterations` {100, 300}; regularization=0.001 fixed | 12 |
| LightGCNTorchPolicy | `n_layers` {2, 3}; `regularization` {1e-4, 1e-3, 1e-2}; `iterations` {300, 1000}; lr=0.001 fixed | 12 |

All four spaces are fully categorical, and `n_trials` equals the grid size in each case.

Notes:

- Box placement for ItemKNN/ALS/BPR comes from an exhaustive sweep on the ml-1m inner
  validation split (`target_interact`, seed 0). Notable results: ItemKNN's optimum is
  interior at 20; ALS capacity is monotonically *decreasing* (32 > 64 > 128 > 256), and
  `alpha` swings ndcg by 0.071 while `regularization` moves it ~0.002 across two decades;
  BPR's `learning_rate` and `iterations` are coupled, and the previous box (lr >= 0.005,
  iterations pinned at 100) capped it at 0.476 where (lr=0.001, iterations=300) reaches
  0.508. The sweep is ml-1m only — re-check on steam before treating these as final.
- LightGCN's box follows He et al. (SIGIR 2020) Sec 4.1.2 and RecBole's published tuning
  range, narrowed to fit the budget; `iterations` is searched in place of the early
  stopping both references use, since `Policy.fit` has no validation data. See the inline
  comment in `SEARCH_SPACES` for the per-parameter rationale.
- With these budgets TPE behaves mostly like seeded random search: `TPESampler` defaults
  to `n_startup_trials=10`, so only trials beyond the tenth are model-guided. Measured
  expected regret of the random draws against the exhaustive grid is small (ALS 0.0016,
  BPR 0.0068 at 12 trials), but a `GridSampler` switch would make selection exhaustive and
  seed-independent.
- `ALSPolicy` exposes `regularization` and `alpha` (added for this feature); defaults
  0.01 / 1.0 equal the implicit library's own defaults, so pre-tuning behavior is unchanged.
  On the all-ones confidence matrix, ALS depends essentially on the alpha/regularization
  ratio — the space spans that ratio rather than pretending the knobs are independent.
- Policy `seed=0` is pinned in both tuning and deployment: artifacts select params for the
  exact model instance the task runs.

### RandomPolicy / PopularityPolicy: nothing is tuned

They have no hyperparameters (`k`, `seed` are protocol constants; popularity's
interaction-vs-preference axis is fixed by the setup's `positive_feedback_column`). They are
still evaluated once and recorded with `tuned: false` for two reasons: (1) one artifact fully
determines the whole roster; (2) their validation NDCG@10 is the audit floor — if a tuned CF
policy ends up below popularity on val in the same artifact, the search failed, and that is
visible before anyone looks at test results.

### Artifact: full trial logs, strict identity

Schema: `{schema_version, setup, generated_at, git_commit, optuna_version, identity: {...},
policies: {ClassName: {tuned, selected_params, val_objective, n_eval_users, sampler, n_trials,
trials: [...]}}}`.

- **Every trial is logged** with its complete metric payload (all NDCG/HR cutoffs from
  `evaluate_policy_recommendations`, mean hit rate, n_users/n_recommendations, timing).
  Failed trials are logged as `{state: "fail", params, error}` — a crashed config is visible,
  not silently missing. A policy whose trials all fail aborts the job instead of writing an
  artifact. The point: you can verify the search landscape, not just reproduce the winner.
- **Identity validation on load is strict**: dataset, version, target,
  `positive_feedback_column`, split seed, policy k/seed, objective, filter config, splitter
  config, inner-split config, search-spaces hash. Any mismatch -> `ValueError` naming the
  fields; missing file -> `RuntimeError` with the re-tune command. This is how cross-setup
  contamination (interaction artifact consumed by the preference builder) and stale artifacts
  after grid/filter/split changes are prevented.
- **Single source of truth for the location**: `default_artifacts_dir()` (repo-root
  `artifacts/`). The runner writes there; both task builders read from there. (A previous
  divergence — runner writing to `outputs/...`, builders reading from `runners/...` — broke
  the pipeline; `outputs/` is also gitignore-restricted, which would have prevented
  committing artifacts.)

### Consumption and provenance

- Task builders call `build_tuned_policy_roster` lazily inside `build_policy_ranking_task`
  (registries stay importable without artifacts).
- Each task manifest records per policy: full `hyperparameters` (public scalar init attrs)
  and `hyperparameter_selection` (`kind`: `optuna_tpe_search` / `fixed_baseline`, objective,
  val objective, n_trials, sampler seed, artifact path, artifact identity).
- The preference setup passes its per-dataset preference target as both
  `target_source_column` and `positive_feedback_column`; the interaction setup uses
  `target_interact` / `None`.

## Usage

```bash
# all jobs: 2 datasets x 2 setups x seeds (default SEEDS from task_builders)
uv run python -m runners.in_distribution.policy_tuning.run

# incremental: skip existing artifacts (e.g. after adding new seeds)
uv run python -m runners.in_distribution.policy_tuning.run --resume

# subset passes: may create a partial artifact, or merge into an existing one
uv run python -m runners.in_distribution.policy_tuning.run \
    --datasets ml-1m --seeds 0 --policies LightGCNTorchPolicy
uv run python -m runners.in_distribution.policy_tuning.run \
    --datasets ml-1m --seeds 0 \
    --policies RandomPolicy,PopularityPolicy,ItemKNNPolicy,ALSPolicy,BPRPolicy
```

## Operational facts worth knowing

- **Adding seeds is incremental**: artifacts are keyed per seed; existing ones stay valid.
  Bump `SEEDS`, run the tuner with `--resume`, done. New tasks appear in the registries
  automatically. Both setups currently use `SEEDS = (0..9)`; the runner's default is to
  resolve seeds **per setup** from each setup's own `SEEDS` registry (an explicit `--seeds`
  list overrides this and applies to all selected setups).
- **Changing `SEARCH_SPACES`, the filter, splitter fractions, `POLICY_K`, or target columns
  invalidates all affected artifacts** (by design, via the identity hash) — re-tune.
- **Until artifacts exist, neither policy-ranking setup can build tasks** — the builders
  raise with the remediation command.
- **`--policies` may write a partial artifact** when the file does not exist yet; a later
  `--policies` pass on the same path merges missing entries in. Do not run two writers on
  the same artifact path concurrently. `--emit-config` refuses partial artifacts (missing
  roster policies).
- LightGCN is the bottleneck: seeds x setups x datasets x 12 trials, each a full
  mini-batch training run (~292 optimiser steps per epoch on ml-1m, up to 1000 epochs).
  That is 144 fits across the sweep. Benchmark one fit on the GPU pod before launching,
  and split it into its own `--policies LightGCNTorchPolicy` pass if needed.
- `LightGCNTorchPolicy` selects its device automatically (`device="auto"` resolves
  cuda -> mps -> cpu at fit time) and records `resolved_device` in the *task* manifest.
  The tuning artifact does not record the device, and PyTorch does not guarantee
  CPU/GPU equivalence — do not mix devices within a result set.
- Setup names in the CLI are `interaction` and `preference` (`--setups`).

## Verification status

- `tests/test_policy_tuning.py`: artifact completeness, argmax selection, determinism
  (params/objectives; timings excluded), JSON serializability, inner-split leakage spy
  (fit/recommend see exactly inner train, disjoint from inner val), failed-trial logging,
  all-fail refusal, roster roundtrip, missing/stale artifact errors.
- Full suite: 556 passed, 2 pre-existing failures unrelated to tuning
  (`cache_stats` expectations vs the in-flight `profiles.py` change).
- Smoke-verified on real ml-1m (ItemKNN, interaction target, seed 0): selected
  `n_neighbors=10`, val NDCG@10 0.5345 vs 0.5156 for `n_neighbors=100`.
