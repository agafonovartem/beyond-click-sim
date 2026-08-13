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

## Caveats (read before trusting a run)

- **The staleness guardrail is weaker than with artifacts.** Artifact identity was
  hash-validated automatically; configs are hand-maintained, so nothing forces re-review
  when the dataset/filter/splitter changes. The config *is* the selection record — treat
  edits like experiment changes (commit message, `tuned_from`, `comment`).
- **Pick from validation-side evidence only.** The tuning artifacts contain inner-val
  metrics; never choose params using test-side outputs.
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
