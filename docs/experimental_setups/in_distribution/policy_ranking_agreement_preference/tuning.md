# Policy hyperparameter tuning

Policy hyperparameters for this setup (and for the interaction-based
`policy_ranking_agreement`) come from offline Optuna tuning artifacts
(`<repo root>/artifacts/`), tuned per (dataset, target, split seed).

The central documentation lives at **`docs/policy_tuning.md`** — protocol,
search spaces, artifact schema, staleness rules, usage, and design decisions.

Setup-specific notes:

- This setup tunes and trains policies on the preference target
  (`target_like_ge4` for ml-1m, `target_played_120` for steam), passed to the
  tuner as both `target_source_column` and `positive_feedback_column`.
- CLI setup name: `preference` (the interaction setup is `interaction`).
