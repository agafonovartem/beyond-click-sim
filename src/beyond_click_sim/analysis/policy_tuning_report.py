"""Read and audit policy-tuning artifacts written by the policy-tuning runner.

One artifact per `(dataset, target_source_column, split_seed)` lives in
`artifacts/` and records the *complete* Optuna trial log — every config's
objective, full ranking-metric block, and wall time — not just the winner.
This module turns a dataset's artifacts into flat tables and a set of audit
panels.

The panels exist to answer two questions:

1. Which single hyperparameter set should be frozen across seeds, and what does
   freezing it cost relative to per-seed tuning (`panel_shared_config`)?
2. Is the tuned roster a sound reference for policy-ranking agreement — i.e. is
   the *ordering* of policies stable, or are some pairs tied within seed noise
   (`panel_policy_ordering`)?

Two properties of the tuning protocol make naive reads of these artifacts
misleading, so several panels exist specifically to measure them:

- `TPESampler` resamples duplicates on small categorical grids, so a seed
  explores only part of its own search space. Per-seed parameter differences
  therefore mix "the optimum moved" with "that cell was never tried"
  (`panel_search_coverage`).
- `val_objective` is a *max* over trials, so it is upward-biased by an amount
  that grows with the number of unique configs evaluated. Comparing per-seed
  bests across seeds is not like-for-like; comparing a fixed config is
  (`panel_signal_vs_noise`).

Nothing here writes to disk. `panel_shared_config` prints a config block for
manual review; it never edits `configs/policies/`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from beyond_click_sim.tasks.policy_tuning import (
    ARTIFACT_SCHEMA_VERSION,
    OBJECTIVE_METRIC,
    SEARCH_SPACES,
    default_artifacts_dir,
    search_spaces_sha256,
)

__all__ = [
    "PARAM_PREFIX",
    "MUGM_PREFIX",
    "MBG_PREFIX",
    "SearchSpace",
    "TuningArtifacts",
    "load_tuning_artifacts",
    "per_policy_tables",
    "resolve_search_spaces",
    "config_label",
    "apply_style",
    "save_open_figures",
    "panel_completeness",
    "panel_search_coverage",
    "panel_determinism",
    "panel_signal_vs_noise",
    "panel_selection_frequency",
    "panel_param_stability",
    "panel_shared_config",
    "panel_policy_ordering",
    "panel_eval_population",
    "panel_metric_decomposition",
    "panel_cost",
]

# Column prefixes in the `trials` table.
PARAM_PREFIX = "param__"
MUGM_PREFIX = "mugm__"  # ranking.macro_by_user_group_mean
MBG_PREFIX = "mbg__"  # ranking.macro_by_group

FIGSIZE = (11.0, 4.0)

# `panel_signal_vs_noise` verdict thresholds, in units of
# (spread across configs) / (spread across seeds). Arbitrary but explicit: a
# hyperparameter whose effect is smaller than the seed-to-seed noise cannot be
# selected reliably, whatever the search budget.
SIGNAL_RATIO_STRONG = 5.0
SIGNAL_RATIO_WEAK = 2.0

try:  # pragma: no cover - display convenience only
    from IPython.display import display as _display
except ImportError:  # pragma: no cover
    def _display(obj: Any) -> None:
        print(obj)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchSpace:
    """Resolved search space for one policy.

    `choices` maps a parameter name to its ordered choice list. `enumerable` is
    False when a parameter uses a continuous (`float`/`int`) range, in which
    case `grid_size` is undefined and coverage cannot be computed.
    """

    policy: str
    choices: dict[str, tuple[Any, ...]]
    grid_size: int | float
    enumerable: bool
    tuned: bool

    def grid_config_keys(self) -> list[str]:
        """Canonical config keys for every cell of the grid (empty if not enumerable)."""

        if not self.enumerable or not self.choices:
            return []
        names = sorted(self.choices)
        return [
            _config_key(dict(zip(names, values, strict=True)))
            for values in product(*(self.choices[name] for name in names))
        ]


@dataclass(frozen=True)
class TuningArtifacts:
    """All tuning artifacts for one dataset, flattened into tables.

    Attributes
    ----------
    files:
        One row per artifact file: seed, target, setup, provenance fields.
    identity:
        One row per file, the flattened `identity` block that
        `load_tuned_artifact` validates on load.
    selection:
        One row per (seed, policy): the tuner's own choice and its
        `val_objective`.
    trials:
        One row per (seed, policy, trial) — every field from the artifact.
        This is the table to build custom panels on; see `per_policy_tables`.
    search_spaces:
        Policy name -> `SearchSpace`.
    space_source:
        "live_module" when the artifacts' `search_spaces_sha256` matches the
        current `SEARCH_SPACES`, else "reconstructed_from_trials" — in which
        case grid sizes are *lower bounds* and coverage is optimistic.
    """

    dataset: str
    artifacts_dir: Path
    files: pd.DataFrame
    identity: pd.DataFrame
    selection: pd.DataFrame
    trials: pd.DataFrame
    search_spaces: dict[str, SearchSpace]
    space_source: str

    @property
    def seeds(self) -> list[int]:
        return sorted(self.files["seed"].unique().tolist())

    @property
    def policies(self) -> list[str]:
        """Policies in artifact order (which is `POLICY_ROSTER` order)."""

        return list(dict.fromkeys(self.trials["policy"].tolist()))

    @property
    def tuned_policies(self) -> list[str]:
        tuned = self.selection.loc[self.selection["tuned"], "policy"]
        return [name for name in self.policies if name in set(tuned)]

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)

    def __repr__(self) -> str:  # pragma: no cover - display convenience
        return (
            f"TuningArtifacts(dataset={self.dataset!r}, "
            f"seeds={self.seeds}, policies={self.policies}, "
            f"trials={len(self.trials)}, space_source={self.space_source!r})"
        )


def _config_key(params: dict[str, Any]) -> str:
    """Canonical, hashable identity of a hyperparameter config."""

    return json.dumps(params, sort_keys=True)


def config_label(config_key: str) -> str:
    """Human-readable one-line label for a config key."""

    params = json.loads(config_key)
    if not params:
        return "(defaults)"
    return ", ".join(f"{name}={value}" for name, value in sorted(params.items()))


def _flatten(obj: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to dotted keys; lists become tuples (hashable)."""

    out: dict[str, Any] = {}
    for key, value in obj.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, prefix=f"{name}."))
        elif isinstance(value, list):
            out[name] = tuple(value)
        else:
            out[name] = value
    return out


def _metric_columns(metrics: dict[str, Any]) -> dict[str, Any]:
    """Flatten one trial's metric payload, keeping both ranking aggregations.

    Walks the dict generically so metric keys added later show up without a
    code change here.
    """

    out: dict[str, Any] = {}
    ranking = metrics.get("ranking") or {}
    for key, value in metrics.items():
        if key == "ranking":
            continue
        out[key] = value
    for key, value in (ranking.get("macro_by_user_group_mean") or {}).items():
        out[f"{MUGM_PREFIX}{key}"] = value
    for key, value in (ranking.get("macro_by_group") or {}).items():
        out[f"{MBG_PREFIX}{key}"] = value
    if "ks" in ranking:
        out["ranking_ks"] = tuple(ranking["ks"])
    if "tie_policy" in ranking:
        out["ranking_tie_policy"] = ranking["tie_policy"]
    return out


def load_tuning_artifacts(
    dataset: str,
    artifacts_dir: str | Path | None = None,
    *,
    seeds: Iterable[int] | None = None,
    setup: str | None = None,
    target: str | None = None,
) -> TuningArtifacts:
    """Load every tuning artifact for one dataset into flat tables.

    Parameters
    ----------
    dataset:
        Dataset name as it appears in the artifact filename, e.g. "ml-1m".
    artifacts_dir:
        Defaults to `default_artifacts_dir()` (`<repo root>/artifacts/`), the
        same location the tuning runner writes to.
    seeds, setup, target:
        Optional filters. `target` is the `target_source_column` embedded in
        the filename; leaving it None picks up whichever targets are present
        and reports a conflict via `panel_completeness` if more than one is.
    """

    artifacts_dir = (
        Path(artifacts_dir).expanduser().resolve()
        if artifacts_dir is not None
        else default_artifacts_dir()
    )
    # Dataset names contain "-" and targets contain "_", so anchor the regex on
    # both the known dataset prefix and the _seed<N>.json suffix.
    pattern = re.compile(rf"^{re.escape(dataset)}_(?P<target>.+)_seed(?P<seed>\d+)\.json$")
    wanted_seeds = None if seeds is None else {int(s) for s in seeds}

    loaded: list[tuple[Path, int, str, dict[str, Any]]] = []
    for path in sorted(artifacts_dir.glob(f"{dataset}_*_seed*.json")):
        match = pattern.match(path.name)
        if match is None:
            continue
        seed = int(match.group("seed"))
        file_target = match.group("target")
        if wanted_seeds is not None and seed not in wanted_seeds:
            continue
        if target is not None and file_target != target:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if setup is not None and payload.get("setup") != setup:
            continue
        loaded.append((path, seed, file_target, payload))

    if not loaded:
        raise FileNotFoundError(
            f"No tuning artifacts for dataset {dataset!r} in {artifacts_dir} "
            f"(seeds={seeds}, setup={setup}, target={target}). Tune first with: "
            "uv run python -m runners.in_distribution.policy_tuning.run "
            f"--datasets {dataset}"
        )
    loaded.sort(key=lambda item: item[1])

    file_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []

    for path, seed, file_target, payload in loaded:
        policies = payload.get("policies", {})
        file_rows.append({
            "seed": seed,
            "file": path.name,
            "target": file_target,
            "setup": payload.get("setup"),
            "schema_version": payload.get("schema_version"),
            "generated_at": payload.get("generated_at"),
            "git_commit": payload.get("git_commit"),
            "optuna_version": payload.get("optuna_version"),
            "n_policies": len(policies),
            "policies": tuple(policies),
        })
        identity_rows.append({"seed": seed, **_flatten(payload.get("identity", {}))})

        for policy, entry in policies.items():
            selected_params = entry.get("selected_params") or {}
            sampler = entry.get("sampler") or {}
            selection_rows.append({
                "seed": seed,
                "policy": policy,
                "tuned": bool(entry.get("tuned", False)),
                "val_objective": entry.get("val_objective"),
                "n_eval_users": entry.get("n_eval_users"),
                "n_trials_budget": entry.get("n_trials"),
                "sampler_name": sampler.get("name"),
                "sampler_seed": sampler.get("seed"),
                "selected_params": selected_params,
                "selected_config_key": _config_key(selected_params),
                **{f"{PARAM_PREFIX}{k}": v for k, v in selected_params.items()},
            })

            for trial in entry.get("trials", []):
                params = trial.get("params") or {}
                metrics = trial.get("metrics") or {}
                trial_rows.append({
                    "seed": seed,
                    "policy": policy,
                    "trial": trial.get("number"),
                    "state": trial.get("state"),
                    "error": trial.get("error"),
                    "config_key": _config_key(params),
                    "value": trial.get("value"),
                    "tuned": bool(entry.get("tuned", False)),
                    "val_objective": entry.get("val_objective"),
                    "n_eval_users": entry.get("n_eval_users"),
                    **{f"{PARAM_PREFIX}{k}": v for k, v in params.items()},
                    **_metric_columns(metrics),
                })

    files = pd.DataFrame(file_rows).sort_values("seed").reset_index(drop=True)
    identity = pd.DataFrame(identity_rows).sort_values("seed").reset_index(drop=True)
    selection = pd.DataFrame(selection_rows)
    trials = pd.DataFrame(trial_rows)

    trials = _add_derived_trial_columns(trials, selection)
    selection = selection.sort_values(["seed", "policy"]).reset_index(drop=True)

    spaces, space_source = resolve_search_spaces(trials, identity)

    return TuningArtifacts(
        dataset=dataset,
        artifacts_dir=artifacts_dir,
        files=files,
        identity=identity,
        selection=selection,
        trials=trials,
        search_spaces=spaces,
        space_source=space_source,
    )


def _add_derived_trial_columns(
    trials: pd.DataFrame,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    """Add duplicate/selection flags and positive-conditional ranking metrics."""

    trials = trials.copy()
    grouped = trials.groupby(["seed", "policy"], sort=False)

    trials["dup_index"] = trials.groupby(
        ["seed", "policy", "config_key"], sort=False
    ).cumcount()
    trials["is_duplicate"] = trials["dup_index"] > 0

    selected = selection.set_index(["seed", "policy"])["selected_config_key"]
    keys = pd.MultiIndex.from_frame(trials[["seed", "policy"]])
    trials["is_selected"] = trials["config_key"].to_numpy() == selected.reindex(
        keys
    ).to_numpy()

    best = grouped["value"].transform("max")
    trials["is_best_in_seed"] = trials["value"].eq(best) & trials["value"].notna()

    # Users whose recommended list contains no positive score exactly 0 for
    # both nDCG and hit rate (evaluation/ranking.py:161-162, 194-195), and
    # `groups_without_positive` counts exactly those groups (ranking.py:248).
    # So the mean restricted to users with >= 1 positive is recoverable
    # exactly, not approximately. Only valid when there is one group per user,
    # which is what `candidate_group = user::policy` guarantees
    # (evaluation/policy_ranking.py:131-133) — verified per row below.
    n_groups = trials.get(f"{MUGM_PREFIX}n_groups")
    n_users = trials.get(f"{MUGM_PREFIX}n_users")
    without = trials.get(f"{MUGM_PREFIX}groups_without_positive")
    if n_groups is not None and without is not None:
        one_group_per_user = (
            n_groups.eq(n_users) if n_users is not None else pd.Series(False, index=trials.index)
        )
        with_positive = n_groups - without
        scale = np.where(
            one_group_per_user & with_positive.gt(0),
            n_groups / with_positive.replace(0, np.nan),
            np.nan,
        )
        for column in trials.columns:
            if not column.startswith(MUGM_PREFIX):
                continue
            metric = column[len(MUGM_PREFIX):]
            if not (metric.startswith("ndcg@") or metric.startswith("hit_rate@")):
                continue
            trials[f"{column}_cond"] = trials[column] * scale

    return trials.sort_values(["seed", "policy", "trial"]).reset_index(drop=True)


def per_policy_tables(arts: TuningArtifacts) -> dict[str, pd.DataFrame]:
    """One complete table per policy — the extension point for custom panels.

    Each table is `arts.trials` restricted to that policy, with parameter
    columns that policy never used dropped and the `param__` prefix stripped.
    Nothing is aggregated: every field present in the artifact is reachable.
    """

    tables: dict[str, pd.DataFrame] = {}
    for policy, frame in arts.trials.groupby("policy", sort=False):
        frame = frame.drop(columns=["policy"]).dropna(axis=1, how="all")
        renamed = {
            column: column[len(PARAM_PREFIX):]
            for column in frame.columns
            if column.startswith(PARAM_PREFIX)
        }
        tables[str(policy)] = (
            frame.rename(columns=renamed)
            .sort_values(["seed", "trial"])
            .reset_index(drop=True)
        )
    return {policy: tables[policy] for policy in arts.policies if policy in tables}


def resolve_search_spaces(
    trials: pd.DataFrame,
    identity: pd.DataFrame,
) -> tuple[dict[str, SearchSpace], str]:
    """Resolve each policy's choice lists, preferring the live `SEARCH_SPACES`.

    The artifact stores only `search_spaces_sha256`, not the space itself. When
    that hash matches the current module the live choice lists are exact. When
    it does not, choices are reconstructed from observed trial parameters,
    which can only contain cells that were actually sampled — so grid sizes
    become *lower bounds* and coverage figures are optimistic.
    """

    found_hashes = set(identity.get("search_spaces_sha256", pd.Series(dtype=str)).dropna())
    live_hash = search_spaces_sha256(SEARCH_SPACES)
    use_live = found_hashes == {live_hash}
    source = "live_module" if use_live else "reconstructed_from_trials"

    spaces: dict[str, SearchSpace] = {}
    for policy, frame in trials.groupby("policy", sort=False):
        policy = str(policy)
        tuned = bool(frame["tuned"].iloc[0])
        if not tuned:
            spaces[policy] = SearchSpace(policy, {}, 1, True, tuned=False)
            continue

        if use_live and policy in SEARCH_SPACES:
            choices: dict[str, tuple[Any, ...]] = {}
            enumerable = True
            for name, spec in SEARCH_SPACES[policy]["params"].items():
                if spec.get("kind") == "categorical":
                    choices[name] = tuple(spec["choices"])
                else:
                    enumerable = False
            grid_size = (
                int(np.prod([len(v) for v in choices.values()])) if enumerable and choices
                else float("nan")
            )
        else:
            enumerable = True
            choices = {}
            for column in frame.columns:
                if not column.startswith(PARAM_PREFIX):
                    continue
                values = frame[column].dropna().unique().tolist()
                if values:
                    choices[column[len(PARAM_PREFIX):]] = tuple(sorted(values))
            grid_size = (
                int(np.prod([len(v) for v in choices.values()])) if choices else 1
            )

        spaces[policy] = SearchSpace(policy, choices, grid_size, enumerable, tuned=True)

    return spaces, source


def save_open_figures(fig_dir: str | Path, prefix: str = "") -> list[Path]:
    """Write every currently open matplotlib figure to `fig_dir` as PNG."""

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for number in plt.get_fignums():
        figure = plt.figure(number)
        label = figure.get_label() or f"figure{number}"
        path = fig_dir / f"{prefix}{label}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        written.append(path)
    return written


# --------------------------------------------------------------------------
# Shared panel helpers
# --------------------------------------------------------------------------


def apply_style() -> None:
    """Consistent seaborn/matplotlib defaults for the report."""

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.autolayout"] = False
    plt.rcParams["axes.titlesize"] = "medium"


def _complete(trials: pd.DataFrame, objective_column: str = "value") -> pd.DataFrame:
    """Completed, deduplicated trials with a usable objective."""

    frame = trials[(trials["state"] == "complete") & trials[objective_column].notna()]
    return frame[~frame["is_duplicate"]]


def _tuned_policies_with_data(arts: TuningArtifacts) -> list[str]:
    present = set(arts.trials.loc[arts.trials["tuned"], "policy"])
    return [policy for policy in arts.policies if policy in present]


def _subplot_grid(n: int, ncols: int = 2, height: float = 3.4) -> tuple[Any, list[Any]]:
    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols))
    figure, axes = plt.subplots(
        nrows, ncols, figsize=(FIGSIZE[0], height * nrows), squeeze=False
    )
    flat = axes.ravel().tolist()
    for ax in flat[n:]:
        ax.set_visible(False)
    return figure, flat[:n]


def _check(name: str, ok: bool | None, detail: str) -> dict[str, Any]:
    status = "WARN" if ok is None else ("PASS" if ok else "FAIL")
    return {"check": name, "status": status, "detail": detail}


# --------------------------------------------------------------------------
# Panel 1 — completeness and provenance
# --------------------------------------------------------------------------


def panel_completeness(
    arts: TuningArtifacts,
    *,
    expected_policies: Sequence[str] | None = None,
    expected_seeds: Sequence[int] | None = None,
    show: bool = True,
) -> dict[str, pd.DataFrame]:
    """Gate the report: is this a coherent, complete set of artifacts?

    Also asserts two invariants that are structurally guaranteed by the
    evaluation code, so a failure means something upstream changed:
    `macro_by_group == macro_by_user_group_mean` (one candidate group per user,
    evaluation/policy_ranking.py:131-133) and `groups_with_score_ties == 0`
    (scores are `-rank`, hence unique within a user, policy_ranking.py:130).
    """

    files = arts.files
    trials = arts.trials
    checks: list[dict[str, Any]] = []

    versions = set(files["schema_version"])
    checks.append(_check(
        "schema_version",
        versions == {ARTIFACT_SCHEMA_VERSION},
        f"found {sorted(versions)}, expected {ARTIFACT_SCHEMA_VERSION}",
    ))

    for field in ("setup", "target", "optuna_version"):
        values = set(files[field].dropna())
        checks.append(_check(
            f"single {field}",
            len(values) <= 1,
            f"{sorted(values)}" if values else "missing",
        ))

    seeds = arts.seeds
    expected = list(expected_seeds) if expected_seeds is not None else list(
        range(min(seeds), max(seeds) + 1)
    )
    missing_seeds = sorted(set(expected) - set(seeds))
    checks.append(_check(
        "seed coverage",
        not missing_seeds,
        f"{len(seeds)} present {seeds}"
        + (f"; missing {missing_seeds}" if missing_seeds else ""),
    ))

    roster = (
        list(expected_policies) if expected_policies is not None
        else sorted({p for row in files["policies"] for p in row})
    )
    per_file_missing = {
        row.file: sorted(set(roster) - set(row.policies))
        for row in files.itertuples()
        if set(roster) - set(row.policies)
    }
    checks.append(_check(
        "policy roster complete",
        not per_file_missing,
        f"roster={roster}" if not per_file_missing
        else f"{len(per_file_missing)} file(s) partial, e.g. "
             f"{next(iter(per_file_missing.items()))}",
    ))

    hashes = set(arts.identity.get("search_spaces_sha256", pd.Series(dtype=str)).dropna())
    checks.append(_check(
        "search_spaces_sha256 consistent",
        len(hashes) == 1,
        f"{len(hashes)} distinct hash(es)",
    ))
    checks.append(_check(
        "search space matches live SEARCH_SPACES",
        arts.space_source == "live_module",
        "grid sizes are exact" if arts.space_source == "live_module"
        else "MISMATCH — grids reconstructed from observed trials; "
             "coverage numbers below are optimistic lower bounds",
    ))

    varying = {"seed", "split_seed", "splitter.seed"}
    unstable = [
        column for column in arts.identity.columns
        if column not in varying and arts.identity[column].nunique(dropna=False) > 1
    ]
    checks.append(_check(
        "identity constant across seeds",
        not unstable,
        "only split_seed varies" if not unstable else f"also varies: {unstable}",
    ))

    n_null_commit = int(files["git_commit"].isna().sum())
    checks.append(_check(
        "git_commit recorded",
        None if n_null_commit else True,
        f"{n_null_commit}/{len(files)} artifacts have no git_commit — "
        "these carry no code provenance" if n_null_commit else "all artifacts stamped",
    ))

    n_failed = int((trials["state"] == "fail").sum())
    checks.append(_check(
        "no failed trials",
        n_failed == 0,
        f"{n_failed} failed trial(s)",
    ))

    if "empty_recommendations" in trials.columns:
        n_empty = int(trials["empty_recommendations"].fillna(False).astype(bool).sum())
        checks.append(_check(
            "no empty recommendation frames", n_empty == 0, f"{n_empty} trial(s)"
        ))

    ndcg = f"ndcg@{_objective_k()}"
    left, right = f"{MUGM_PREFIX}{ndcg}", f"{MBG_PREFIX}{ndcg}"
    if left in trials.columns and right in trials.columns:
        delta = (trials[left] - trials[right]).abs().max()
        checks.append(_check(
            "macro_by_group == macro_by_user_group_mean",
            bool(delta == 0),
            f"max |delta| = {delta:.3g} (one candidate group per user)",
        ))
    ties = trials.get(f"{MUGM_PREFIX}groups_with_score_ties")
    if ties is not None:
        checks.append(_check(
            "no score ties within a user",
            bool(ties.fillna(0).max() == 0),
            f"max groups_with_score_ties = {ties.max()} (scores are -rank)",
        ))
    n_groups, n_users = (
        trials.get(f"{MUGM_PREFIX}n_groups"), trials.get(f"{MUGM_PREFIX}n_users")
    )
    if n_groups is not None and n_users is not None:
        checks.append(_check(
            "one candidate group per user",
            bool(n_groups.eq(n_users).all()),
            "required for the *_cond columns to be exact",
        ))

    checks_frame = pd.DataFrame(checks)
    summary = files[[
        "seed", "file", "setup", "target", "n_policies",
        "generated_at", "git_commit", "optuna_version",
    ]]

    if show:
        print(f"dataset={arts.dataset}  dir={arts.artifacts_dir}")
        print(f"objective={OBJECTIVE_METRIC}  space_source={arts.space_source}")
        print(f"{len(files)} artifact(s), {len(trials)} trial rows\n")
        _display(checks_frame)
        _display(summary)
    return {"checks": checks_frame, "files": summary}


def _objective_k() -> int:
    """The k in the tuning objective, parsed from `OBJECTIVE_METRIC`."""

    return int(OBJECTIVE_METRIC.rsplit("@", 1)[-1])


# --------------------------------------------------------------------------
# Panel 2 — search coverage
# --------------------------------------------------------------------------


def panel_search_coverage(
    arts: TuningArtifacts,
    *,
    objective_column: str = "value",
    plot: bool = True,
    show: bool = True,
) -> dict[str, pd.DataFrame]:
    """How much of each search space was actually evaluated, per seed.

    `TPESampler` resamples configs it has already tried, so the trial budget
    overstates the search. Rows the sampler never reached appear blank in the
    heatmap: those cells are unknown, not bad.
    """

    complete = _complete(arts.trials, objective_column)
    rows: list[dict[str, Any]] = []
    for (seed, policy), frame in arts.trials.groupby(["seed", "policy"], sort=False):
        space = arts.search_spaces.get(str(policy))
        grid_size = space.grid_size if space is not None else float("nan")
        n_unique = frame["config_key"].nunique()
        n_trials = len(frame)
        rows.append({
            "policy": policy,
            "seed": seed,
            "n_trials": n_trials,
            "n_complete": int((frame["state"] == "complete").sum()),
            "n_unique_configs": n_unique,
            "grid_size": grid_size,
            "coverage": n_unique / grid_size if grid_size and grid_size == grid_size else np.nan,
            "n_duplicate_trials": int(frame["is_duplicate"].sum()),
            "duplicate_fraction": float(frame["is_duplicate"].mean()),
        })
    table = (
        pd.DataFrame(rows)
        .sort_values(["policy", "seed"])
        .reset_index(drop=True)
    )

    by_policy = (
        table.groupby("policy", sort=False)
        .agg(
            n_trials=("n_trials", "sum"),
            grid_size=("grid_size", "first"),
            unique_min=("n_unique_configs", "min"),
            unique_max=("n_unique_configs", "max"),
            coverage_min=("coverage", "min"),
            coverage_max=("coverage", "max"),
            duplicate_fraction=("duplicate_fraction", "mean"),
        )
        .reindex(arts.policies)
        .dropna(how="all")
        .reset_index()
    )

    if plot:
        for policy in _tuned_policies_with_data(arts):
            space = arts.search_spaces[policy]
            frame = complete[complete["policy"] == policy]
            if frame.empty:
                continue
            grid = pd.pivot_table(
                frame, index="config_key", columns="seed",
                values=objective_column, aggfunc="mean",
            )
            grid_keys = space.grid_config_keys()
            if grid_keys:
                grid = grid.reindex(grid_keys)
            grid = grid.loc[grid.mean(axis=1).sort_values(ascending=False).index]
            grid.index = [config_label(key) for key in grid.index]
            figure, ax = plt.subplots(
                figsize=(FIGSIZE[0], 0.34 * len(grid) + 1.8),
                label=f"coverage_{policy}",
            )
            sns.heatmap(
                grid, mask=grid.isna(), annot=False, cmap="viridis",
                cbar_kws={"label": objective_column}, ax=ax, linewidths=0.4,
                linecolor="white",
            )
            n_missing = int(grid.isna().sum().sum())
            ax.set_title(
                f"{arts.dataset} · {policy} — objective per config × seed\n"
                f"blank = never evaluated ({n_missing}/{grid.size} cells)"
            )
            ax.set_xlabel("split seed")
            ax.set_ylabel("")
            figure.tight_layout()

    if show:
        if arts.space_source != "live_module":
            print(
                "WARNING: grids reconstructed from observed trials — "
                "coverage below is an optimistic upper bound.\n"
            )
        _display(by_policy)
        _display(table)
    return {"by_policy": by_policy, "by_policy_seed": table}


# --------------------------------------------------------------------------
# Panel 3 — determinism
# --------------------------------------------------------------------------


def panel_determinism(
    arts: TuningArtifacts,
    *,
    objective_column: str = "value",
    show: bool = True,
) -> dict[str, pd.DataFrame]:
    """Do repeated evaluations of the same config give the same number?

    The sampler's duplicate trials are a free reproducibility test: with a
    fixed `policy_seed`, re-fitting one config within one seed must reproduce
    its objective exactly. A non-zero deviation means part of the apparent
    per-seed spread is run-to-run noise rather than data — most likely for
    GPU/torch policies.
    """

    frame = arts.trials[arts.trials["state"] == "complete"]
    grouped = frame.groupby(["policy", "seed", "config_key"], sort=False)[objective_column]
    stats = grouped.agg(["count", "min", "max"]).reset_index()
    repeated = stats[stats["count"] > 1].copy()
    repeated["deviation"] = repeated["max"] - repeated["min"]

    rows: list[dict[str, Any]] = []
    for policy in arts.policies:
        subset = repeated[repeated["policy"] == policy]
        deviation = float(subset["deviation"].max()) if not subset.empty else np.nan
        # Scale the deviation against the seed-to-seed spread of the policy's
        # own best value: a re-fit that moves the objective as much as changing
        # the data split does makes per-seed comparisons meaningless.
        best = frame[frame["policy"] == policy].groupby("seed")[objective_column].max()
        sd = float(best.std(ddof=1)) if len(best) > 1 else np.nan
        rows.append({
            "policy": policy,
            "n_repeated_configs": int(len(subset)),
            "n_repeat_trials": int((subset["count"] - 1).sum()) if not subset.empty else 0,
            "max_abs_deviation": deviation,
            "best_sd_across_seeds": sd,
            "deviation_over_sd": deviation / sd if sd and sd == sd else np.nan,
            "status": (
                "no repeats" if subset.empty
                else "PASS (bit-identical)" if deviation == 0
                else "FAIL (nondeterministic)"
            ),
        })
    summary = pd.DataFrame(rows)
    offenders = repeated[repeated["deviation"] > 0].sort_values(
        "deviation", ascending=False
    ).reset_index(drop=True)

    if show:
        _display(summary)
        if offenders.empty:
            print("All repeated configs reproduced their objective exactly.")
        else:
            print("Configs that did NOT reproduce:")
            _display(offenders)
    return {"summary": summary, "offenders": offenders}


# --------------------------------------------------------------------------
# Panel 4 — signal vs noise
# --------------------------------------------------------------------------


def panel_signal_vs_noise(
    arts: TuningArtifacts,
    *,
    objective_column: str = "value",
    plot: bool = True,
    show: bool = True,
) -> dict[str, pd.DataFrame]:
    """Is the hyperparameter effect larger than the seed-to-seed noise?

    If the spread across configs within a seed is no bigger than the spread of
    the same config across seeds, the tuner is selecting noise and any
    "stability" reading of its choices is meaningless.

    Note that `val_objective` is a maximum over trials, so it is upward-biased
    by an amount that grows with the number of unique configs evaluated — which
    varies by seed. The `best_sd` column inherits that bias; `modal_sd`, taken
    on one fixed config, does not, and is the honest noise estimate.
    """

    complete = _complete(arts.trials, objective_column)
    rows: list[dict[str, Any]] = []
    for policy in arts.policies:
        frame = complete[complete["policy"] == policy]
        if frame.empty:
            continue
        per_seed_range = frame.groupby("seed")[objective_column].agg(
            lambda values: values.max() - values.min()
        )
        best_per_seed = frame.groupby("seed")[objective_column].max()
        counts = frame["config_key"].value_counts()
        modal_key = counts.index[0]
        modal_values = frame.loc[frame["config_key"] == modal_key, objective_column]
        modal_sd = float(modal_values.std(ddof=1)) if len(modal_values) > 1 else np.nan
        mean_range = float(per_seed_range.mean())
        noise = modal_sd if modal_sd == modal_sd and modal_sd > 0 else np.nan
        ratio = mean_range / noise if noise == noise else np.nan
        rows.append({
            "policy": policy,
            "n_configs": int(frame["config_key"].nunique()),
            "mean_within_seed_range": mean_range,
            "max_within_seed_range": float(per_seed_range.max()),
            "best_sd_across_seeds": float(best_per_seed.std(ddof=1))
            if len(best_per_seed) > 1 else np.nan,
            "modal_config": config_label(modal_key),
            "modal_n_seeds": int(len(modal_values)),
            "modal_sd_across_seeds": modal_sd,
            "range_over_noise": ratio,
            "verdict": (
                "n/a (single config)" if frame["config_key"].nunique() < 2
                else "unknown" if ratio != ratio
                else "signal" if ratio >= SIGNAL_RATIO_STRONG
                else "borderline" if ratio >= SIGNAL_RATIO_WEAK
                else "NOISE — selection is a coin flip"
            ),
        })
    table = pd.DataFrame(rows)

    if plot:
        # Horizontal: config labels are long, and reading them rotated is
        # hopeless. One figure per policy so the height can follow the number
        # of configs, as in `panel_search_coverage`.
        for policy in _tuned_policies_with_data(arts):
            frame = complete[complete["policy"] == policy]
            if frame["config_key"].nunique() < 2:
                continue
            stats = frame.groupby("config_key")[objective_column].agg(["mean", "std"])
            stats = stats.sort_values("mean")
            positions = {key: i for i, key in enumerate(stats.index)}
            figure, ax = plt.subplots(
                figsize=(FIGSIZE[0], 0.32 * len(stats) + 1.8),
                label=f"signal_vs_noise_{policy}",
            )
            ax.errorbar(
                stats["mean"], np.arange(len(stats)), xerr=stats["std"],
                fmt="none", ecolor="#C44E52", elinewidth=1.4, capsize=3,
                label="±1 sd across seeds", zorder=2,
            )
            ax.scatter(
                frame[objective_column], frame["config_key"].map(positions),
                s=20, alpha=0.55, color="#4C72B0", label="one seed", zorder=3,
            )
            ax.scatter(
                stats["mean"], np.arange(len(stats)), s=42, color="#C44E52",
                marker="|", zorder=4,
            )
            ax.set_yticks(np.arange(len(stats)))
            ax.set_yticklabels([config_label(key) for key in stats.index], fontsize=7)
            ax.set_xlabel(objective_column)
            ax.set_title(
                f"{arts.dataset} · {policy} — objective per config\n"
                "error bars are seed-to-seed noise; overlap means the configs "
                "are not distinguishable"
            )
            ax.legend(fontsize=7, loc="lower right")
            figure.tight_layout()

    if show:
        _display(table)
    return {"summary": table}


# --------------------------------------------------------------------------
# Panel 5 — selection frequency and boundary checks
# --------------------------------------------------------------------------


def panel_selection_frequency(
    arts: TuningArtifacts,
    *,
    show: bool = True,
) -> dict[str, pd.DataFrame]:
    """Which value did each parameter take, and does it sit on a range edge?

    A selection on the edge of its choice list means the box may be misplaced —
    the tuner would have gone further if allowed. Deliberately pinned
    single-value parameters are flagged separately: they were not searched at
    all, so the recorded trial budget overstates the search.
    """

    rows: list[dict[str, Any]] = []
    for policy in arts.policies:
        space = arts.search_spaces.get(policy)
        if space is None or not space.tuned:
            continue
        selection = arts.selection[arts.selection["policy"] == policy]
        n_seeds = len(selection)
        for name, choices in space.choices.items():
            column = f"{PARAM_PREFIX}{name}"
            if column not in selection.columns:
                continue
            counts = selection[column].value_counts(dropna=True)
            low, high = min(choices), max(choices)
            for value, count in counts.items():
                rows.append({
                    "policy": policy,
                    "param": name,
                    "value": value,
                    "n_seeds_selected": int(count),
                    "fraction": count / n_seeds if n_seeds else np.nan,
                    "n_choices": len(choices),
                    "choices": choices,
                    "is_pinned": len(choices) == 1,
                    "is_boundary": len(choices) > 1 and value in (low, high),
                    "boundary_informative": len(choices) >= 3,
                })
    table = pd.DataFrame(rows)

    flagged = (
        table[table["is_boundary"] & table["boundary_informative"]]
        .sort_values("n_seeds_selected", ascending=False)
        .reset_index(drop=True)
        if not table.empty else table
    )
    pinned = (
        table[table["is_pinned"]][["policy", "param", "value"]].drop_duplicates()
        .reset_index(drop=True)
        if not table.empty else table
    )

    if show:
        _display(table)
        if not flagged.empty:
            print(
                "Selections on a choice-list edge — the search box may be "
                "misplaced (see the note at policy_tuning.py:77-78):"
            )
            _display(flagged)
        if not pinned.empty:
            print("Pinned to a single value, i.e. not searched at all:")
            _display(pinned)
    return {"by_value": table, "boundary": flagged, "pinned": pinned}


# --------------------------------------------------------------------------
# Panel 6 — per-seed parameter stability
# --------------------------------------------------------------------------


def panel_param_stability(
    arts: TuningArtifacts,
    *,
    plot: bool = True,
    show: bool = True,
) -> dict[str, pd.DataFrame]:
    """How much do the tuner's per-seed choices actually disagree?

    Read alongside `panel_search_coverage`: a disagreement is only evidence
    that the optimum moved if both seeds actually evaluated both cells.
    """

    rows: list[dict[str, Any]] = []
    for policy in arts.policies:
        selection = arts.selection[arts.selection["policy"] == policy]
        if selection.empty:
            continue
        counts = selection["selected_config_key"].value_counts()
        modal_key = counts.index[0]
        rows.append({
            "policy": policy,
            "n_seeds": int(len(selection)),
            "n_distinct_configs_selected": int(counts.size),
            "modal_config": config_label(modal_key),
            "n_seeds_agreeing": int(counts.iloc[0]),
            "agreement_fraction": float(counts.iloc[0] / len(selection)),
            "modal_val_objective_mean": float(
                selection.loc[
                    selection["selected_config_key"] == modal_key, "val_objective"
                ].mean()
            ),
        })
    summary = pd.DataFrame(rows)

    wide = (
        arts.selection.assign(config=arts.selection["selected_config_key"].map(config_label))
        .pivot(index="policy", columns="seed", values="config")
        .reindex(arts.policies)
        .dropna(how="all")
    )

    if plot:
        # Pinned single-value parameters carry no per-seed information; they
        # are reported by `panel_selection_frequency` instead.
        searched = {
            policy: sorted(
                name for name, choices in arts.search_spaces[policy].choices.items()
                if len(choices) > 1
            )
            for policy in _tuned_policies_with_data(arts)
        }
        policies = [policy for policy, names in searched.items() if names]
        if policies:
            ncols = max(len(searched[policy]) for policy in policies)
            figure, axes = plt.subplots(
                len(policies), ncols,
                figsize=(3.6 * ncols + 1.2, 2.7 * len(policies)),
                squeeze=False, label="param_stability", constrained_layout=True,
            )
            for row, policy in enumerate(policies):
                space = arts.search_spaces[policy]
                selection = arts.selection[arts.selection["policy"] == policy]
                # Normalise per policy: a shared scale spanning Random to
                # LightGCN would flatten every within-policy difference.
                objective = selection["val_objective"].astype(float)
                norm = plt.Normalize(float(objective.min()), float(objective.max()))
                names = searched[policy]
                for col in range(ncols):
                    ax = axes[row][col]
                    if col >= len(names):
                        ax.set_visible(False)
                        continue
                    name = names[col]
                    choices = list(space.choices[name])
                    ticks = {value: i for i, value in enumerate(choices)}
                    y = selection[f"{PARAM_PREFIX}{name}"].map(ticks)
                    ax.scatter(
                        selection["seed"], y, c=objective,
                        cmap="viridis", norm=norm, s=70, edgecolor="white", zorder=3,
                    )
                    ax.set_yticks(range(len(choices)))
                    ax.set_yticklabels(choices, fontsize=7)
                    ax.set_ylim(-0.6, len(choices) - 0.4)
                    ax.set_xticks(arts.seeds)
                    ax.tick_params(axis="x", labelsize=7)
                    ax.set_title(name, fontsize=9)
                    if row == len(policies) - 1:
                        ax.set_xlabel("split seed", fontsize=8)
                axes[row][0].set_ylabel(policy, fontsize=9, labelpad=26)
                figure.colorbar(
                    plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
                    ax=axes[row].tolist(), shrink=0.85, pad=0.01,
                    label="val_objective",
                )
            figure.suptitle(
                f"{arts.dataset} — selected value per seed (full choice list on "
                "each y-axis; edges are boundaries, colour scaled per policy)"
            )

    if show:
        _display(summary)
        _display(wide)
    return {"summary": summary, "selected_by_seed": wide}


# --------------------------------------------------------------------------
# Panel 7 — one shared config across seeds
# --------------------------------------------------------------------------


def panel_shared_config(
    arts: TuningArtifacts,
    *,
    min_seeds: int | None = None,
    objective_column: str = "value",
    plot: bool = True,
    show: bool = True,
) -> dict[str, Any]:
    """Pick one config per policy to use for every seed, and price the choice.

    Selection is by **mean rank across seeds**, not mean objective: seeds sit
    at different overall levels, and ranks are invariant to that shift. Ranks
    are taken against *every* config evaluated in that seed, so they stay
    comparable no matter which configs clear the eligibility bar.

    Eligibility needs care. Uneven TPE coverage means each config's mean is
    taken over a different subset of seeds, so a config sampled once on a
    favourable seed is not comparable to one measured everywhere. But
    demanding full coverage is worse: on these artifacts BPR's two best
    configs appear in 9 of 10 seeds while a config 0.04 worse appears in all
    10, so `min_seeds=n_seeds` would select the bad one. The default is
    therefore a simple majority, and `n_seeds_covered` is reported for the
    winner and every runner-up so the trade-off stays visible.

    `regret` is measured against the best config *actually evaluated* in that
    seed, so it is the honest price of freezing one config versus tuning per
    seed. Compare it to `sd_across_seeds`: regret inside the noise band costs
    nothing real.
    """

    min_seeds = max(2, int(np.ceil(arts.n_seeds / 2))) if min_seeds is None else int(min_seeds)
    complete = _complete(arts.trials, objective_column)

    summary_rows: list[dict[str, Any]] = []
    regret_rows: list[dict[str, Any]] = []
    candidate_frames: list[pd.DataFrame] = []
    chosen_params: dict[str, dict[str, Any]] = {}

    for policy in arts.policies:
        frame = complete[complete["policy"] == policy]
        if frame.empty:
            continue
        matrix = pd.pivot_table(
            frame, index="config_key", columns="seed",
            values=objective_column, aggfunc="mean",
        )
        # Rank against everything tried in that seed, then restrict to configs
        # measured widely enough for the average to mean anything.
        ranks = matrix.rank(axis=0, ascending=False, method="min")
        seed_coverage = matrix.notna().sum(axis=1)
        candidates = pd.DataFrame({
            "policy": policy,
            "config": [config_label(key) for key in matrix.index],
            "n_seeds_covered": seed_coverage,
            "mean_rank": ranks.mean(axis=1),
            "mean_value": matrix.mean(axis=1),
        })
        candidates["eligible"] = candidates["n_seeds_covered"] >= min_seeds
        relaxed = not candidates["eligible"].any()
        if relaxed:
            candidates["eligible"] = (
                candidates["n_seeds_covered"] == candidates["n_seeds_covered"].max()
            )
        candidates = candidates.sort_values(
            ["eligible", "mean_rank", "mean_value"], ascending=[False, True, False]
        )
        candidate_frames.append(candidates.reset_index(names="config_key"))

        chosen_key = str(candidates.index[0])
        chosen_params[policy] = json.loads(chosen_key)
        ordered = candidates

        best_per_seed = matrix.max(axis=0)
        chosen_per_seed = matrix.loc[chosen_key]
        regret = best_per_seed - chosen_per_seed
        for seed in matrix.columns:
            regret_rows.append({
                "policy": policy,
                "seed": seed,
                "chosen_value": chosen_per_seed.get(seed, np.nan),
                "best_value": best_per_seed.get(seed, np.nan),
                "regret": regret.get(seed, np.nan),
            })

        modal_sd = (
            float(matrix.loc[chosen_key].std(ddof=1))
            if chosen_per_seed.notna().sum() > 1 else np.nan
        )
        tuner_choices = arts.selection.loc[
            arts.selection["policy"] == policy, "selected_config_key"
        ]
        summary_rows.append({
            "policy": policy,
            "chosen_config": config_label(chosen_key),
            "chosen_params": chosen_params[policy],
            "n_eligible_configs": int(candidates["eligible"].sum()),
            "n_configs_seen": int(len(matrix)),
            "eligibility_relaxed": relaxed,
            "seeds_covered": int(chosen_per_seed.notna().sum()),
            "mean_rank": float(ordered["mean_rank"].iloc[0]),
            "mean_value": float(ordered["mean_value"].iloc[0]),
            "mean_regret": float(np.nanmean(regret.to_numpy(dtype=float))),
            "max_regret": float(np.nanmax(regret.to_numpy(dtype=float))),
            "sd_across_seeds": modal_sd,
            "regret_within_noise": bool(
                np.nanmax(regret.to_numpy(dtype=float)) <= modal_sd
            ) if modal_sd == modal_sd else None,
            "matches_tuner_choice_in_seeds": int((tuner_choices == chosen_key).sum()),
        })

    summary = pd.DataFrame(summary_rows)
    regret_table = pd.DataFrame(regret_rows)
    candidates_table = (
        pd.concat(candidate_frames, ignore_index=True)
        if candidate_frames else pd.DataFrame()
    )

    if plot and not regret_table.empty:
        policies = [p for p in arts.policies if p in set(regret_table["policy"])]
        figure, axes = _subplot_grid(len(policies))
        figure.set_label("shared_config_regret")
        for ax, policy in zip(axes, policies, strict=True):
            frame = regret_table[regret_table["policy"] == policy]
            ax.bar(frame["seed"], frame["regret"], color="#4C72B0", width=0.6)
            sd = summary.loc[summary["policy"] == policy, "sd_across_seeds"].iloc[0]
            if sd == sd:
                ax.axhspan(0, sd, color="#C44E52", alpha=0.15, label="±1 sd (seed noise)")
                ax.legend(fontsize=7)
            ax.set_title(policy)
            ax.set_xlabel("seed")
            ax.set_ylabel("regret vs best-in-seed")
            ax.set_xticks(arts.seeds)
        figure.suptitle(
            f"{arts.dataset} — cost of freezing one config per policy", y=1.0
        )
        figure.tight_layout()

    if show:
        print(
            f"Eligibility: config must be evaluated in >= {min_seeds} of "
            f"{arts.n_seeds} seeds. Aggregation: mean rank across all configs "
            "tried in each seed.\n"
            "Because TPE coverage is uneven, each config's mean is taken over "
            "a different subset of seeds — read `n_seeds_covered` alongside "
            "`mean_value` before trusting a pick.\n"
        )
        _display(summary.drop(columns=["chosen_params"]))
        if not summary.empty and summary["eligibility_relaxed"].any():
            print(
                "WARNING: no config met the coverage bar for "
                f"{summary.loc[summary['eligibility_relaxed'], 'policy'].tolist()}; "
                "fell back to the best-covered configs. Treat those picks as weak."
            )
        print("\nCandidates per policy (chosen config first):")
        _display(candidates_table)
        print("\nChosen configs (shape matches `selected_params`):")
        print(json.dumps(chosen_params, indent=2, sort_keys=True))
    return {
        "summary": summary,
        "regret": regret_table,
        "candidates": candidates_table,
        "chosen_params": chosen_params,
    }


# --------------------------------------------------------------------------
# Panel 8 — policy ordering stability
# --------------------------------------------------------------------------


def panel_policy_ordering(
    arts: TuningArtifacts,
    *,
    objective_column: str = "value",
    plot: bool = True,
    show: bool = True,
) -> dict[str, pd.DataFrame]:
    """Is the ordering of policies stable across seeds?

    This is the input the policy-ranking-agreement experiment is built on: a
    simulator is scored on reproducing the ordering of policies under real
    data. Where two policies are tied within seed noise, their relative order
    is a coin flip, and any rank correlation computed over them is partly
    measuring that noise rather than simulator quality.

    Caveat: this is the ordering on the tuner's *inner validation* split, not
    the outer test split the experiment reports. A tie here is a strong hint of
    a tie there, not proof of one.
    """

    complete = _complete(arts.trials, objective_column)
    utility = (
        complete.groupby(["seed", "policy"])[objective_column].max().unstack()
        .reindex(columns=[p for p in arts.policies])
        .dropna(axis=1, how="all")
    )
    n_policies = utility.shape[1]

    ordering = pd.DataFrame({
        "seed": utility.index,
        "ordering": [
            " > ".join(row.sort_values(ascending=False).index.str.replace("Policy", ""))
            for _, row in utility.iterrows()
        ],
    }).reset_index(drop=True)

    pair_rows: list[dict[str, Any]] = []
    for left, right in combinations(utility.columns, 2):
        gap = utility[left] - utility[right]
        sd = float(gap.std(ddof=1)) if len(gap) > 1 else np.nan
        mean_gap = float(gap.mean())
        pair_rows.append({
            "policy_a": left,
            "policy_b": right,
            "mean_gap": mean_gap,
            "sd_of_paired_gap": sd,
            "abs_gap_over_sd": abs(mean_gap) / sd if sd and sd == sd else np.nan,
            "seeds_a_beats_b": int((gap > 0).sum()),
            "n_seeds": int(gap.notna().sum()),
            "tied": bool(abs(mean_gap) <= sd) if sd == sd else None,
        })
    pairs = (
        pd.DataFrame(pair_rows).sort_values("abs_gap_over_sd").reset_index(drop=True)
        if pair_rows
        else pd.DataFrame(columns=[
            "policy_a", "policy_b", "mean_gap", "sd_of_paired_gap",
            "abs_gap_over_sd", "seeds_a_beats_b", "n_seeds", "tied",
        ])
    )

    tau = pd.DataFrame(index=utility.index, columns=utility.index, dtype=float)
    if n_policies >= 3:
        from scipy.stats import kendalltau

        for left in utility.index:
            for right in utility.index:
                tau.loc[left, right] = float(
                    kendalltau(utility.loc[left], utility.loc[right]).statistic
                )

    if plot:
        figure, axes = plt.subplots(
            1, 2 if n_policies >= 3 else 1,
            figsize=FIGSIZE, squeeze=False, label="policy_ordering",
        )
        ax = axes[0][0]
        for policy in utility.columns:
            ax.plot(utility.index, utility[policy], marker="o", ms=4, label=policy)
        ax.set_xlabel("split seed")
        ax.set_ylabel(objective_column)
        ax.set_xticks(arts.seeds)
        ax.set_title("Policy utility per seed")
        ax.legend(fontsize=7)
        if n_policies >= 3:
            sns.heatmap(
                tau.astype(float), vmin=-1, vmax=1, cmap="RdBu_r", center=0,
                annot=True, fmt=".2f", annot_kws={"size": 6},
                cbar_kws={"label": "Kendall τ"}, ax=axes[0][1],
            )
            axes[0][1].set_title("Ordering agreement between seeds")
        figure.suptitle(f"{arts.dataset} — policy ordering stability", y=1.02)
        figure.tight_layout()

    if show:
        if n_policies < 3:
            print(
                f"Only {n_policies} policy/policies present — Kendall τ is "
                "undefined below 3; showing utilities and pairwise gaps only."
            )
        _display(ordering)
        _display(pairs)
        tied = pairs[pairs["tied"].fillna(False).astype(bool)] if not pairs.empty else pairs
        if not tied.empty:
            print(
                "Pairs tied within seed noise — their relative order is not a "
                "reliable reference for ranking agreement:"
            )
            _display(tied[["policy_a", "policy_b", "mean_gap",
                           "sd_of_paired_gap", "seeds_a_beats_b", "n_seeds"]])
    return {
        "utility": utility,
        "ordering": ordering,
        "pairs": pairs,
        "kendall_tau": tau,
    }


# --------------------------------------------------------------------------
# Panel 9 — evaluation population
# --------------------------------------------------------------------------


def panel_eval_population(
    arts: TuningArtifacts,
    *,
    plot: bool = True,
    show: bool = True,
) -> dict[str, pd.DataFrame]:
    """Are all policies scored on the same users?

    `n_eval_users` is the size of the validation-user frame; `n_users` is how
    many users actually received recommendations. Policies that cannot serve
    users unseen in inner train are averaged over a strict subset of the users
    the baselines are averaged over, so their objectives are not computed on
    identical populations.
    """

    selected = arts.trials[arts.trials["is_selected"] & (arts.trials["state"] == "complete")]
    selected = selected[~selected["is_duplicate"]]
    columns = {
        "n_users": "n_users",
        f"{MUGM_PREFIX}n_groups": "n_groups",
        f"{MUGM_PREFIX}n_positive": "n_positive",
        f"{MUGM_PREFIX}groups_without_positive": "groups_without_positive",
        "n_recommendations": "n_recommendations",
    }
    keep = ["seed", "policy", "n_eval_users"] + [c for c in columns if c in selected.columns]
    table = selected[keep].rename(columns=columns).copy()
    table["users_not_served"] = table["n_eval_users"] - table["n_users"]
    table["frac_without_positive"] = (
        table["groups_without_positive"] / table["n_groups"]
    )
    table = table.sort_values(["policy", "seed"]).reset_index(drop=True)

    per_seed = table.groupby("seed")["n_users"].nunique()
    mismatched = per_seed[per_seed > 1]
    populations = (
        table.groupby("policy")["n_users"]
        .agg(["min", "max"]).reindex(arts.policies).dropna(how="all").reset_index()
    )

    if plot and not table.empty:
        figure, axes = plt.subplots(1, 2, figsize=FIGSIZE, label="eval_population")
        sns.barplot(
            table, x="policy", y="users_not_served", errorbar="sd", ax=axes[0],
            color="#4C72B0",
        )
        axes[0].set_title("Validation users receiving no recommendation")
        axes[0].tick_params(axis="x", rotation=90, labelsize=7)
        sns.barplot(
            table, x="policy", y="frac_without_positive", errorbar="sd", ax=axes[1],
            color="#C44E52",
        )
        axes[1].set_title("Users with no positive in the top-k (score exactly 0)")
        axes[1].tick_params(axis="x", rotation=90, labelsize=7)
        figure.suptitle(f"{arts.dataset} — evaluation population", y=1.03)
        figure.tight_layout()

    if show:
        _display(populations)
        if not mismatched.empty:
            print(
                "Policies are NOT scored on the same user set in seed(s) "
                f"{mismatched.index.tolist()} — objectives are means over "
                "different populations:"
            )
        _display(table)
    return {"by_policy_seed": table, "populations": populations}


# --------------------------------------------------------------------------
# Panel 10 — metric decomposition
# --------------------------------------------------------------------------


def panel_metric_decomposition(
    arts: TuningArtifacts,
    *,
    k: int | None = None,
    plot: bool = True,
    show: bool = True,
) -> dict[str, pd.DataFrame]:
    """Split the objective into "finds positives" and "orders them well".

    `groups_with_size_lte@k == n_groups`: k equals the recommendation list
    length, so this nDCG has no retrieval component beyond what the policy put
    in the list. Two configs can tie on it while differing in how many
    positives they surfaced. The `_cond` columns restrict the mean to users who
    had at least one positive, removing the large exact-zero mass.
    """

    k = _objective_k() if k is None else int(k)
    ndcg = f"{MUGM_PREFIX}ndcg@{k}"
    hit = f"{MUGM_PREFIX}hit_rate@{k}"
    cond = f"{ndcg}_cond"
    complete = _complete(arts.trials)

    selected = complete[complete["is_selected"]]
    available = [c for c in (ndcg, cond, hit, f"{hit}_cond",
                             f"{MUGM_PREFIX}n_positive") if c in selected.columns]
    summary = (
        selected.groupby("policy")[available].mean()
        .reindex(arts.policies).dropna(how="all")
    )
    if f"{MUGM_PREFIX}groups_without_positive" in selected.columns:
        summary["frac_users_without_positive"] = (
            selected.groupby("policy")
            .apply(
                lambda frame: (
                    frame[f"{MUGM_PREFIX}groups_without_positive"]
                    / frame[f"{MUGM_PREFIX}n_groups"]
                ).mean(),
                include_groups=False,
            )
            .reindex(summary.index)
        )
    summary = summary.reset_index()

    if plot and ndcg in complete.columns:
        figure, axes = plt.subplots(1, 2, figsize=FIGSIZE, label="metric_decomposition")
        positives = f"{MUGM_PREFIX}n_positive"
        if positives in complete.columns:
            sns.scatterplot(
                complete, x=positives, y=ndcg, hue="policy", s=22, alpha=0.75,
                ax=axes[0],
            )
            axes[0].set_title("Positives surfaced vs objective")
            axes[0].legend(fontsize=6)
        if hit in complete.columns:
            sns.scatterplot(
                complete, x=hit, y=ndcg, hue="policy", s=22, alpha=0.75,
                ax=axes[1], legend=False,
            )
            axes[1].set_title(f"hit_rate@{k} vs ndcg@{k}")
        figure.suptitle(
            f"{arts.dataset} — one point per (config, seed)", y=1.03
        )
        figure.tight_layout()

    if show:
        _display(summary)
    return {"selected_configs": summary}


# --------------------------------------------------------------------------
# Panel 11 — cost
# --------------------------------------------------------------------------


def panel_cost(
    arts: TuningArtifacts,
    *,
    plot: bool = True,
    show: bool = True,
) -> dict[str, pd.DataFrame]:
    """Where did the tuning budget go, and how much of it was spent twice?"""

    column = "fit_recommend_seconds"
    if column not in arts.trials.columns:
        return {"by_policy": pd.DataFrame()}
    frame = arts.trials[arts.trials["state"] == "complete"]
    table = (
        frame.groupby("policy")
        .agg(
            n_trials=(column, "size"),
            total_seconds=(column, "sum"),
            median_seconds=(column, "median"),
            max_seconds=(column, "max"),
        )
        .reindex(arts.policies).dropna(how="all")
    )
    duplicated = frame[frame["is_duplicate"]].groupby("policy")[column].sum()
    table["duplicate_seconds"] = duplicated.reindex(table.index).fillna(0.0)
    table["duplicate_fraction"] = table["duplicate_seconds"] / table["total_seconds"]
    table = table.reset_index()

    if plot and not frame.empty:
        figure, ax = plt.subplots(figsize=FIGSIZE, label="cost")
        sns.stripplot(frame, x="policy", y=column, hue="policy", legend=False,
                      alpha=0.6, ax=ax)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.set_title(f"{arts.dataset} — fit + recommend time per trial")
        figure.tight_layout()

    if show:
        total = float(table["total_seconds"].sum())
        wasted = float(table["duplicate_seconds"].sum())
        print(
            f"Total tuning compute: {total / 3600:.2f} h; "
            f"spent re-evaluating configs already tried: {wasted / 3600:.2f} h "
            f"({wasted / total:.0%})" if total else "no timing recorded"
        )
        _display(table)
    return {"by_policy": table}
