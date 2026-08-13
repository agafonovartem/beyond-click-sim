"""Offline hyperparameter tuning for recommender policies via Optuna.

Protocol: for each (dataset, target_source_column, split_seed), a validation
split is carved from the *outer train* frame of the policy-ranking task
(test data is never passed in). Each tunable policy is searched with a seeded
Optuna GridSampler over its categorical space; every trial — not just the
winner — is recorded with its full metric payload so the search can be
audited. The resulting artifact is consumed by the policy-ranking task
builders via `build_tuned_policy_roster`.

Artifacts are strictly validated on load: any change to the dataset, filter,
splitter, target, search spaces, or split seed invalidates the artifact and
forces a re-tune.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import optuna
import pandas as pd
from loguru import logger

from beyond_click_sim.evaluation.policy_ranking import (
    evaluate_policy_recommendations,
)
from beyond_click_sim.tasks.policy_ranking import POLICY_EVAL_KS
from beyond_click_sim.tasks.policies import (
    ALSPolicy,
    BPRPolicy,
    ItemKNNPolicy,
    Policy,
    PopularityPolicy,
    RandomPolicy,
)
from beyond_click_sim.tasks.policies_torch import LightGCNTorchPolicy
from beyond_click_sim.tasks.splitters import RandomFractionSplitter

ARTIFACT_SCHEMA_VERSION = 1
OBJECTIVE_METRIC = "macro_by_user_group_mean.ndcg@10"
OBJECTIVE_K = 10
INNER_VAL_FRACTION_OF_TRAIN = 0.25
INNER_SEED_OFFSET = 1000
SAMPLER_SEED_OFFSET = 2000

# Fixed roster order shared by tuning artifacts and task builders.
#
# LightGCN runs on the PyTorch/PyG backend, which implements the paper's
# mini-batch Adam protocol. The pure-NumPy `LightGCNPolicy` is kept in
# `policies.py` as a reference implementation and fallback. To revert: import
# `LightGCNPolicy` from `.policies`, put it back in this tuple, and rename the
# matching SEARCH_SPACES key. Both classes take the same constructor
# parameters (torch adds `device`), but note that `iterations` counts
# mini-batch *epochs* on torch and single full-batch updates on NumPy — the
# values are not comparable.
POLICY_ROSTER: tuple[type[Policy], ...] = (
    RandomPolicy,
    PopularityPolicy,
    ItemKNNPolicy,
    ALSPolicy,
    BPRPolicy,
    LightGCNTorchPolicy,
)
POLICY_CLASSES: dict[str, type[Policy]] = {
    cls.__name__: cls for cls in POLICY_ROSTER
}

# Declarative Optuna search spaces. Serializable by construction: the canonical
# JSON of this dict is hashed into the artifact identity, so any space or
# budget change invalidates existing artifacts.
#
# Box placement for ItemKNN/ALS/BPR comes from an exhaustive sweep on the ml-1m
# inner validation split (target_interact, split_seed 0) — the same split and
# objective the tuner uses. Values sit inside the measured optimum where one was
# found, so a selection landing on a range edge is a signal the box is wrong.
# The sweep is ml-1m only; re-check on steam before treating these as final.
SEARCH_SPACES: dict[str, dict[str, Any]] = {
    # Interior optimum at 20 (0.5358); 50/100/200 fall off monotonically to
    # 0.504, so the original bracket was already right.
    "ItemKNNPolicy": {
        "n_trials": 4,
        "params": {
            "n_neighbors": {"kind": "categorical", "choices": [10, 20, 50, 100]},
        },
    },
    # alpha dominates (0.497 -> 0.568 across the old {1,10,40} grid) while
    # regularization moved ndcg by ~0.002 across two decades, so the budget
    # moves off reg and onto a finer alpha bracket around 10. Capacity is
    # monotonically decreasing (32 > 64 > 128 > 256), hence the extension down
    # to 16 rather than up. iterations stays fixed: 20 -> 50 bought +0.002.
    "ALSPolicy": {
        "n_trials": 24,
        "params": {
            "n_factors": {"kind": "categorical", "choices": [16, 32, 64]},
            "regularization": {"kind": "categorical", "choices": [0.01, 0.1]},
            "alpha": {"kind": "categorical", "choices": [2.0, 5.0, 10.0, 20.0]},
        },
    },
    # learning_rate and iterations are coupled and must be searched jointly:
    # the old box (lr >= 0.005, iterations pinned at 100) capped BPR at 0.476,
    # while (lr=0.001, iterations=300) reaches 0.508 and lr=0.002 alone reaches
    # 0.498. More iterations is not uniformly better — 600 regresses to 0.465 —
    # so both ends of the iterations grid are informative. regularization is
    # pinned at the value that won at every (n_factors, lr) probed; 0.01 and
    # 0.1 cost 0.014 and 0.073 respectively.
    "BPRPolicy": {
        "n_trials": 6,
        "params": {
            "n_factors": {"kind": "categorical", "choices": [32, 64]},
            "learning_rate": {
                "kind": "categorical",
                "choices": [0.001, 0.002, 0.005],
            },
            # "iterations": {"kind": "categorical", "choices": [100, 300]},
            "iterations": {"kind": "categorical", "choices": [300]},
            "regularization": {"kind": "categorical", "choices": [0.001]},
        },
    },
    # PyTorch/PyG backend running the paper's mini-batch Adam protocol, so
    # `iterations` counts epochs (~292 optimiser steps each on ml-1m), not
    # full-batch updates. The box follows the two published references:
    #
    #   He et al. (SIGIR 2020) Sec 4.1.2 — searches lambda over
    #     {1e-6 ... 1e-2} (optimum "in most cases" 1e-4) and K over 1..4
    #     (K=3 best); fixes lr=0.001, batch 1024-2048, dim 64, ~1000 epochs.
    #   RecBole's published LightGCN tuning range — reg_weight
    #     {1e-5, 1e-4, 1e-3, 1e-2}, n_layers {1,2,3,4},
    #     learning_rate {1e-2, 5e-3, 1e-3, 5e-4, 1e-4}.
    #
    # `regularization` here is the reference implementation's `decay`, applied
    # to the layer-0 embeddings of each batch triple over a mean BPR loss, so
    # it is directly comparable to both references' lambda.
    #
    # Deviations from those ranges, and why:
    # - K restricted to {2, 3}: the paper reports K=3 best and RecBole defaults
    #   to 2; K=1 underfits and K=4 oversmooths. Spending budget on the two
    #   plausible values instead of four.
    # - lambda restricted to {1e-4, 1e-3, 1e-2}: 1e-5 is effectively
    #   unregularised, and ml-1m is far denser than the paper's Gowalla /
    #   Yelp2018 / Amazon-Book, so the overfitting risk sits at the upper end.
    # - lr fixed at 0.001, the default in both the paper and the reference
    #   implementation. RecBole does suggest tuning it; this is a deliberate
    #   budget trade so `iterations` can be searched instead. Revisit if a
    #   selected config lands on a range edge.
    # - `iterations` searched, which neither reference does: they early-stop on
    #   a validation split, and `Policy.fit` has no val data to do that with.
    #   Searching it on the inner split is the equivalent protocol here.
    #
    # 6 configs x 12 tuning jobs = 72 full-length fits. Benchmark one fit on
    # the GPU pod before launching the sweep.
    "LightGCNTorchPolicy": {
        "n_trials": 6,
        "params": {
            "n_layers": {"kind": "categorical", "choices": [2, 3]},
            "regularization": {
                "kind": "categorical",
                "choices": [1e-4, 1e-3, 1e-2],
            },
            # "iterations": {"kind": "categorical", "choices": [300, 1000]},
            "iterations": {"kind": "categorical", "choices": [300]},
        },
    },
}


def search_spaces_sha256(search_spaces: dict[str, Any]) -> str:
    """Fingerprint of the search spaces + budgets for staleness checks."""

    raw = json.dumps(search_spaces, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def artifact_filename(
    dataset_name: str,
    target_source_column: str,
    split_seed: int,
) -> str:
    return f"{dataset_name}_{target_source_column}_seed{split_seed}.json"


def default_artifacts_dir() -> Path:
    """Default tuning-artifact location: `<repo root>/artifacts/`.

    Single source of truth shared by the tuning runner (writer) and the
    policy-ranking task builders (readers).
    """

    here = Path(__file__).resolve()
    for path in [here, *here.parents]:
        pyproject = path / "pyproject.toml"
        if pyproject.exists() and 'name = "beyond-click-sim"' in pyproject.read_text(
            encoding="utf-8"
        ):
            return path / "artifacts"
    raise RuntimeError("Could not find beyond-click-sim repo root")


def _grid_search_space(spec: dict[str, Any]) -> dict[str, list[Any]]:
    """Build an Optuna GridSampler search_space from a categorical SEARCH_SPACES spec."""

    grid: dict[str, list[Any]] = {}
    for name, param_spec in spec["params"].items():
        if param_spec["kind"] != "categorical":
            raise ValueError(
                f"GridSampler only supports categorical params; got {name}="
                f"{param_spec['kind']!r}"
            )
        grid[name] = list(param_spec["choices"])
    return grid


def _grid_n_trials(grid: dict[str, list[Any]]) -> int:
    n_trials = 1
    for values in grid.values():
        n_trials *= len(values)
    return n_trials


def _suggest_params(trial: Any, spec: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, param_spec in spec["params"].items():
        kind = param_spec["kind"]
        if kind == "categorical":
            params[name] = trial.suggest_categorical(name, param_spec["choices"])
        elif kind == "float":
            params[name] = trial.suggest_float(
                name,
                param_spec["low"],
                param_spec["high"],
                log=param_spec.get("log", False),
            )
        elif kind == "int":
            params[name] = trial.suggest_int(
                name,
                param_spec["low"],
                param_spec["high"],
                log=param_spec.get("log", False),
            )
        else:
            raise ValueError(f"Unknown search-space param kind: {kind!r}")
    return params


def _objective_value(metric: dict[str, Any]) -> float:
    value = metric["ranking"]["macro_by_user_group_mean"][f"ndcg@{OBJECTIVE_K}"]
    return float(value)


def _evaluate_once(
    policy: Policy,
    *,
    inner_train: pd.DataFrame,
    val_users: pd.DataFrame,
    val_targets: dict[tuple[Any, Any], Any],
    items: pd.DataFrame,
    user_column: str,
    item_column: str,
    policy_k: int,
) -> tuple[float, dict[str, Any]]:
    """Fit on inner train and score on inner val. Returns (objective, payload)."""

    t0 = perf_counter()
    policy.fit(inner_train, items=items, user_column=user_column, item_column=item_column)
    recs = policy.recommend(
        val_users,
        train_interactions=inner_train,
        items=items,
        user_column=user_column,
        item_column=item_column,
    )
    fit_recommend_seconds = perf_counter() - t0
    if len(recs) == 0:
        return 0.0, {
            "mean_hit_rate": 0.0,
            "n_users": 0,
            "n_recommendations": 0,
            "fit_recommend_seconds": fit_recommend_seconds,
            "empty_recommendations": True,
        }
    recs = recs.copy()
    recs["target"] = [
        val_targets.get((user_id, item_id), 0)
        for user_id, item_id in zip(recs[user_column], recs[item_column])
    ]
    metric = evaluate_policy_recommendations(
        recs,
        targets=recs["target"],
        user_column=user_column,
        policy_name=policy.name,
        k=policy_k,
        ks=POLICY_EVAL_KS,
        fit_recommend_seconds=fit_recommend_seconds,
    )
    payload = {
        "mean_hit_rate": metric["mean_hit_rate"],
        "ranking": metric["ranking"],
        "n_users": metric["n_users"],
        "n_recommendations": metric["n_recommendations"],
        "fit_recommend_seconds": metric["fit_recommend_seconds"],
    }
    return _objective_value(metric), payload


def tune_policies_for_split(
    *,
    train_interactions: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
    dataset_name: str,
    dataset_version: str,
    target_source_column: str,
    positive_feedback_column: str | None,
    split_seed: int,
    policy_k: int = 10,
    policy_seed: int = 0,
    filter_config: dict[str, Any],
    splitter_config: dict[str, Any],
    search_spaces: dict[str, Any] | None = None,
    policy_classes: tuple[type[Policy], ...] | None = None,
    user_column: str = "user_id",
    item_column: str = "item_id",
) -> dict[str, Any]:
    """Grid-search tunable policies on an inner split carved from outer train.

    `train_interactions` must be the task's outer train frame only; test data
    is structurally excluded from this function. Returns the full artifact
    payload (identity + per-policy selection + complete trial logs).
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if search_spaces is None:
        search_spaces = SEARCH_SPACES
    classes = POLICY_ROSTER if policy_classes is None else policy_classes
    # A key that matches no policy class in the roster would silently leave
    # that policy untuned (see the `spec is None` branch below) while the
    # artifact still validates on load.
    # `classes` may be a subset of the roster (runner's --policies flag) or a
    # set of injected classes, so accept either source of names.
    class_names = set(POLICY_CLASSES) | {cls.__name__ for cls in classes}
    assert set(search_spaces) <= class_names, (
        "Unknown search-space keys (no matching policy class): "
        f"{sorted(set(search_spaces) - class_names)}"
    )

    inner = RandomFractionSplitter(
        train_fraction=1.0 - INNER_VAL_FRACTION_OF_TRAIN,
        val_fraction=INNER_VAL_FRACTION_OF_TRAIN,
        test_fraction=0.0,
        seed=split_seed + INNER_SEED_OFFSET,
    ).split(train_interactions)
    inner_train, inner_val = inner.train, inner.val

    val_users = users[users[user_column].isin(inner_val[user_column].unique())].copy()
    val_targets: dict[tuple[Any, Any], Any] = {}
    for row in inner_val.itertuples(index=False):
        key = (getattr(row, user_column), getattr(row, item_column))
        val_targets[key] = getattr(row, target_source_column)

    sampler_seed = split_seed + SAMPLER_SEED_OFFSET
    policies_result: dict[str, Any] = {}
    for cls in classes:
        name = cls.__name__
        spec = search_spaces.get(name)
        logger.info(f"Tuning {name} with {spec}")

        def _make_policy(params: dict[str, Any]) -> Policy:
            return cls(
                k=policy_k,
                seed=policy_seed,
                positive_feedback_column=positive_feedback_column,
                **params,
            )

        if spec is None:
            value, payload = _evaluate_once(
                _make_policy({}),
                inner_train=inner_train,
                val_users=val_users,
                val_targets=val_targets,
                items=items,
                user_column=user_column,
                item_column=item_column,
                policy_k=policy_k,
            )
            policies_result[name] = {
                "tuned": False,
                "selected_params": {},
                "val_objective": value,
                "n_eval_users": int(len(val_users)),
                "trials": [
                    {"number": 0, "state": "complete", "params": {},
                     "value": value, "metrics": payload}
                ],
            }
            continue

        grid = _grid_search_space(spec)
        grid_size = _grid_n_trials(grid)
        n_trials = int(spec["n_trials"])
        # Allow n_trials > grid_size so failed points can be retried; forbid a
        # budget smaller than the grid (that would make selection non-exhaustive).
        if n_trials < grid_size:
            raise ValueError(
                f"SEARCH_SPACES[{name!r}] n_trials={n_trials} is smaller than "
                f"categorical grid size {grid_size}. Raise n_trials or shrink "
                "the choices lists."
            )
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.GridSampler(grid, seed=sampler_seed),
        )
        trial_log: list[dict[str, Any]] = []

        def objective(trial: Any) -> float:
            params = _suggest_params(trial, spec)
            try:
                value, payload = _evaluate_once(
                    _make_policy(params),
                    inner_train=inner_train,
                    val_users=val_users,
                    val_targets=val_targets,
                    items=items,
                    user_column=user_column,
                    item_column=item_column,
                    policy_k=policy_k,
                )
            except Exception as error:
                trial_log.append({
                    "number": trial.number,
                    "state": "fail",
                    "params": params,
                    "error": repr(error),
                })
                raise
            trial_log.append({
                "number": trial.number,
                "state": "complete",
                "params": params,
                "value": value,
                "metrics": payload,
            })
            return value

        study.optimize(
            objective,
            n_trials=n_trials,
            catch=(Exception,),
            show_progress_bar=True,
        )
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            raise RuntimeError(
                f"All {n_trials} tuning trials failed for {name} "
                f"(dataset={dataset_name}, target={target_source_column}, "
                f"seed={split_seed}); refusing to write a tuning artifact. "
                f"Trial errors: {[t['error'] for t in trial_log][:3]}"
            )
        policies_result[name] = {
            "tuned": True,
            "selected_params": dict(study.best_params),
            "val_objective": float(study.best_value),
            "n_eval_users": int(len(val_users)),
            "sampler": {"name": "GridSampler", "seed": sampler_seed},
            "n_trials": n_trials,
            "trials": trial_log,
        }

    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "identity": {
            "dataset": dataset_name,
            "dataset_version": dataset_version,
            "target_source_column": target_source_column,
            "positive_feedback_column": positive_feedback_column,
            "split_seed": split_seed,
            "policy_k": policy_k,
            "policy_seed": policy_seed,
            "objective": OBJECTIVE_METRIC,
            "filter": filter_config,
            "splitter": splitter_config,
            "inner_split": {
                "val_fraction_of_train": INNER_VAL_FRACTION_OF_TRAIN,
                "seed_offset": INNER_SEED_OFFSET,
                "group_column": "user_id",
            },
            "search_spaces_sha256": search_spaces_sha256(search_spaces),
        },
        "policies": policies_result,
    }


def load_tuned_artifact(
    artifacts_dir: Path,
    *,
    dataset_name: str,
    dataset_version: str,
    target_source_column: str,
    positive_feedback_column: str | None,
    split_seed: int,
    policy_k: int,
    policy_seed: int = 0,
    filter_config: dict[str, Any],
    splitter_config: dict[str, Any],
    search_spaces: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Load a tuning artifact and validate its identity strictly.

    Raises RuntimeError with remediation instructions when the artifact is
    missing, and ValueError naming the mismatched fields when any identity
    field disagrees (stale artifact).
    """

    if search_spaces is None:
        search_spaces = SEARCH_SPACES
    path = artifacts_dir / artifact_filename(
        dataset_name, target_source_column, split_seed
    )
    if not path.exists():
        raise RuntimeError(
            f"Tuned policy artifact not found: {path}. Tune first with: "
            "uv run python -m runners.in_distribution.policy_tuning.run "
            f"--datasets {dataset_name} --seeds {split_seed}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported tuning artifact schema in {path}: "
            f"{payload.get('schema_version')!r} (expected {ARTIFACT_SCHEMA_VERSION})"
        )
    expected_identity = {
        "dataset": dataset_name,
        "dataset_version": dataset_version,
        "target_source_column": target_source_column,
        "positive_feedback_column": positive_feedback_column,
        "split_seed": split_seed,
        "policy_k": policy_k,
        "policy_seed": policy_seed,
        "objective": OBJECTIVE_METRIC,
        "filter": filter_config,
        "splitter": splitter_config,
        "inner_split": {
            "val_fraction_of_train": INNER_VAL_FRACTION_OF_TRAIN,
            "seed_offset": INNER_SEED_OFFSET,
            "group_column": "user_id",
        },
        "search_spaces_sha256": search_spaces_sha256(search_spaces),
    }
    found_identity = payload.get("identity", {})
    mismatches = [
        key
        for key, expected in expected_identity.items()
        if found_identity.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            f"Stale tuning artifact {path}: identity mismatch on "
            f"{mismatches}. Expected { {k: expected_identity[k] for k in mismatches} }, "
            f"found { {k: found_identity.get(k) for k in mismatches} }. "
            "Re-tune with: uv run python -m runners.in_distribution.policy_tuning.run "
            f"--datasets {dataset_name} --seeds {split_seed}"
        )
    return payload, path


def build_tuned_policy_roster(
    *,
    artifacts_dir: Path,
    dataset_name: str,
    dataset_version: str,
    target_source_column: str,
    positive_feedback_column: str | None,
    split_seed: int,
    policy_k: int,
    policy_seed: int = 0,
    filter_config: dict[str, Any],
    splitter_config: dict[str, Any],
    search_spaces: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> list[Policy]:
    """Build the full policy roster from a validated tuning artifact."""

    payload, path = load_tuned_artifact(
        artifacts_dir,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        target_source_column=target_source_column,
        positive_feedback_column=positive_feedback_column,
        split_seed=split_seed,
        policy_k=policy_k,
        policy_seed=policy_seed,
        filter_config=filter_config,
        splitter_config=splitter_config,
        search_spaces=search_spaces,
    )
    artifact_ref = (
        str(path) if repo_root is None else str(path.relative_to(repo_root))
    )
    roster: list[Policy] = []
    for cls in POLICY_ROSTER:
        entry = payload["policies"][cls.__name__]
        policy = cls(
            k=policy_k,
            seed=policy_seed,
            positive_feedback_column=positive_feedback_column,
            **entry["selected_params"],
        )
        policy.hyperparameter_selection = {
            "kind": "optuna_grid_search" if entry["tuned"] else "fixed_baseline",
            "objective": payload["identity"]["objective"],
            "val_objective": entry["val_objective"],
            "n_trials": entry.get("n_trials"),
            "sampler": entry.get("sampler"),
            "artifact": artifact_ref,
            "artifact_identity": payload["identity"],
        }
        roster.append(policy)
    return roster
