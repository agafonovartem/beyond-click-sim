"""Hyperparameter tuning for the Coat re-ranking protocol.

A deliberate fork of `policy_tuning.py`, not an extension of it. The free
retrieval protocol that module tunes under cannot be applied to Coat:

1. Its inner validation split is carved from the outer train frame. On Coat that
   frame holds 22.74 rows per user, so a 25% inner split yields 5.68 candidates
   per user against `POLICY_K = 5` — a top-5 covering 88% of the pool, where no
   hyperparameter setting can distinguish itself from another.
2. `_evaluate_once` scores `policy.recommend()` over the full catalog and
   zero-fills unlabelled pairs. With 5.68 labelled cells among 300 items, ~98%
   of every top-10 slate would be a fabricated negative.
3. Its objective is `ndcg@10` under free retrieval, while the re-ranking runner
   ranks policies by the mean target over a 5-row slate. On a 5-row slate
   `ndcg@10` silently equals `ndcg@5`, so the mismatch would never raise.

So this module tunes under the protocol the experiment actually evaluates:
policies fit on the runner's full train frame and re-rank each user's own
uniformly-assigned (MCAR) pool, scored by the mean target over the top-k slate —
which is precisely the utility `compute_policy_utilities` reports.

**The validation labels are the evaluation labels.** Coat offers no third
partition, so hyperparameters are selected on the same cells that later produce
the reference policy ordering. That inflates every tunable policy's real
utility, and differentially: best-of-12 inflates more than best-of-4, and
`RandomPolicy`/`PopularityPolicy` — which have nothing to tune — not at all.
`tuning_user_fraction < 1.0` exists to measure that bias rather than assert it;
see `runners/in_distribution/policy_tuning_rerank/run.py`. Every artifact
carries a `leakage` block stating the overlap.
"""

from __future__ import annotations

import random
from time import perf_counter
from typing import Any

import optuna
import pandas as pd
from loguru import logger

from beyond_click_sim.evaluation.policy_ranking import (
    evaluate_policy_recommendations,
)
from beyond_click_sim.tasks.policies import Policy
from beyond_click_sim.tasks.policy_ranking import POLICY_EVAL_KS
from beyond_click_sim.tasks.policy_reranking import PolicyRerankingTaskBuilder
from beyond_click_sim.tasks.policy_tuning import (
    ARTIFACT_SCHEMA_VERSION,
    POLICY_CLASSES,
    POLICY_ROSTER,
    SAMPLER_SEED_OFFSET,
    _grid_n_trials,
    _grid_search_space,
    _suggest_params,
    search_spaces_sha256,
)

PROTOCOL = "rerank_pool"
# The mean target over a fixed-size slate. Identical to the runner's utility:
# every Coat pool is 16 items and k is 5, so each policy emits exactly k rows
# per user and this pooled mean equals `compute_policy_utilities`' macro-by-user
# mean. `_slate_frame` asserts the constant slate size that identity rests on.
OBJECTIVE_METRIC = "mean_hit_rate"
TUNING_USER_SEED_OFFSET = 3000

# Search spaces for a 290 x 300 matrix with 1,814 positive training rows —
# roughly three orders of magnitude smaller than ml-1m, so factor counts go far
# lower and regularisation far higher than `policy_tuning.SEARCH_SPACES`.
#
# `iterations` is tuned here, on every policy that has it. The 300 pinned in the
# ml-1m space is a budget chosen for a matrix a thousand times larger; carrying
# it over would be assuming the answer. ALS is the sharper case — its
# `iterations` was never in any grid, so it silently ran the class default of 20.
#
# GRID SIZE. Measured cost per point on this data (fit + re-rank 290 users):
# ALS 0.03-0.13s, BPR 0.04-0.32s, LightGCN 0.93-2.16s, ItemKNN ~2.0s (dominated
# by ranking, not fitting). A full pass over the grids below is a few minutes,
# so the budget is set by what is worth learning rather than by compute.
#
# These grids are deliberately NOT balanced, unlike revision 2. The winner's
# curse grows with the number of points as sqrt(ln G), so the 35-fold spread
# between ItemKNN's 11 and ALS's 384 buys only about 1.6x more inflation on ALS
# — and the diagnostic pass measures that directly rather than assuming it (at
# revision 2 the largest measured curse was +0.0085, on ItemKNN's 4-point grid,
# i.e. grid size was not what drove it). What stays irreducible is that
# RandomPolicy and PopularityPolicy have nothing to select and so carry no curse
# at all; no grid design fixes that asymmetry, which is why it is stated in the
# leakage block instead.
#
# BOX PROVENANCE — revision 4. Revision 1 drew every box too narrow: all four
# policies selected on a boundary, ALS on all three axes at once. Revision 2
# widened each in the direction its edge pointed, and most selections still sat
# on an edge, every one pointing the same way — fewer factors, more smoothing,
# more neighbours. Revision 3 therefore extends each axis all the way to its
# degenerate endpoint: `n_factors=1`, `n_neighbors=300` (the entire catalog,
# where ItemKNN is popularity by another name). The point is to make "these
# policies collapse toward popularity on a matrix this small" a selection the
# search can actually land on and report, rather than a boundary artifact that
# looks the same whether or not it is true.
#
# Revision 4 closed the edges revision 3 exposed. Making `iterations` tunable
# moved BPR and LightGCN to the ceiling of {100, 300, 1000}, confirming that the
# 300 inherited from ml-1m was far too short here, so both now reach 3000. Note
# what that did to BPR: with a long budget and a slow learning rate it prefers
# the *smallest* model on offer, having wanted the largest when the budget was
# pinned. Tuning `iterations` and tuning capacity are not separable questions.
#
# Each revision is one more decision made against labels that are also the
# evaluation labels. That is a layer of selection the curse diagnostic does NOT
# model: it controls for choosing a point within a grid, not for choosing the
# grid. Boxes 1 and 2 were revised on the 145-user diagnostic half only.
SEARCH_SPACES: dict[str, Any] = {
    # 11 points. n_neighbors=300 is the whole catalog: the explicit
    # popularity-equivalent endpoint.
    "ItemKNNPolicy": {
        "n_trials": 11,
        "params": {
            "n_neighbors": {
                "kind": "categorical",
                "choices": [5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300],
            },
        },
    },
    # 6 * 5 * 4 * 4 = 480 points, ~50s. `regularization` reaches down to 1e-4
    # because 0.01 was selected at the floor of the previous box.
    "ALSPolicy": {
        "n_trials": 480,
        "params": {
            "n_factors": {"kind": "categorical", "choices": [1, 2, 4, 8, 16, 32]},
            "regularization": {
                "kind": "categorical",
                "choices": [0.0001, 0.001, 0.01, 0.1, 1.0],
            },
            "alpha": {"kind": "categorical", "choices": [1.0, 5.0, 20.0, 100.0]},
            "iterations": {"kind": "categorical", "choices": [10, 20, 50, 100]},
        },
    },
    # 5 * 3 * 3 * 3 = 135 points, ~1min. Every axis shifted down or out after the
    # previous box selected `iterations` at its ceiling with the smallest model
    # and the slowest learning rate — a slow-and-small optimum that only became
    # reachable once `iterations` was tunable at all.
    "BPRPolicy": {
        "n_trials": 135,
        "params": {
            "n_factors": {"kind": "categorical", "choices": [2, 4, 8, 16, 32]},
            "learning_rate": {
                "kind": "categorical",
                "choices": [0.0005, 0.002, 0.01],
            },
            "regularization": {
                "kind": "categorical",
                "choices": [0.01, 0.1, 1.0],
            },
            "iterations": {"kind": "categorical", "choices": [300, 1000, 3000]},
        },
    },
    # 3 * 3 * 3 * 3 = 81 points, ~4min — the expensive one, so the axes that
    # already selected interior (n_factors) are narrowed to pay for the ones that
    # hit a ceiling (iterations, n_layers) or a floor (regularization).
    # `iterations` counts mini-batch epochs on the torch backend, not full-batch
    # updates as on the NumPy LightGCN; the values are not comparable across the
    # two.
    "LightGCNTorchPolicy": {
        "n_trials": 81,
        "params": {
            "n_factors": {"kind": "categorical", "choices": [4, 8, 16]},
            "n_layers": {"kind": "categorical", "choices": [2, 3, 4]},
            "regularization": {
                "kind": "categorical",
                "choices": [0.0001, 0.001, 0.01],
            },
            "iterations": {"kind": "categorical", "choices": [300, 1000, 3000]},
        },
    },
}

# Recorded in the artifact so a later reader can tell a first-draft box from a
# revised one without diffing the source.
SEARCH_SPACE_REVISION = 4


def artifact_filename(
    dataset_name: str,
    target_source_column: str,
    n_tuning_users: int,
    split_seed: int,
) -> str:
    """Artifact name carrying the tuning-user count.

    The production roster and the winner's-curse diagnostic differ only in how
    many users were tuned on, so that count has to be in the filename or the two
    artifacts collide.
    """

    return (
        f"{dataset_name}_{target_source_column}_rerank"
        f"_users{n_tuning_users}_seed{split_seed}.json"
    )


def select_tuning_users(
    all_users: list[Any],
    *,
    fraction: float,
    split_seed: int,
) -> list[Any]:
    """A seeded, reproducible user subset to select hyperparameters on."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"tuning_user_fraction must be in (0, 1]; got {fraction}")
    ordered = sorted(all_users, key=str)
    if fraction == 1.0:
        return ordered
    n = max(1, round(len(ordered) * fraction))
    rng = random.Random(split_seed + TUNING_USER_SEED_OFFSET)
    return sorted(rng.sample(ordered, n), key=str)


def boundary_params(spec: dict[str, Any], selected: dict[str, Any]) -> list[str]:
    """Names of selected hyperparameters sitting on an edge of their choice list.

    A boundary selection means the optimum is probably outside the box, so the
    policy is being reported at a handicap. Flagging it mechanically beats
    re-reading trial logs: it is the one tuning defect that looks exactly like a
    normal result. Widening the box is not free here — the selection labels are
    the evaluation labels, so each revision spends a little more of them (see
    the SEARCH_SPACES box-provenance note).
    """

    flagged = []
    for name, value in sorted(selected.items()):
        choices = spec["params"][name]["choices"]
        if len(choices) > 1 and value in (min(choices), max(choices)):
            edge = "min" if value == min(choices) else "max"
            flagged.append(f"{name}={value} ({edge} of {choices})")
    return flagged


def _slate_frame(
    policy: Policy,
    *,
    eval_users: list[Any],
    pool_by_user: dict[Any, list[Any]],
    val_targets: dict[tuple[Any, Any], Any],
    fit_interactions: pd.DataFrame,
    user_column: str,
    item_column: str,
) -> tuple[pd.DataFrame, int]:
    """Re-rank each user's pool and return the labelled top-k slate frame.

    Mirrors `PolicyRerankingTaskBuilder.build` exactly: same permutation
    validation, same fallback accounting, same strict label join. Tuning under a
    different candidate construction than the run would select hyperparameters
    for a task the experiment never measures.
    """

    rows: list[dict[str, Any]] = []
    fallbacks = 0
    for user_id in eval_users:
        pool = pool_by_user[user_id]
        ordering = policy.rank_candidates(
            user_id,
            pool,
            train_interactions=fit_interactions,
            user_column=user_column,
            item_column=item_column,
        )
        if sorted(map(str, ordering)) != sorted(map(str, pool)):
            raise ValueError(
                f"{policy.name}.rank_candidates did not return a permutation "
                f"of the candidate pool for user {user_id!r}"
            )
        if PolicyRerankingTaskBuilder._is_untied_fallback(
            policy, user_id, pool, ordering
        ):
            fallbacks += 1
        rows.extend(
            policy._build_recommendation_rows(
                user_id,
                list(ordering)[: policy.k],
                user_column=user_column,
                item_column=item_column,
            )
        )

    recs = pd.DataFrame(rows, columns=[user_column, item_column, "policy", "rank"])

    # The pooled-mean == macro-by-user-mean identity that makes `mean_hit_rate`
    # the runner's utility holds only while every user contributes k rows.
    sizes = recs.groupby(user_column).size()
    if len(sizes) and not (sizes == policy.k).all():
        raise ValueError(
            f"{policy.name}: slate sizes are not constant at k={policy.k} "
            f"(min={sizes.min()}, max={sizes.max()}); the tuning objective would "
            "no longer equal the runner's per-user utility"
        )

    missing = [
        key
        for key in zip(recs[user_column], recs[item_column], strict=True)
        if key not in val_targets
    ]
    if missing:
        raise ValueError(
            "Re-ranking slates must be fully labelled, but "
            f"{len(missing)} (user, item) pairs had no validation label for "
            f"policy {policy.name!r}; first few: {missing[:5]}"
        )
    recs["target"] = [
        val_targets[(u, i)]
        for u, i in zip(recs[user_column], recs[item_column], strict=True)
    ]
    return recs, fallbacks


def _score_slate(
    policy: Policy,
    recs: pd.DataFrame,
    *,
    fallbacks: int,
    user_column: str,
    fit_recommend_seconds: float | None = None,
) -> tuple[float, dict[str, Any]]:
    metric = evaluate_policy_recommendations(
        recs,
        targets=recs["target"],
        user_column=user_column,
        policy_name=policy.name,
        k=policy.k,
        ks=POLICY_EVAL_KS,
        fit_recommend_seconds=fit_recommend_seconds,
    )
    payload = {
        "mean_hit_rate": metric["mean_hit_rate"],
        # Logged but not optimized: keeping the full ranking block means the
        # objective choice can be revisited from the artifact without re-running.
        "ranking": metric["ranking"],
        "n_users": metric["n_users"],
        "n_recommendations": metric["n_recommendations"],
        "rank_candidates_fallback_users": fallbacks,
    }
    if fit_recommend_seconds is not None:
        payload["fit_recommend_seconds"] = metric["fit_recommend_seconds"]
    return float(metric["mean_hit_rate"]), payload


def _evaluate_pool_once(
    policy: Policy,
    *,
    fit_interactions: pd.DataFrame,
    arms: dict[str, list[Any]],
    pool_by_user: dict[Any, list[Any]],
    val_targets: dict[tuple[Any, Any], Any],
    items: pd.DataFrame,
    user_column: str,
    item_column: str,
) -> tuple[float, dict[str, Any], dict[str, list[float]]]:
    """Fit on the full train frame, then score on each named arm.

    The first arm is the selection arm and supplies the objective; any further
    arm is scored from the same fit and only recorded. Returns
    (objective, payload, per-user utilities per arm).
    """

    # Timed end to end, fit plus every arm's ranking. Ranking is not a rounding
    # error here: ItemKNN spends ~90% of its per-point cost there, so a fit-only
    # timer would under-size the next box revision by two orders of magnitude.
    t0 = perf_counter()
    policy.fit(
        fit_interactions,
        items=items,
        user_column=user_column,
        item_column=item_column,
    )

    payloads: dict[str, dict[str, Any]] = {}
    per_user: dict[str, list[float]] = {}
    for label, group in arms.items():
        recs, fallbacks = _slate_frame(
            policy,
            eval_users=group,
            pool_by_user=pool_by_user,
            val_targets=val_targets,
            fit_interactions=fit_interactions,
            user_column=user_column,
            item_column=item_column,
        )
        if len(recs) == 0:
            payloads[label] = {
                "mean_hit_rate": 0.0,
                "n_users": 0,
                "n_recommendations": 0,
                "empty_recommendations": True,
            }
            per_user[label] = []
            continue
        _, payload = _score_slate(
            policy, recs, fallbacks=fallbacks, user_column=user_column
        )
        payloads[label] = payload
        by_user = recs.groupby(user_column)["target"].mean()
        per_user[label] = [float(by_user[u]) for u in group]

    primary = next(iter(arms))
    payload = dict(payloads[primary])
    payload["fit_recommend_seconds"] = perf_counter() - t0
    for label, other in payloads.items():
        if label != primary:
            payload[f"{label}_mean_hit_rate"] = other["mean_hit_rate"]
    return float(payload["mean_hit_rate"]), payload, per_user


def tune_policies_for_rerank_split(
    *,
    train_interactions: pd.DataFrame,
    val_interactions: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
    dataset_name: str,
    dataset_version: str,
    target_source_column: str,
    positive_feedback_column: str | None,
    split_seed: int,
    policy_k: int,
    policy_seed: int = 0,
    tuning_user_fraction: float = 1.0,
    filter_config: dict[str, Any],
    splitter_config: dict[str, Any],
    search_spaces: dict[str, Any] | None = None,
    policy_classes: tuple[type[Policy], ...] | None = None,
    user_column: str = "user_id",
    item_column: str = "item_id",
) -> dict[str, Any]:
    """Grid-search policies that re-rank held-out pools.

    `train_interactions` is the runner's full train frame — policies fit on
    exactly what they fit on at run time, so unlike the ml-1m protocol there is
    no train-size mismatch between tuning and deployment.

    `val_interactions` is the runner's held-out MCAR frame. It is also the
    evaluation frame; see the module docstring. When `tuning_user_fraction < 1`
    the selected setting is additionally re-scored on the untuned users, and the
    gap between the two is written to `curse_diagnostic`.
    """

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if search_spaces is None:
        search_spaces = SEARCH_SPACES
    classes = POLICY_ROSTER if policy_classes is None else policy_classes
    class_names = set(POLICY_CLASSES) | {cls.__name__ for cls in classes}
    assert set(search_spaces) <= class_names, (
        "Unknown search-space keys (no matching policy class): "
        f"{sorted(set(search_spaces) - class_names)}"
    )

    val_targets: dict[tuple[Any, Any], Any] = {}
    for row in val_interactions.itertuples(index=False):
        key = (getattr(row, user_column), getattr(row, item_column))
        val_targets[key] = getattr(row, target_source_column)

    pool_by_user: dict[Any, list[Any]] = (
        val_interactions.groupby(user_column, sort=True)[item_column]
        .agg(list)
        .to_dict()
    )
    known_users = set(users[user_column])
    all_eval_users = [u for u in sorted(pool_by_user, key=str) if u in known_users]
    tuning_users = select_tuning_users(
        all_eval_users, fraction=tuning_user_fraction, split_seed=split_seed
    )
    holdout_users = [u for u in all_eval_users if u not in set(tuning_users)]

    pool_sizes = {len(pool) for pool in pool_by_user.values()}
    logger.info(
        f"Tuning on {len(tuning_users)}/{len(all_eval_users)} users, "
        f"pool sizes {sorted(pool_sizes)}, k={policy_k}, "
        f"fit frame {len(train_interactions)} rows"
    )

    sampler_seed = split_seed + SAMPLER_SEED_OFFSET
    policies_result: dict[str, Any] = {}
    grid_records: dict[str, dict[str, Any]] = {}
    arms: dict[str, list[Any]] = {"tuning": tuning_users}
    if holdout_users:
        arms["holdout"] = holdout_users

    for cls in classes:
        name = cls.__name__
        spec = search_spaces.get(name)

        def _make_policy(params: dict[str, Any]) -> Policy:
            return cls(
                k=policy_k,
                seed=policy_seed,
                positive_feedback_column=positive_feedback_column,
                **params,
            )

        def _evaluate(params: dict[str, Any]):
            value, payload, per_user = _evaluate_pool_once(
                _make_policy(params),
                fit_interactions=train_interactions,
                arms=arms,
                pool_by_user=pool_by_user,
                val_targets=val_targets,
                items=items,
                user_column=user_column,
                item_column=item_column,
            )
            if holdout_users:
                grid_points.append({
                    "params": params,
                    "tuning": per_user["tuning"],
                    "holdout": per_user["holdout"],
                })
            return value, payload

        grid_points: list[dict[str, Any]] = []

        if spec is None:
            logger.info(f"{name}: no search space, evaluating baseline once")
            value, payload = _evaluate({})
            entry: dict[str, Any] = {
                "tuned": False,
                "selected_params": {},
                "val_objective": value,
                "n_eval_users": len(tuning_users),
                "grid_size": 1,
                "boundary_params": [],
                "trials": [
                    {"number": 0, "state": "complete", "params": {},
                     "value": value, "metrics": payload},
                ],
            }
        else:
            grid = _grid_search_space(spec)
            grid_size = _grid_n_trials(grid)
            n_trials = int(spec["n_trials"])
            if n_trials < grid_size:
                raise ValueError(
                    f"SEARCH_SPACES[{name!r}] n_trials={n_trials} is smaller than "
                    f"categorical grid size {grid_size}. Raise n_trials or shrink "
                    "the choices lists."
                )
            logger.info(f"{name}: grid of {grid_size} points, budget {n_trials}")
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.GridSampler(grid, seed=sampler_seed),
            )
            trial_log: list[dict[str, Any]] = []

            def objective(trial: Any) -> float:
                params = _suggest_params(trial, spec)
                try:
                    value, payload = _evaluate(params)
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
                objective, n_trials=n_trials, catch=(Exception,), show_progress_bar=False
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
                    f"Trial errors: {[t.get('error') for t in trial_log][:3]}"
                )
            entry = {
                "tuned": True,
                "selected_params": dict(study.best_params),
                "val_objective": float(study.best_value),
                "n_eval_users": len(tuning_users),
                "sampler": {"name": "GridSampler", "seed": sampler_seed},
                "n_trials": n_trials,
                # The winner's-curse magnitude scales with how many points the
                # argmax was taken over, so the artifact has to carry it.
                "grid_size": grid_size,
                "boundary_params": boundary_params(spec, dict(study.best_params)),
                "trials": trial_log,
            }

        if holdout_users:
            # Grid points are appended in evaluation order, which for a failed
            # trial is no order at all, so locate the winner by its params.
            selected_index = next(
                i
                for i, point in enumerate(grid_points)
                if point["params"] == entry["selected_params"]
            )
            grid_records[name] = {
                "points": grid_points,
                "selected_index": selected_index,
            }
        logger.info(
            f"{name}: selected {entry['selected_params']} "
            f"(objective {entry['val_objective']:.4f})"
        )
        policies_result[name] = entry

    curse_summary: dict[str, Any] | None = None
    if grid_records:
        curse_summary = _curse_summary(grid_records)
        for name, diagnostic in curse_summary["per_policy"].items():
            policies_result[name]["curse_diagnostic"] = diagnostic
            logger.info(
                f"{name}: tuned {diagnostic['tuning_utility']:.4f} vs heldout "
                f"{diagnostic['holdout_utility']:.4f} | excess over grid "
                f"{diagnostic['excess_over_grid']:+.4f} "
                f"[{diagnostic['excess_ci95'][0]:+.4f}, "
                f"{diagnostic['excess_ci95'][1]:+.4f}]"
            )

    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "identity": {
            "dataset": dataset_name,
            "dataset_version": dataset_version,
            "protocol": PROTOCOL,
            "target_source_column": target_source_column,
            "positive_feedback_column": positive_feedback_column,
            "split_seed": split_seed,
            "policy_k": policy_k,
            "policy_seed": policy_seed,
            "objective": OBJECTIVE_METRIC,
            "filter": filter_config,
            "splitter": splitter_config,
            "fit_frame": {
                "n_rows": int(len(train_interactions)),
                "n_users": int(train_interactions[user_column].nunique()),
                "source": "runner_train_split_in_full",
            },
            # Stated in machine-readable form because `load_policy_roster` never
            # validates `tuned_from`: a config entry can outlive the artifact
            # that justified it, and this is the caveat that must travel with it.
            "leakage": {
                "validation_source": "mcar_eval_pool",
                "overlaps_eval_users": True,
                "tuning_user_fraction": float(tuning_user_fraction),
                "n_tuning_users": len(tuning_users),
                "n_holdout_users": len(holdout_users),
                "n_eval_users_in_run": len(all_eval_users),
                "note": (
                    "Hyperparameters were selected on cells that also form the "
                    "reference ordering of the re-ranking run. Real utilities of "
                    "tuned policies are optimistically biased; RandomPolicy and "
                    "PopularityPolicy are not tuned and carry no such bias."
                ),
            },
            "search_spaces_sha256": search_spaces_sha256(search_spaces),
            "search_space_revision": SEARCH_SPACE_REVISION,
        },
        "policies": policies_result,
        "curse_summary": curse_summary,
    }


def _curse_summary(
    records: dict[str, dict[str, Any]],
    *,
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
    """Winner's curse per policy, controlled by that policy's own grid.

    A raw tuned-minus-holdout gap is dominated by a nuisance: the two user
    halves have different base rates, and every setting feels it. The gap
    averaged over *all* grid points estimates that nuisance cleanly, because
    averaging over the whole grid never conditions on which point won. The
    selected point's gap minus the grid's mean gap is therefore the selection
    bias alone.

    Controlling with the policy's own grid rather than with a different policy
    matters: settings of one recommender agree closely on which users are easy,
    so the nuisance cancels almost exactly. It also makes a one-point grid
    report a curse of exactly zero, which is the truth — nothing was selected.

    `excess` is the curse for a selection made on the tuning arm. A selection
    made on twice as many users carries roughly 1/sqrt(2) of it, since the curse
    tracks the standard error of the quantity being maximized.
    """

    import numpy as np

    names = sorted(records)
    n_tune = len(records[names[0]]["points"][0]["tuning"])
    n_hold = len(records[names[0]]["points"][0]["holdout"])

    rng = np.random.default_rng(0)
    # One set of resampled user indices per replicate, shared by every policy
    # and every grid point so all the deltas stay aligned.
    tune_idx = rng.integers(0, n_tune, size=(n_bootstrap, n_tune))
    hold_idx = rng.integers(0, n_hold, size=(n_bootstrap, n_hold))

    per_policy: dict[str, Any] = {}
    for name in names:
        points = records[name]["points"]
        selected = records[name]["selected_index"]
        tuning = np.asarray([pt["tuning"] for pt in points])
        holdout = np.asarray([pt["holdout"] for pt in points])

        # (n_points, n_bootstrap) gap per grid point per replicate.
        deltas = (
            tuning[:, tune_idx].mean(axis=2) - holdout[:, hold_idx].mean(axis=2)
        )
        excess = deltas[selected] - deltas.mean(axis=0)

        t_sel = tuning[selected]
        h_sel = holdout[selected]
        per_policy[name] = {
            "tuning_utility": float(t_sel.mean()),
            "holdout_utility": float(h_sel.mean()),
            "delta": float(t_sel.mean() - h_sel.mean()),
            "grid_mean_delta": float(
                (tuning.mean(axis=1) - holdout.mean(axis=1)).mean()
            ),
            "excess_over_grid": float(np.mean(excess)),
            "excess_ci95": [
                float(np.percentile(excess, 2.5)),
                float(np.percentile(excess, 97.5)),
            ],
            "n_grid_points": int(len(points)),
            "n_tuning_users": n_tune,
            "n_holdout_users": n_hold,
        }
    return {
        "per_policy": per_policy,
        "control": "own_grid_mean_delta",
        "n_bootstrap": n_bootstrap,
        "note": (
            "excess_over_grid is the winner's curse for a selection made on "
            f"{n_tune} users; a selection on all {n_tune + n_hold} carries "
            "roughly 1/sqrt(2) of it. A one-point grid reports exactly 0."
        ),
    }
