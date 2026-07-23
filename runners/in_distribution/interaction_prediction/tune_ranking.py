"""Tune classical ranking-baseline hyperparameters on validation (Optuna).

Objective is the ranking main metric measured on the *validation* candidate
sets: ``val.macro_by_user_group_mean.ndcg@5``. Model hyperparameters are chosen
to maximize it; the held-out ``test`` split is never seen by the search. For each
method we then evaluate both the default and the best config on test via the
standard ``run_classical_ranking`` runner, so the written artifacts are directly
comparable to the untuned baselines.

Search strategy (per the project decision):
  - item_knn / als / bpr : Optuna TPE, with the default config enqueued as the
    first trial so "did tuning help?" is always answerable.
  - lightgcn              : a small Optuna GridSampler grid (each fit is ~13 min).

Run (via a launcher that is import-safe on the current platform), e.g.:
  uv run python runners/in_distribution/interaction_prediction/tune_ranking.py \
      --task ml-1m_cap20_eval_users1000_cg5_m19_seed0 \
      --methods item_knn,als,bpr,lightgcn --output-dir <dir>
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import optuna

from beyond_click_sim.scorers import (
    ALSScorer,
    BPRScorer,
    ItemKNNScorer,
    LightGCNScorer,
)
from beyond_click_sim.scorers.base import Scorer
from beyond_click_sim.tasks import Task
from runners.in_distribution.interaction_prediction.methods.classical import (
    run_classical_ranking,
)
from runners.in_distribution.interaction_prediction.methods.common import (
    ranking_metrics_for_split,
    task_xy,
    write_json,
)
from runners.in_distribution.interaction_prediction.task_builders import (
    TASK_BUILDERS,
    repo_root,
)


SEED = 0
# Objective: val macro_by_user_group_mean ndcg@5 (the ranking main metric on val).
OBJECTIVE_AGG = "macro_by_user_group_mean"
OBJECTIVE_METRIC = "ndcg@5"


@dataclass
class MethodSpec:
    """How to search and build one method's scorer."""

    build: Callable[[dict[str, Any]], Scorer]
    suggest: Callable[[optuna.Trial], dict[str, Any]]
    default: dict[str, Any]
    n_trials: int
    grid: dict[str, list[Any]] | None = None  # set → use GridSampler, ignore n_trials


# --- item_knn ---------------------------------------------------------------
def _build_item_knn(p: dict[str, Any]) -> Scorer:
    return ItemKNNScorer(
        n_neighbors=p["n_neighbors"],
        aggregation=p["aggregation"],
        item_column="item_id",
        user_column="user_id",
    )


def _suggest_item_knn(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 5, 400, log=True),
        "aggregation": trial.suggest_categorical("aggregation", ["mean", "sum"]),
    }


# --- als --------------------------------------------------------------------
def _build_als(p: dict[str, Any]) -> Scorer:
    return ALSScorer(
        n_factors=p["n_factors"],
        iterations=p["iterations"],
        regularization=p["regularization"],
        seed=SEED,
        num_threads=1,
        item_column="item_id",
        user_column="user_id",
    )


def _suggest_als(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_factors": trial.suggest_int("n_factors", 16, 256, log=True),
        "iterations": trial.suggest_int("iterations", 10, 60),
        "regularization": trial.suggest_float("regularization", 1e-3, 1.0, log=True),
    }


# --- bpr --------------------------------------------------------------------
def _build_bpr(p: dict[str, Any]) -> Scorer:
    return BPRScorer(
        n_factors=p["n_factors"],
        learning_rate=p["learning_rate"],
        regularization=p["regularization"],
        iterations=p["iterations"],
        seed=SEED,
        num_threads=1,
        item_column="item_id",
        user_column="user_id",
    )


def _suggest_bpr(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_factors": trial.suggest_int("n_factors", 16, 256, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "regularization": trial.suggest_float("regularization", 1e-4, 1e-1, log=True),
        "iterations": trial.suggest_int("iterations", 50, 300),
    }


# --- lightgcn (small grid; each fit ~13 min) --------------------------------
_LIGHTGCN_FIXED = {"n_factors": 64, "n_layers": 3, "regularization": 1e-4, "batch_size": 2048}
_LIGHTGCN_GRID = {"learning_rate": [0.01, 0.05], "iterations": [200, 400]}


def _build_lightgcn(p: dict[str, Any]) -> Scorer:
    return LightGCNScorer(
        n_factors=p["n_factors"],
        n_layers=p["n_layers"],
        learning_rate=p["learning_rate"],
        regularization=p["regularization"],
        iterations=p["iterations"],
        batch_size=p["batch_size"],
        seed=SEED,
        item_column="item_id",
        user_column="user_id",
    )


def _suggest_lightgcn(trial: optuna.Trial) -> dict[str, Any]:
    return {
        **_LIGHTGCN_FIXED,
        "learning_rate": trial.suggest_categorical(
            "learning_rate", _LIGHTGCN_GRID["learning_rate"]
        ),
        "iterations": trial.suggest_categorical(
            "iterations", _LIGHTGCN_GRID["iterations"]
        ),
    }


METHOD_SPECS: dict[str, MethodSpec] = {
    "item_knn": MethodSpec(
        build=_build_item_knn,
        suggest=_suggest_item_knn,
        default={"n_neighbors": 20, "aggregation": "mean"},
        n_trials=16,
    ),
    "als": MethodSpec(
        build=_build_als,
        suggest=_suggest_als,
        default={"n_factors": 64, "iterations": 20, "regularization": 0.01},
        n_trials=30,
    ),
    "bpr": MethodSpec(
        build=_build_bpr,
        suggest=_suggest_bpr,
        default={
            "n_factors": 64,
            "learning_rate": 0.01,
            "regularization": 0.01,
            "iterations": 100,
        },
        n_trials=30,
    ),
    "lightgcn": MethodSpec(
        build=_build_lightgcn,
        suggest=_suggest_lightgcn,
        default={**_LIGHTGCN_FIXED, "learning_rate": 0.001, "iterations": 200},
        n_trials=len(_LIGHTGCN_GRID["learning_rate"]) * len(_LIGHTGCN_GRID["iterations"]),
        grid=_LIGHTGCN_GRID,
    ),
}


def _val_ndcg(scorer: Scorer, xy: dict, group_col: str) -> float:
    """Fit on train, score val, return val macro_by_user_group_mean ndcg@5."""
    X_train, y_train = xy["train"]
    X_val, y_val = xy["val"]
    scorer.fit(X_train, y_train)
    val_scores = scorer.score(X_val)
    metrics = ranking_metrics_for_split(
        X=X_val, y=y_val, scores=val_scores, candidate_group_column=group_col
    )
    return float(metrics[OBJECTIVE_AGG][OBJECTIVE_METRIC])


def _study_for(spec: MethodSpec) -> optuna.Study:
    if spec.grid is not None:
        sampler: optuna.samplers.BaseSampler = optuna.samplers.GridSampler(
            spec.grid, seed=SEED
        )
    else:
        sampler = optuna.samplers.TPESampler(seed=SEED)
    return optuna.create_study(direction="maximize", sampler=sampler)


def tune_method(
    method: str,
    spec: MethodSpec,
    task: Task,
    xy: dict,
    group_col: str,
    output_root: Path,
) -> dict[str, Any]:
    """Run the study, then evaluate default and best configs on test."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = _study_for(spec)
    if spec.grid is None:
        # Always evaluate the current default first so tuning is comparable.
        study.enqueue_trial(spec.default)

    trials: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = spec.suggest(trial)
        trial.set_user_attr("full_params", params)
        t0 = perf_counter()
        value = _val_ndcg(spec.build(params), xy, group_col)
        trial.set_user_attr("fit_seconds", round(perf_counter() - t0, 2))
        trials.append(
            {"params": params, "val_ndcg5": value, "seconds": trial.user_attrs["fit_seconds"]}
        )
        print(
            f"[tune:{method}] trial {trial.number}: val_ndcg@5={value:.4f} "
            f"({trial.user_attrs['fit_seconds']}s) {params}",
            flush=True,
        )
        return value

    study.optimize(objective, n_trials=spec.n_trials)
    best_params = study.best_trial.user_attrs["full_params"]
    best_val = float(study.best_value)
    print(f"[tune:{method}] BEST val_ndcg@5={best_val:.4f} {best_params}", flush=True)

    # Evaluate default and best on test via the standard ranking runner.
    default_metrics = _eval_on_test(
        method, "default", spec.default, spec.build, task, output_root,
        tuning={"objective": f"val.{OBJECTIVE_AGG}.{OBJECTIVE_METRIC}"},
    )
    best_metrics = _eval_on_test(
        method, "tuned", best_params, spec.build, task, output_root,
        tuning={
            "objective": f"val.{OBJECTIVE_AGG}.{OBJECTIVE_METRIC}",
            "sampler": type(study.sampler).__name__,
            "n_trials": len(study.trials),
            "best_val_ndcg5": best_val,
        },
    )

    def _test_ndcg(m: dict) -> float:
        return float(m["test"][OBJECTIVE_AGG][OBJECTIVE_METRIC])

    return {
        "method": method,
        "objective": f"val.{OBJECTIVE_AGG}.{OBJECTIVE_METRIC}",
        "default_params": spec.default,
        "default_val_ndcg5": next(
            (t["val_ndcg5"] for t in trials if t["params"] == spec.default), None
        ),
        "default_test_ndcg5": _test_ndcg(default_metrics),
        "best_params": best_params,
        "best_val_ndcg5": best_val,
        "best_test_ndcg5": _test_ndcg(best_metrics),
        "n_trials": len(study.trials),
        "trials": sorted(trials, key=lambda t: -t["val_ndcg5"]),
    }


def _eval_on_test(
    method: str,
    tag: str,
    params: dict[str, Any],
    build: Callable[[dict[str, Any]], Scorer],
    task: Task,
    output_root: Path,
    tuning: dict[str, Any],
) -> dict[str, Any]:
    method_name = f"{method}_ranking_{tag}"
    output_dir = output_root / f"{task.name}_{method_name}"
    scorer_manifest = {
        "class": type(build(params)).__name__,
        "params": params,
        "tuning": {**tuning, "selection": tag},
    }
    return run_classical_ranking(
        task,
        output_dir,
        scorer=build(params),
        method_name=method_name,
        scorer_manifest=scorer_manifest,
        log_tag=f"{method}:{tag}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", default="ml-1m_cap20_eval_users1000_cg5_m19_seed0",
        help="Task name to tune on (val=selection, test=report).",
    )
    parser.add_argument(
        "--methods", default="item_knn,als,bpr,lightgcn",
        help="Comma-separated methods to tune.",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.task not in TASK_BUILDERS:
        raise ValueError(f"Unknown task: {args.task!r}")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHOD_SPECS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {sorted(METHOD_SPECS)}")

    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else repo_root() / "outputs" / "in_distribution" / "interaction_prediction" / "tuning"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    t0 = perf_counter()
    print(f"Building task: {args.task}", flush=True)
    task = TASK_BUILDERS[args.task]()
    print(f"Built task in {perf_counter() - t0:.1f}s", flush=True)

    group_col = task.schema.candidate_group_column
    if group_col is None:
        raise ValueError("Task has no candidate_group_column")
    xy = task_xy(task)

    summary: dict[str, Any] = {"task": args.task, "seed": SEED, "methods": {}}
    for method in methods:
        print(f"\n===== tuning {method} =====", flush=True)
        summary["methods"][method] = tune_method(
            method, METHOD_SPECS[method], task, xy, group_col, output_root
        )

    summary_path = output_root / f"tuning_summary_{args.task}.json"
    write_json(summary_path, summary)
    print(f"\nWrote tuning summary -> {summary_path}", flush=True)
    for method, res in summary["methods"].items():
        print(
            f"  {method:9s} test ndcg@5: default={res['default_test_ndcg5']:.4f} "
            f"-> tuned={res['best_test_ndcg5']:.4f}  best={res['best_params']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
