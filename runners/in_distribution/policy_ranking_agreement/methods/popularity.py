from __future__ import annotations

from pathlib import Path
from time import perf_counter

from beyond_click_sim.evaluation.policy_ranking import policy_ranking_agreement_metrics
from beyond_click_sim.scorers.popularity import PopularityScorer
from beyond_click_sim.tasks import Task

from runners.in_distribution.policy_ranking_agreement.metrics import (
    MAIN_METRIC,
    METRICS_FILENAME,
    UTILITY_AGGREGATION,
)
from runners.in_distribution.policy_ranking_agreement.methods.common import (
    compute_policy_utilities,
    current_git_commit,
    json_safe,
    task_xy,
    write_policy_metrics,
    write_json,
)

METHOD_NAME = "popularity_scorer"


def run(task: Task, output_dir: Path) -> dict[str, object]:
    """Score each policy's recommendation list using train item popularity.

    Fits a PopularityScorer on training interactions and uses it to assign a
    simulated score to every (user, recommended_item) row in the test set.
    Aggregates mean simulated score and mean real target per policy, then
    computes Kendall's tau and Spearman's rho between the resulting policy
    rankings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_times: dict[str, float] = {}

    t = perf_counter()
    xy = task_xy(task)
    X_train, y_train = xy["train"]
    X_test, y_test = xy["test"]
    _record(stage_times, "prepare_xy", t)

    t = perf_counter()
    scorer = PopularityScorer(item_column="item_id").fit(X_train, y_train)
    _record(stage_times, "fit", t)

    t = perf_counter()
    test_scores = scorer.score(X_test)
    _record(stage_times, "score_test", t)

    t = perf_counter()
    simulated_utilities, real_utilities = compute_policy_utilities(
        X_test, y_test, test_scores, policy_column="policy"
    )
    policy_names = sorted(simulated_utilities.keys())
    sim_vals = [simulated_utilities[p] for p in policy_names]
    real_vals = [real_utilities[p] for p in policy_names]
    _record(stage_times, "aggregate_utilities", t)

    t = perf_counter()
    agreement = policy_ranking_agreement_metrics(policy_names, sim_vals, real_vals)
    _record(stage_times, "rank_correlation", t)

    t = perf_counter()
    predictions = X_test.copy()
    predictions.insert(0, "split", "test")
    predictions["target"] = y_test.to_numpy()
    predictions["score"] = test_scores.to_numpy()
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    _record(stage_times, "write_predictions", t)

    t = perf_counter()
    from runners.in_distribution.policy_ranking_agreement.task_builders import repo_root

    manifest = {
        "method": METHOD_NAME,
        "protocol": "policy_ranking",
        "scorer": {"class": "PopularityScorer", "item_column": "item_id"},
        "utility_aggregation": UTILITY_AGGREGATION,
        "task": {"name": task.name, "manifest": json_safe(task.manifest)},
        "stage_times_seconds": stage_times,
        "git_commit": current_git_commit(repo_root()),
    }
    metrics = {
        "method": METHOD_NAME,
        "task": task.name,
        "protocol": "policy_ranking",
        "main_metric": MAIN_METRIC,
        "utility_aggregation": UTILITY_AGGREGATION,
        "test": agreement,
        "stage_times_seconds": stage_times,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / METRICS_FILENAME, metrics)
    write_policy_metrics(task, output_dir)
    _record(stage_times, "write_metadata", t)

    return metrics


def _record(stage_times: dict[str, float], stage: str, t0: float) -> None:
    from time import perf_counter

    stage_times[stage] = perf_counter() - t0
