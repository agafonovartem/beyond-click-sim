from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from beyond_click_sim.tasks import Task, TaskSchema

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.in_distribution.policy_ranking_agreement.metrics import (  # noqa: E402
    MAIN_METRIC,
    METRICS_FILENAME,
)
from runners.in_distribution.policy_ranking_agreement.methods.popularity import (  # noqa: E402
    run,
)


def _make_task() -> Task:
    """Build a minimal Task with two policies in the test set."""
    user_ids = ["u1", "u1", "u1", "u2", "u2", "u2"]
    item_ids = ["i1", "i2", "i3", "i4", "i5", "i6"]
    targets = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    policies = ["RandomPolicy", "RandomPolicy", "RandomPolicy", "PopularityPolicy", "PopularityPolicy", "PopularityPolicy"]
    ranks = [1, 2, 3, 1, 2, 3]
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3"],
            "item_id": ["i7", "i8", "i9", "i10", "i11"],
            "target": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    test = pd.DataFrame(
        {
            "user_id": user_ids,
            "item_id": item_ids,
            "target": targets,
            "policy": policies,
            "rank": ranks,
        }
    )
    return Task(
        name="toy_policy_ranking_seed0",
        train=train,
        val=pd.DataFrame(columns=train.columns),
        test=test,
        schema=TaskSchema(
            target_column="target",
            feature_columns=("user_id", "item_id"),
        ),
        manifest={
            "protocol": "policy_ranking",
            "dataset": "toy",
            "target_source_column": "target_interact",
        },
    )


def test_popularity_runner_writes_all_artifacts(tmp_path: Path) -> None:
    task = _make_task()
    metrics = run(task, tmp_path)

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / METRICS_FILENAME).exists()
    assert (tmp_path / "predictions.parquet").exists()


def test_popularity_runner_metrics_structure(tmp_path: Path) -> None:
    task = _make_task()
    metrics = run(task, tmp_path)

    assert metrics["protocol"] == "policy_ranking"
    assert metrics["method"] == "popularity_scorer"
    assert metrics["main_metric"] == MAIN_METRIC
    assert "test" in metrics
    test_m = metrics["test"]
    assert "n_policies" in test_m
    assert test_m["n_policies"] == 2
    # kendall_tau and spearman_rho may be None (K=2 case) or a float (K>=3)
    assert "kendall_tau" in test_m
    assert "spearman_rho" in test_m


def test_popularity_runner_predictions_no_prediction_column(tmp_path: Path) -> None:
    task = _make_task()
    run(task, tmp_path)

    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert "prediction" not in predictions.columns
    assert "score" in predictions.columns
    assert "target" in predictions.columns
    assert "split" in predictions.columns
    assert set(predictions["split"].unique()) == {"test"}


def test_popularity_runner_manifest_json_structure(tmp_path: Path) -> None:
    task = _make_task()
    run(task, tmp_path)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"] == "policy_ranking"
    assert manifest["method"] == "popularity_scorer"
    assert "scorer" in manifest
    assert manifest["scorer"]["class"] == "PopularityScorer"
