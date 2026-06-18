from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

from beyond_click_sim.tasks import Task, TaskSchema

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.in_distribution.regression_prediction.methods.item import (  # noqa: E402
    run_item_mean,
    run_item_mode,
)


def test_item_regression_runners_write_regression_artifacts(tmp_path: Path) -> None:
    task = Task(
        name="ml-1m_rating_eval_users2_seed0",
        train=pd.DataFrame(
            {
                "user_id": ["u1", "u2", "u1", "u2"],
                "item_id": ["i1", "i1", "i2", "i3"],
                "target": [5.0, 3.0, 1.0, 5.0],
            }
        ),
        val=pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "item_id": ["i1", "cold"],
                "target": [4.0, 2.0],
            }
        ),
        test=pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2"],
                "item_id": ["i2", "cold", "i1"],
                "target": [1.0, 3.0, 5.0],
            }
        ),
        schema=TaskSchema(
            target_column="target",
            feature_columns=("user_id", "item_id"),
            id_columns=("user_id", "item_id"),
        ),
        manifest={
            "protocol": "regression",
            "dataset": "ml-1m",
            "target_source_column": "target_rating",
        },
    )

    mean_dir = tmp_path / "mean"
    mode_dir = tmp_path / "mode"
    mean_metrics = run_item_mean(task, mean_dir)
    mode_metrics = run_item_mode(task, mode_dir)

    for run_dir, method_name, fallback_scorer, fallback_value in [
        (mean_dir, "item_mean_regressor", "MeanRegressor", 3.5),
        (mode_dir, "item_mode_regressor", "ModeRegressor", 5.0),
    ]:
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "predictions.parquet").exists()

        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["method"] == method_name
        assert manifest["evaluated_splits"] == ["val", "test"]
        assert manifest["cold_item_rows"] == {"test": 1, "val": 1}
        assert manifest["scorer"]["class"] in {"ItemMeanRegressor", "ItemModeRegressor"}
        assert manifest["scorer"]["item_column"] == "item_id"
        assert manifest["scorer"]["stat_source"] == "train_targets_grouped_by_item"
        assert manifest["scorer"]["cold_item_policy"] == "global_fallback"
        assert manifest["scorer"]["n_items_seen_train"] == 3
        assert manifest["scorer"]["fallback"] == {
            "scorer": fallback_scorer,
            "value": fallback_value,
        }

        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        assert set(metrics) >= {"val", "test", "cold_item_rows"}
        assert metrics["cold_item_rows"] == {"test": 1, "val": 1}
        assert "micro" in metrics["test"]
        assert "macro_by_user_mean" in metrics["test"]
        assert "candidate_groups" not in metrics

    mean_predictions = pd.read_parquet(mean_dir / "predictions.parquet")
    mode_predictions = pd.read_parquet(mode_dir / "predictions.parquet")
    assert mean_metrics["method"] == "item_mean_regressor"
    assert mode_metrics["method"] == "item_mode_regressor"
    assert mean_predictions["split"].tolist() == ["val", "val", "test", "test", "test"]
    assert mean_predictions["score"].tolist() == pytest.approx(
        [4.0, 3.5, 1.0, 3.5, 4.0]
    )
    assert mode_predictions["score"].tolist() == [3.0, 5.0, 1.0, 5.0, 3.0]
    assert "prediction" not in mean_predictions.columns
