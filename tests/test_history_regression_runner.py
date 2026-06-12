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

from runners.in_distribution.regression_prediction.methods.history import (  # noqa: E402
    run_history_mean,
    run_history_mode,
)


def test_history_regression_runners_write_regression_artifacts(tmp_path: Path) -> None:
    task = Task(
        name="ml-1m_rating_eval_users2_seed0",
        train=pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1", "u2", "u2"],
                "item_id": ["i1", "i2", "i3", "i4", "i5"],
                "rating": [5, 4, 4, 1, 2],
                "target": [5.0, 4.0, 4.0, 1.0, 2.0],
            }
        ),
        val=pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "item_id": ["i6", "i7"],
                "target": [4.0, 2.0],
            }
        ),
        test=pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2"],
                "item_id": ["i8", "i9", "i10"],
                "target": [5.0, 4.0, 1.0],
            }
        ),
        schema=TaskSchema(
            target_column="target",
            feature_columns=("user_id", "item_id"),
        ),
        manifest={
            "protocol": "regression",
            "dataset": "ml-1m",
            "target_source_column": "target_rating",
        },
    )

    mean_dir = tmp_path / "mean"
    mode_dir = tmp_path / "mode"
    mean_metrics = run_history_mean(task, mean_dir)
    mode_metrics = run_history_mode(task, mode_dir)

    for run_dir, method_name in [
        (mean_dir, "history_mean_regressor"),
        (mode_dir, "history_mode_regressor"),
    ]:
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "predictions.parquet").exists()

        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["method"] == method_name
        assert manifest["evaluated_splits"] == ["val", "test"]
        assert manifest["scorer"]["history_value_column"] == "rating"
        assert manifest["scorer"]["history_window"] == {
            "source": "train_rows",
            "selection": "last_rows_in_input_order",
            "max_history_items": 20,
            "matched_llm_regressor_history": True,
        }
        assert "threshold" not in manifest
        assert "candidate_group" not in manifest["scorer"]

        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        assert set(metrics) >= {"val", "test"}
        assert "micro" in metrics["test"]
        assert "macro_by_user_mean" in metrics["test"]
        assert "candidate_groups" not in metrics

    mean_predictions = pd.read_parquet(mean_dir / "predictions.parquet")
    mode_predictions = pd.read_parquet(mode_dir / "predictions.parquet")
    assert mean_metrics["method"] == "history_mean_regressor"
    assert mode_metrics["method"] == "history_mode_regressor"
    assert mean_predictions["split"].tolist() == ["val", "val", "test", "test", "test"]
    assert mean_predictions["score"].tolist() == pytest.approx(
        [13 / 3, 1.5, 13 / 3, 13 / 3, 1.5]
    )
    assert mode_predictions["score"].tolist() == [4.0, 1.0, 4.0, 4.0, 1.0]
    assert "prediction" not in mean_predictions.columns
