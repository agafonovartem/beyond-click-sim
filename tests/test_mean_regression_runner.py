from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

from beyond_click_sim.tasks import Task, TaskSchema

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.in_distribution.regression_prediction.metrics import (  # noqa: E402
    REGRESSION_MAIN_METRIC,
    REGRESSION_METRICS_FILENAME,
)
from runners.in_distribution.regression_prediction.methods.mean import run  # noqa: E402


def test_mean_regression_runner_writes_regression_artifacts(tmp_path: Path) -> None:
    task = Task(
        name="toy_rating_eval_users2_seed0",
        train=pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2"],
                "item_id": ["i1", "i2", "i3"],
                "target": [1.0, 3.0, 5.0],
            }
        ),
        val=pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "item_id": ["i4", "i5"],
                "target": [2.0, 4.0],
            }
        ),
        test=pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2"],
                "item_id": ["i6", "i7", "i8"],
                "target": [1.0, 5.0, 3.0],
            }
        ),
        schema=TaskSchema(
            target_column="target",
            feature_columns=("user_id", "item_id"),
        ),
        manifest={
            "protocol": "regression",
            "dataset": "toy",
            "target_source_column": "target_rating",
        },
    )

    metrics = run(task, tmp_path)

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / REGRESSION_METRICS_FILENAME).exists()
    assert (tmp_path / "predictions.parquet").exists()
    assert metrics["protocol"] == "regression"
    assert metrics["main_metric"] == REGRESSION_MAIN_METRIC
    assert metrics["test"]["micro"]["mae"] == 4 / 3
    assert metrics["test"]["macro_by_user_mean"]["mae"] == 1.0
    assert "threshold" not in metrics
    assert "candidate_groups" not in metrics

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"] == "regression"
    assert manifest["scorer"] == {"class": "MeanRegressor", "mean": 3.0}
    assert "decision_rule" not in manifest
    assert "candidate_groups" not in manifest

    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert predictions["split"].tolist() == ["val", "val", "test", "test", "test"]
    assert predictions["score"].tolist() == [3.0] * 5
    assert "prediction" not in predictions.columns
