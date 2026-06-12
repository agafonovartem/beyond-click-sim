from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

from beyond_click_sim.tasks import Task, TaskSchema

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.in_distribution.regression_prediction.methods.mode import run  # noqa: E402


def test_mode_regression_runner_writes_regression_artifacts(tmp_path: Path) -> None:
    task = Task(
        name="toy_rating_eval_users2_seed0",
        train=pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2", "u2"],
                "item_id": ["i1", "i2", "i3", "i4"],
                "target": [1.0, 4.0, 4.0, 5.0],
            }
        ),
        val=pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "item_id": ["i5", "i6"],
                "target": [4.0, 5.0],
            }
        ),
        test=pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2"],
                "item_id": ["i7", "i8", "i9"],
                "target": [1.0, 4.0, 5.0],
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
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "predictions.parquet").exists()
    assert metrics["method"] == "mode_regressor"
    assert metrics["test"]["micro"]["mae"] == 4 / 3
    assert "threshold" not in metrics
    assert "candidate_groups" not in metrics

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"] == {
        "class": "ModeRegressor",
        "mode": 4.0,
        "tie_break": "smallest",
    }
    assert "decision_rule" not in manifest

    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert predictions["split"].tolist() == ["val", "val", "test", "test", "test"]
    assert predictions["score"].tolist() == [4.0] * 5
    assert "prediction" not in predictions.columns
