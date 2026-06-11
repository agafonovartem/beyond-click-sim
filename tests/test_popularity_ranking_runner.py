from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

from beyond_click_sim.tasks import Task, TaskSchema

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.in_distribution.interaction_prediction.metrics import (  # noqa: E402
    RANKING_MAIN_METRIC,
    RANKING_METRICS_FILENAME,
)
from runners.in_distribution.interaction_prediction.methods.popularity import run_ranking


def test_popularity_ranking_runner_writes_ranking_artifact_only(tmp_path: Path) -> None:
    task = Task(
        name="toy",
        train=pd.DataFrame(
            {
                "user_id": ["train-u1", "train-u2", "train-u3"],
                "item_id": ["i1", "i1", "i2"],
                "target": [1, 1, 1],
            }
        ),
        val=pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["i1", "i3"],
                "candidate_group": ["val-g1", "val-g1"],
                "target": [1, 0],
            }
        ),
        test=pd.DataFrame(
            {
                "user_id": ["u2", "u2"],
                "item_id": ["i3", "i1"],
                "candidate_group": ["test-g1", "test-g1"],
                "target": [0, 1],
            }
        ),
        schema=TaskSchema(
            target_column="target",
            candidate_group_column="candidate_group",
            feature_columns=("user_id", "item_id"),
        ),
        manifest={"dataset": "toy", "seed": 0},
    )

    metrics = run_ranking(task, tmp_path)

    assert not (tmp_path / "metrics.json").exists()
    assert (tmp_path / RANKING_METRICS_FILENAME).exists()
    assert (tmp_path / "predictions.parquet").exists()
    assert metrics["protocol"] == "ranking"
    assert metrics["main_metric"] == RANKING_MAIN_METRIC
    assert metrics["test"]["macro_by_user_group_mean"]["ndcg@1"] == 1.0
    assert "threshold" not in metrics

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"] == "ranking"
    assert "decision_rule" not in manifest

    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert "score" in predictions.columns
    assert "prediction" not in predictions.columns
