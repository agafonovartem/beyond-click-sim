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
    POINTWISE_MAIN_METRIC,
    POINTWISE_METRICS_FILENAME,
    RANKING_MAIN_METRIC,
    RANKING_METRICS_FILENAME,
)
from runners.in_distribution.interaction_prediction.methods.item_knn import (  # noqa: E402
    run,
    run_ranking,
)


def _toy_task() -> Task:
    """Toy task with a clean CF signal for ItemKNN.

    A and C co-occur across u2..u4, so they are similar; B shares no user with
    either, so its cosine similarity to A is 0. Eval users are warm (present in
    train) because ItemKNN builds each user's profile from the train split.

    val  : u1 (profile {A}) ranks C (positive) over B (zero-similarity negative)
    test : u6 (profile {A}) ranks C (positive) over COLD, an item absent from
           train, which also exercises the cold-item coverage counter.

    The val and test users deliberately have equal-size profiles: the default
    "mean" aggregation divides by profile size, so a threshold picked on a
    1-item-profile user would not transfer to a 2-item-profile user.
    """
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u2", "u3", "u3", "u4", "u4", "u5", "u6"],
            "item_id": ["A", "A", "C", "A", "C", "A", "C", "B", "A"],
            "target": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    )
    val = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_id": ["C", "B"],
            "candidate_group": ["val-g1", "val-g1"],
            "target": [1, 0],
        }
    )
    test = pd.DataFrame(
        {
            "user_id": ["u6", "u6"],
            "item_id": ["COLD", "C"],
            "candidate_group": ["test-g1", "test-g1"],
            "target": [0, 1],
        }
    )
    return Task(
        name="toy",
        train=train,
        val=val,
        test=test,
        schema=TaskSchema(
            target_column="target",
            candidate_group_column="candidate_group",
            feature_columns=("user_id", "item_id"),
        ),
        manifest={"dataset": "toy", "seed": 0},
    )


def test_classical_pointwise_runner_writes_pointwise_artifact_only(tmp_path: Path) -> None:
    metrics = run(_toy_task(), tmp_path)

    assert (tmp_path / POINTWISE_METRICS_FILENAME).exists()
    assert not (tmp_path / RANKING_METRICS_FILENAME).exists()
    assert (tmp_path / "predictions.parquet").exists()

    assert metrics["method"] == "item_knn_f1_threshold"
    assert metrics["main_metric"] == POINTWISE_MAIN_METRIC
    # The positive is the only candidate with nonzero similarity in each group,
    # so the val-selected threshold separates them perfectly on test too.
    assert metrics["test"]["macro_by_user_group_mean"]["f1"] == 1.0
    assert "threshold" in metrics

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"]["class"] == "ItemKNNScorer"
    assert manifest["scorer"]["n_neighbors"] == 20
    assert manifest["decision_rule"]["kind"] == "threshold_on_validation"

    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert {"score", "prediction"} <= set(predictions.columns)


def test_classical_ranking_runner_writes_ranking_artifact_only(tmp_path: Path) -> None:
    metrics = run_ranking(_toy_task(), tmp_path)

    assert (tmp_path / RANKING_METRICS_FILENAME).exists()
    assert not (tmp_path / POINTWISE_METRICS_FILENAME).exists()
    assert (tmp_path / "predictions.parquet").exists()

    assert metrics["method"] == "item_knn_ranking"
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


def test_classical_runner_records_coverage_diagnostics(tmp_path: Path) -> None:
    """Coverage must reach both metrics and manifest: it is how the paper can
    report that CF scorers abstain (score 0) where popularity/LLM score everything.
    """
    metrics = run_ranking(_toy_task(), tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert metrics["coverage"] == manifest["coverage"]

    test_coverage = metrics["coverage"]["test"]
    assert test_coverage["rows"] == 2
    assert test_coverage["cold_user_rows"] == 0  # u6 is in train
    assert test_coverage["cold_item_rows"] == 1  # COLD is not
    assert test_coverage["nonzero_score_rows"] == 1
    assert test_coverage["nonzero_score_fraction"] == 0.5
    assert test_coverage["positive_rows"] == 1
    assert test_coverage["positive_nonzero_score_fraction"] == 1.0

    val_coverage = metrics["coverage"]["val"]
    assert val_coverage["cold_user_rows"] == 0
    assert val_coverage["cold_item_rows"] == 0  # B is in train, just dissimilar
    assert val_coverage["positive_nonzero_score_fraction"] == 1.0
