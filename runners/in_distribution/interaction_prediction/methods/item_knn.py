"""ItemKNN collaborative-filtering baseline for interaction prediction."""

from __future__ import annotations

from pathlib import Path

from beyond_click_sim.scorers import ItemKNNScorer
from beyond_click_sim.tasks import Task
from runners.in_distribution.interaction_prediction.methods.classical import (
    run_classical_pointwise,
    run_classical_ranking,
)


N_NEIGHBORS = 20
AGGREGATION = "mean"
METHOD_NAME = "item_knn_f1_threshold"
RANKING_METHOD_NAME = "item_knn_ranking"


def _scorer() -> ItemKNNScorer:
    return ItemKNNScorer(
        n_neighbors=N_NEIGHBORS,
        aggregation=AGGREGATION,
        item_column="item_id",
        user_column="user_id",
    )


def _scorer_manifest() -> dict[str, object]:
    return {
        "class": "ItemKNNScorer",
        "n_neighbors": N_NEIGHBORS,
        "aggregation": AGGREGATION,
        "item_column": "item_id",
        "user_column": "user_id",
    }


def run(task: Task, output_dir: Path) -> dict[str, object]:
    """Run the ItemKNN baseline for pointwise interaction alignment."""
    return run_classical_pointwise(
        task,
        output_dir,
        scorer=_scorer(),
        method_name=METHOD_NAME,
        scorer_manifest=_scorer_manifest(),
        log_tag="item_knn",
    )


def run_ranking(task: Task, output_dir: Path) -> dict[str, object]:
    """Run the ItemKNN baseline for raw-score ranking."""
    return run_classical_ranking(
        task,
        output_dir,
        scorer=_scorer(),
        method_name=RANKING_METHOD_NAME,
        scorer_manifest=_scorer_manifest(),
        log_tag="item_knn",
    )
