"""LightGCN baseline for interaction prediction."""

from __future__ import annotations

from pathlib import Path

from beyond_click_sim.scorers import LightGCNScorer
from beyond_click_sim.tasks import Task
from runners.in_distribution.interaction_prediction.methods.classical import (
    run_classical_pointwise,
    run_classical_ranking,
)


N_FACTORS = 64
N_LAYERS = 3
LEARNING_RATE = 0.001
REGULARIZATION = 1e-4
ITERATIONS = 200
BATCH_SIZE = 2048
SEED = 0
METHOD_NAME = "lightgcn_f1_threshold"
RANKING_METHOD_NAME = "lightgcn_ranking"


def _scorer() -> LightGCNScorer:
    return LightGCNScorer(
        n_factors=N_FACTORS,
        n_layers=N_LAYERS,
        learning_rate=LEARNING_RATE,
        regularization=REGULARIZATION,
        iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        seed=SEED,
        item_column="item_id",
        user_column="user_id",
    )


def _scorer_manifest() -> dict[str, object]:
    return {
        "class": "LightGCNScorer",
        "n_factors": N_FACTORS,
        "n_layers": N_LAYERS,
        "learning_rate": LEARNING_RATE,
        "regularization": REGULARIZATION,
        "iterations": ITERATIONS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "item_column": "item_id",
        "user_column": "user_id",
    }


def run(task: Task, output_dir: Path) -> dict[str, object]:
    """Run the LightGCN baseline for pointwise interaction alignment."""
    return run_classical_pointwise(
        task, output_dir,
        scorer=_scorer(),
        method_name=METHOD_NAME,
        scorer_manifest=_scorer_manifest(),
        log_tag="lightgcn",
    )


def run_ranking(task: Task, output_dir: Path) -> dict[str, object]:
    """Run the LightGCN baseline for raw-score ranking."""
    return run_classical_ranking(
        task, output_dir,
        scorer=_scorer(),
        method_name=RANKING_METHOD_NAME,
        scorer_manifest=_scorer_manifest(),
        log_tag="lightgcn",
    )
