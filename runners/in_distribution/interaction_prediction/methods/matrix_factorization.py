"""ALS and BPR matrix-factorization baselines for interaction prediction."""

from __future__ import annotations

from pathlib import Path

from beyond_click_sim.scorers import ALSScorer, BPRScorer
from beyond_click_sim.tasks import Task
from runners.in_distribution.interaction_prediction.methods.classical import (
    run_classical_pointwise,
    run_classical_ranking,
)


# --- ALS ---------------------------------------------------------------------
ALS_N_FACTORS = 64
ALS_ITERATIONS = 20
ALS_REGULARIZATION = 0.01
ALS_SEED = 0
ALS_METHOD_NAME = "als_f1_threshold"
ALS_RANKING_METHOD_NAME = "als_ranking"


def _als_scorer() -> ALSScorer:
    return ALSScorer(
        n_factors=ALS_N_FACTORS,
        iterations=ALS_ITERATIONS,
        regularization=ALS_REGULARIZATION,
        seed=ALS_SEED,
        item_column="item_id",
        user_column="user_id",
    )


def _als_manifest() -> dict[str, object]:
    return {
        "class": "ALSScorer",
        "n_factors": ALS_N_FACTORS,
        "iterations": ALS_ITERATIONS,
        "regularization": ALS_REGULARIZATION,
        "seed": ALS_SEED,
        "num_threads": 1,
        "item_column": "item_id",
        "user_column": "user_id",
    }


def run_als(task: Task, output_dir: Path) -> dict[str, object]:
    """Run the ALS baseline for pointwise interaction alignment."""
    return run_classical_pointwise(
        task, output_dir,
        scorer=_als_scorer(),
        method_name=ALS_METHOD_NAME,
        scorer_manifest=_als_manifest(),
        log_tag="als",
    )


def run_als_ranking(task: Task, output_dir: Path) -> dict[str, object]:
    """Run the ALS baseline for raw-score ranking."""
    return run_classical_ranking(
        task, output_dir,
        scorer=_als_scorer(),
        method_name=ALS_RANKING_METHOD_NAME,
        scorer_manifest=_als_manifest(),
        log_tag="als",
    )


# --- BPR ---------------------------------------------------------------------
BPR_N_FACTORS = 64
BPR_LEARNING_RATE = 0.01
BPR_REGULARIZATION = 0.01
BPR_ITERATIONS = 100
BPR_SEED = 0
BPR_METHOD_NAME = "bpr_f1_threshold"
BPR_RANKING_METHOD_NAME = "bpr_ranking"


def _bpr_scorer() -> BPRScorer:
    return BPRScorer(
        n_factors=BPR_N_FACTORS,
        learning_rate=BPR_LEARNING_RATE,
        regularization=BPR_REGULARIZATION,
        iterations=BPR_ITERATIONS,
        seed=BPR_SEED,
        item_column="item_id",
        user_column="user_id",
    )


def _bpr_manifest() -> dict[str, object]:
    return {
        "class": "BPRScorer",
        "n_factors": BPR_N_FACTORS,
        "learning_rate": BPR_LEARNING_RATE,
        "regularization": BPR_REGULARIZATION,
        "iterations": BPR_ITERATIONS,
        "seed": BPR_SEED,
        "num_threads": 1,
        "item_column": "item_id",
        "user_column": "user_id",
    }


def run_bpr(task: Task, output_dir: Path) -> dict[str, object]:
    """Run the BPR baseline for pointwise interaction alignment."""
    return run_classical_pointwise(
        task, output_dir,
        scorer=_bpr_scorer(),
        method_name=BPR_METHOD_NAME,
        scorer_manifest=_bpr_manifest(),
        log_tag="bpr",
    )


def run_bpr_ranking(task: Task, output_dir: Path) -> dict[str, object]:
    """Run the BPR baseline for raw-score ranking."""
    return run_classical_ranking(
        task, output_dir,
        scorer=_bpr_scorer(),
        method_name=BPR_RANKING_METHOD_NAME,
        scorer_manifest=_bpr_manifest(),
        log_tag="bpr",
    )
