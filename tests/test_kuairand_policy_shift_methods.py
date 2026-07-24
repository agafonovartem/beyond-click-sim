from __future__ import annotations

from pathlib import Path
import sys

from beyond_click_sim.scorers import (
    BPRScorer,
    ExposureLogisticBiasScorer,
    ItemKNNScorer,
    LLMTargetResponseProbabilityScorer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.ood.kuairand_policy_shift.methods import (
    CLASSICAL_MODEL_SEED,
    METHOD_REGISTRY,
    make_scorer,
)


def test_item_knn_is_registered_with_reproducible_hyperparameters() -> None:
    spec = METHOD_REGISTRY["item_knn"]

    scorer = make_scorer(spec, seed=999)
    manifest = spec.manifest_config()

    assert isinstance(scorer, ItemKNNScorer)
    assert scorer.n_neighbors == 20
    assert scorer.aggregation == "sum"
    assert manifest["model_seed"] is None
    assert manifest["hyperparameters"] == {
        "aggregation": "sum",
        "cold_candidate_policy": "zero_score_with_coverage_report",
        "n_neighbors": 20,
        "positive_source_interactions_only": True,
    }


def test_bpr_model_seed_does_not_follow_candidate_task_seed() -> None:
    spec = METHOD_REGISTRY["bpr"]

    seed_42 = make_scorer(spec, seed=42)
    seed_44 = make_scorer(spec, seed=44)

    assert isinstance(seed_42, BPRScorer)
    assert isinstance(seed_44, BPRScorer)
    assert seed_42.seed == CLASSICAL_MODEL_SEED
    assert seed_44.seed == CLASSICAL_MODEL_SEED
    assert spec.manifest_config()["model_seed"] == CLASSICAL_MODEL_SEED
    assert spec.manifest_config()["hyperparameters"] == {
        "cold_candidate_policy": "zero_score_with_coverage_report",
        "iterations": 100,
        "learning_rate": 0.01,
        "n_factors": 64,
        "positive_source_interactions_only": True,
        "regularization": 0.01,
    }


def test_exposure_logistic_and_probability_llm_are_registered() -> None:
    logistic_spec = METHOD_REGISTRY["exposure_logistic_bias"]
    logistic = make_scorer(logistic_spec, seed=999)

    assert isinstance(logistic, ExposureLogisticBiasScorer)
    assert logistic.seed == CLASSICAL_MODEL_SEED
    assert logistic_spec.score_semantics == "probability"

    probability_spec = METHOD_REGISTRY[
        "llm_openai_vk_gpt56sol_probability_caption_categories"
    ]
    probability = make_scorer(
        probability_spec,
        seed=0,
        client=object(),
    )

    assert isinstance(probability, LLMTargetResponseProbabilityScorer)
    assert probability_spec.model == "gpt-5.6-sol"
    assert probability_spec.max_completion_tokens == 48
    assert probability_spec.expected_output_tokens == 40
    assert probability_spec.score_semantics == "probability"
