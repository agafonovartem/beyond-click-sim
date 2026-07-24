from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from beyond_click_sim.scorers import (
    BPRScorer,
    ExposureLogisticBiasScorer,
    HistoryTextAffinityScorer,
    ItemMeanRegressor,
    ItemKNNScorer,
    LLMTargetResponseListwiseRankingScorer,
    LLMTargetResponseProbabilityScorer,
    MeanRegressor,
    Scorer,
    SmoothedUserCTRScorer,
    UserMeanRegressor,
)


MetadataCondition = Literal["caption_categories", "categories_only", "not_applicable"]
MethodKind = Literal[
    "mean",
    "item_mean",
    "user_mean",
    "user_ctr_smoothed",
    "item_knn",
    "bpr",
    "exposure_logistic",
    "text_affinity",
    "llm",
]
LLMOutputMode = Literal["ranking", "probability"]
ScoreSemantics = Literal["ranking_score", "probability"]

TARGET_DESCRIPTION = (
    "the user clicks the displayed short video after this exposure"
)
MAX_HISTORY_ITEMS: None = None
RANKING_MAX_COMPLETION_TOKENS = 24
RANKING_EXPECTED_OUTPUT_TOKENS = 18
PROBABILITY_MAX_COMPLETION_TOKENS = 48
PROBABILITY_EXPECTED_OUTPUT_TOKENS = 40
TEMPERATURE = 0.0
QWEN_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
CLASSICAL_MODEL_SEED = 42
ITEM_KNN_N_NEIGHBORS = 20
ITEM_KNN_AGGREGATION = "sum"
BPR_N_FACTORS = 64
BPR_LEARNING_RATE = 0.01
BPR_REGULARIZATION = 0.01
BPR_ITERATIONS = 100


@dataclass(frozen=True)
class MethodSpec:
    name: str
    kind: MethodKind
    metadata_condition: MetadataCondition = "not_applicable"
    client_name: str | None = None
    model: str | None = None
    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0
    max_completion_tokens: int = RANKING_MAX_COMPLETION_TOKENS
    expected_output_tokens: int = RANKING_EXPECTED_OUTPUT_TOKENS
    llm_output_mode: LLMOutputMode = "ranking"
    score_semantics: ScoreSemantics = "ranking_score"
    extra_body: dict[str, Any] | None = None
    model_seed: int | None = None
    hyperparameters: tuple[tuple[str, Any], ...] = ()

    @property
    def is_llm(self) -> bool:
        return self.kind == "llm"

    @property
    def is_paid(self) -> bool:
        return self.input_price_per_million > 0 or self.output_price_per_million > 0

    @property
    def hyperparameter_dict(self) -> dict[str, Any]:
        return dict(self.hyperparameters)

    def manifest_config(self) -> dict[str, Any]:
        """Return the reproducibility fields written for every method run."""

        return {
            "name": self.name,
            "kind": self.kind,
            "metadata_condition": self.metadata_condition,
            "client_name": self.client_name,
            "model": self.model,
            "model_seed": self.model_seed,
            "hyperparameters": self.hyperparameter_dict,
            "max_completion_tokens": self.max_completion_tokens,
            "expected_output_tokens": self.expected_output_tokens,
            "llm_output_mode": self.llm_output_mode,
            "score_semantics": self.score_semantics,
        }

    def supports_control(self, control: str) -> bool:
        # Empty history is a semantic LLM control. Aggregate recommenders still
        # consume source-policy logs and therefore must not be mislabeled as
        # "no history".
        return control != "none" or self.is_llm


def make_scorer(
    spec: MethodSpec,
    *,
    seed: int,
    client: Any | None = None,
) -> Scorer:
    if spec.kind == "mean":
        return MeanRegressor()
    if spec.kind == "item_mean":
        return ItemMeanRegressor(item_column="item_id")
    if spec.kind == "user_mean":
        return UserMeanRegressor(
            history_value_column="history_click",
            user_column="user_id",
            max_history_items=None,
        )
    if spec.kind == "user_ctr_smoothed":
        return SmoothedUserCTRScorer(
            history_value_column="history_click",
            user_column="user_id",
            alpha=float(spec.hyperparameter_dict["alpha"]),
            beta=float(spec.hyperparameter_dict["beta"]),
        )
    if spec.kind == "item_knn":
        return ItemKNNScorer(
            n_neighbors=ITEM_KNN_N_NEIGHBORS,
            aggregation=ITEM_KNN_AGGREGATION,
            user_column="user_id",
            item_column="item_id",
            cold_score_policy="zero",
        )
    if spec.kind == "bpr":
        return BPRScorer(
            user_column="user_id",
            item_column="item_id",
            n_factors=BPR_N_FACTORS,
            learning_rate=BPR_LEARNING_RATE,
            regularization=BPR_REGULARIZATION,
            iterations=BPR_ITERATIONS,
            # Model randomness is fixed across candidate/task seeds.  Those
            # seeds change only the sampled evaluation pseudo-lists.
            seed=CLASSICAL_MODEL_SEED,
            cold_score_policy="zero",
        )
    if spec.kind == "exposure_logistic":
        return ExposureLogisticBiasScorer(
            user_column="user_id",
            item_column="item_id",
            alpha=float(spec.hyperparameter_dict["alpha"]),
            max_iter=int(spec.hyperparameter_dict["max_iter"]),
            tolerance=float(spec.hyperparameter_dict["tolerance"]),
            seed=CLASSICAL_MODEL_SEED,
        )
    if spec.kind == "text_affinity":
        return HistoryTextAffinityScorer(
            text_columns=_text_columns(spec.metadata_condition),
            user_column="user_id",
            cold_score_policy="zero",
        )
    if spec.kind != "llm":
        raise ValueError(f"Unsupported method kind: {spec.kind!r}")
    if client is None:
        raise ValueError("LLM methods require a client")
    history_columns = [
        *_text_columns(spec.metadata_condition),
        "history_click",
    ]
    uses_openai_reasoning_request = (
        spec.client_name == "openai_vk_proxy"
        and str(spec.model).startswith("gpt-5")
    )
    scorer_class = (
        LLMTargetResponseProbabilityScorer
        if spec.llm_output_mode == "probability"
        else LLMTargetResponseListwiseRankingScorer
    )
    return scorer_class(
        client=client,
        model=str(spec.model),
        history_description_columns=tuple(history_columns),
        candidate_description_columns=_text_columns(spec.metadata_condition),
        user_column="user_id",
        candidate_group_column="candidate_group",
        max_history_items=MAX_HISTORY_ITEMS,
        temperature=None if uses_openai_reasoning_request else TEMPERATURE,
        max_tokens=(
            None if uses_openai_reasoning_request else spec.max_completion_tokens
        ),
        column_labels={
            "item_description": "description (native Chinese)",
            "item_category_path": "categories",
            "history_click": "clicked after exposure (1=yes, 0=no)",
        },
        target_description=TARGET_DESCRIPTION,
        extra_body=spec.extra_body,
        request_kwargs=(
            {
                "max_completion_tokens": spec.max_completion_tokens,
                "reasoning_effort": "none",
            }
            if uses_openai_reasoning_request
            else None
        ),
    )


def _text_columns(
    condition: MetadataCondition,
) -> tuple[str, ...]:
    if condition == "caption_categories":
        return ("item_description", "item_category_path")
    if condition == "categories_only":
        return ("item_category_path",)
    raise ValueError(
        f"Method requires a semantic metadata condition, got {condition!r}"
    )


METHOD_REGISTRY: dict[str, MethodSpec] = {
    "global_mean": MethodSpec(
        "global_mean",
        "mean",
        score_semantics="probability",
    ),
    "item_ctr": MethodSpec(
        "item_ctr",
        "item_mean",
        score_semantics="probability",
    ),
    "user_ctr": MethodSpec(
        "user_ctr",
        "user_mean",
        score_semantics="probability",
    ),
    "user_ctr_laplace": MethodSpec(
        "user_ctr_laplace",
        "user_ctr_smoothed",
        score_semantics="probability",
        hyperparameters=(
            ("alpha", 1.0),
            ("beta", 1.0),
            ("selection", "fixed_before_target_scoring"),
            ("visible_signal", "five_binary_history_labels_only"),
        ),
    ),
    "item_knn": MethodSpec(
        "item_knn",
        "item_knn",
        hyperparameters=(
            ("n_neighbors", ITEM_KNN_N_NEIGHBORS),
            ("aggregation", ITEM_KNN_AGGREGATION),
            ("positive_source_interactions_only", True),
            ("cold_candidate_policy", "zero_score_with_coverage_report"),
        ),
    ),
    "bpr": MethodSpec(
        "bpr",
        "bpr",
        model_seed=CLASSICAL_MODEL_SEED,
        hyperparameters=(
            ("n_factors", BPR_N_FACTORS),
            ("learning_rate", BPR_LEARNING_RATE),
            ("regularization", BPR_REGULARIZATION),
            ("iterations", BPR_ITERATIONS),
            ("positive_source_interactions_only", True),
            ("cold_candidate_policy", "zero_score_with_coverage_report"),
        ),
    ),
    "exposure_logistic_bias": MethodSpec(
        "exposure_logistic_bias",
        "exposure_logistic",
        model_seed=CLASSICAL_MODEL_SEED,
        score_semantics="probability",
        hyperparameters=(
            ("alpha", 1e-5),
            ("max_iter", 1_000),
            ("tolerance", 1e-4),
            ("class_weight", None),
            ("uses_positive_and_negative_exposures", True),
            ("features", "one_hot_user_plus_item_biases"),
            ("cold_candidate_policy", "error"),
        ),
    ),
    "text_affinity_caption_categories": MethodSpec(
        "text_affinity_caption_categories",
        "text_affinity",
        metadata_condition="caption_categories",
        hyperparameters=(
            ("analyzer", "char"),
            ("ngram_range", (2, 5)),
            ("min_df", 1),
            ("max_features", None),
            ("vectorizer_fit_scope", "all_source_history_rows"),
            (
                "user_profile",
                "l2_normalized_mean_positive_minus_mean_negative",
            ),
            (
                "constant_label_source_users",
                "kept_for_vectorizer_zero_score_with_coverage_report",
            ),
        ),
    ),
    "text_affinity_categories_only": MethodSpec(
        "text_affinity_categories_only",
        "text_affinity",
        metadata_condition="categories_only",
        hyperparameters=(
            ("analyzer", "char"),
            ("ngram_range", (2, 5)),
            ("min_df", 1),
            ("max_features", None),
            ("vectorizer_fit_scope", "all_source_history_rows"),
            (
                "user_profile",
                "l2_normalized_mean_positive_minus_mean_negative",
            ),
            (
                "constant_label_source_users",
                "kept_for_vectorizer_skipped_for_profile",
            ),
        ),
    ),
    "llm_vllm_qwen36_27b_caption_categories": MethodSpec(
        "llm_vllm_qwen36_27b_caption_categories",
        "llm",
        metadata_condition="caption_categories",
        client_name="vllm_local",
        model="Qwen/Qwen3.6-27B",
        extra_body=QWEN_EXTRA_BODY,
    ),
    "llm_vllm_qwen36_27b_categories_only": MethodSpec(
        "llm_vllm_qwen36_27b_categories_only",
        "llm",
        metadata_condition="categories_only",
        client_name="vllm_local",
        model="Qwen/Qwen3.6-27B",
        extra_body=QWEN_EXTRA_BODY,
    ),
    "llm_vllm_qwen36_27b_probability_caption_categories": MethodSpec(
        "llm_vllm_qwen36_27b_probability_caption_categories",
        "llm",
        metadata_condition="caption_categories",
        client_name="vllm_local",
        model="Qwen/Qwen3.6-27B",
        extra_body=QWEN_EXTRA_BODY,
        max_completion_tokens=PROBABILITY_MAX_COMPLETION_TOKENS,
        expected_output_tokens=PROBABILITY_EXPECTED_OUTPUT_TOKENS,
        llm_output_mode="probability",
        score_semantics="probability",
    ),
    "llm_openai_vk_gpt54_mini_caption_categories": MethodSpec(
        "llm_openai_vk_gpt54_mini_caption_categories",
        "llm",
        metadata_condition="caption_categories",
        client_name="openai_vk_proxy",
        model="gpt-5.4-mini",
        input_price_per_million=0.75,
        output_price_per_million=4.50,
    ),
    "llm_openai_vk_gpt54_mini_categories_only": MethodSpec(
        "llm_openai_vk_gpt54_mini_categories_only",
        "llm",
        metadata_condition="categories_only",
        client_name="openai_vk_proxy",
        model="gpt-5.4-mini",
        input_price_per_million=0.75,
        output_price_per_million=4.50,
    ),
    "llm_openai_vk_gpt54_mini_probability_caption_categories": MethodSpec(
        "llm_openai_vk_gpt54_mini_probability_caption_categories",
        "llm",
        metadata_condition="caption_categories",
        client_name="openai_vk_proxy",
        model="gpt-5.4-mini",
        input_price_per_million=0.75,
        output_price_per_million=4.50,
        max_completion_tokens=PROBABILITY_MAX_COMPLETION_TOKENS,
        expected_output_tokens=PROBABILITY_EXPECTED_OUTPUT_TOKENS,
        llm_output_mode="probability",
        score_semantics="probability",
    ),
    "llm_openai_vk_gpt54_mini_probability_categories_only": MethodSpec(
        "llm_openai_vk_gpt54_mini_probability_categories_only",
        "llm",
        metadata_condition="categories_only",
        client_name="openai_vk_proxy",
        model="gpt-5.4-mini",
        input_price_per_million=0.75,
        output_price_per_million=4.50,
        max_completion_tokens=PROBABILITY_MAX_COMPLETION_TOKENS,
        expected_output_tokens=PROBABILITY_EXPECTED_OUTPUT_TOKENS,
        llm_output_mode="probability",
        score_semantics="probability",
    ),
    "llm_openai_vk_gpt56sol_probability_caption_categories": MethodSpec(
        "llm_openai_vk_gpt56sol_probability_caption_categories",
        "llm",
        metadata_condition="caption_categories",
        client_name="openai_vk_proxy",
        model="gpt-5.6-sol",
        input_price_per_million=5.0,
        output_price_per_million=30.0,
        max_completion_tokens=PROBABILITY_MAX_COMPLETION_TOKENS,
        expected_output_tokens=PROBABILITY_EXPECTED_OUTPUT_TOKENS,
        llm_output_mode="probability",
        score_semantics="probability",
    ),
}
