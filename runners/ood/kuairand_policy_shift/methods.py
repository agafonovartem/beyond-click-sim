from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from beyond_click_sim.scorers import (
    LLMTargetResponseListwiseRankingScorer,
    PopularityScorer,
    Scorer,
)


MetadataCondition = Literal["english_semantics", "not_applicable"]
MethodKind = Literal["popularity", "llm"]
HistoryScope = Literal["full_standard_source", "visible_h20"]
HistoryControl = Literal["observed", "none"]
TargetPool = Literal["common_warm"]

TARGET_DESCRIPTION = (
    "the user produces a positive interaction after exposure to the displayed "
    "short video"
)
ENGLISH_SEMANTIC_COLUMNS = (
    "item_caption_en",
    "item_cover_text_en",
    "item_category_path_en",
)
MAX_HISTORY_ITEMS = 20
RANKING_MAX_COMPLETION_TOKENS = 64
RANKING_EXPECTED_OUTPUT_TOKENS = 32
SCORER_FORMAT_VERSION = "target-response-listwise-ranking-v1"
PARSER_FORMAT_VERSION = "complete-unique-candidate-label-ranking-v1"


@dataclass(frozen=True)
class MethodSpec:
    """Fixed method configuration for the KuaiRand standard-source suite."""

    name: str
    kind: MethodKind
    history_scope: HistoryScope
    supported_controls: tuple[HistoryControl, ...]
    supported_target_pools: tuple[TargetPool, ...]
    metadata_condition: MetadataCondition = "not_applicable"
    client_name: str | None = None
    model: str | None = None
    scorer_format_version: str | None = None
    parser_format_version: str | None = None
    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0
    max_completion_tokens: int = RANKING_MAX_COMPLETION_TOKENS
    expected_output_tokens: int = RANKING_EXPECTED_OUTPUT_TOKENS
    score_semantics: Literal["ranking_score"] = "ranking_score"
    extra_body: dict[str, Any] | None = None
    hyperparameters: tuple[tuple[str, Any], ...] = ()

    @property
    def is_llm(self) -> bool:
        return self.kind == "llm"

    @property
    def is_paid(self) -> bool:
        return (
            self.input_price_per_million > 0
            or self.output_price_per_million > 0
        )

    @property
    def hyperparameter_dict(self) -> dict[str, Any]:
        return dict(self.hyperparameters)

    def manifest_config(self) -> dict[str, Any]:
        """Return every method choice needed to interpret a run."""

        return {
            "name": self.name,
            "kind": self.kind,
            "history_scope": self.history_scope,
            "supported_controls": list(self.supported_controls),
            "supported_target_pools": list(self.supported_target_pools),
            "metadata_condition": self.metadata_condition,
            "client_name": self.client_name,
            "model": self.model,
            "scorer_format_version": self.scorer_format_version,
            "parser_format_version": self.parser_format_version,
            "hyperparameters": self.hyperparameter_dict,
            "max_completion_tokens": self.max_completion_tokens,
            "expected_output_tokens": self.expected_output_tokens,
            "score_semantics": self.score_semantics,
        }

    def supports_control(self, control: str) -> bool:
        return control in self.supported_controls

    def supports_target_pool(self, target_pool: str) -> bool:
        return target_pool in self.supported_target_pools


def make_scorer(
    spec: MethodSpec,
    *,
    seed: int,
    client: Any | None = None,
) -> Scorer:
    """Instantiate one of the two fixed first-suite scorers."""

    del seed  # Candidate resampling must not alter either scorer.
    if spec.kind == "popularity":
        return PopularityScorer(item_column="item_id")
    if spec.kind != "llm":
        raise ValueError(f"Unsupported method kind: {spec.kind!r}")
    if client is None:
        raise ValueError("LLM methods require a client")

    return LLMTargetResponseListwiseRankingScorer(
        client=client,
        model=str(spec.model),
        history_description_columns=(
            *ENGLISH_SEMANTIC_COLUMNS,
            "history_click",
        ),
        candidate_description_columns=ENGLISH_SEMANTIC_COLUMNS,
        user_column="user_id",
        candidate_group_column="candidate_group",
        max_history_items=MAX_HISTORY_ITEMS,
        # GPT-5 reasoning endpoints reject or ignore the legacy temperature
        # and max_tokens fields. Pass only their supported request parameters.
        temperature=None,
        max_tokens=None,
        column_labels={
            "item_caption_en": "English caption",
            "item_cover_text_en": "English cover text",
            "item_category_path_en": "English category",
            "history_click": (
                "observed positive interaction after exposure (1=yes, 0=no)"
            ),
        },
        target_description=TARGET_DESCRIPTION,
        request_kwargs={
            "max_completion_tokens": spec.max_completion_tokens,
            "reasoning_effort": "none",
        },
    )


METHOD_REGISTRY: dict[str, MethodSpec] = {
    "popularity": MethodSpec(
        name="popularity",
        kind="popularity",
        history_scope="full_standard_source",
        supported_controls=("observed",),
        supported_target_pools=("common_warm",),
        hyperparameters=(
            ("statistic", "sum_of_positive_standard_source_targets"),
            ("cold_item_score", 0.0),
        ),
    ),
    "llm_openai_vk_gpt54_mini_listwise_english": MethodSpec(
        name="llm_openai_vk_gpt54_mini_listwise_english",
        kind="llm",
        history_scope="visible_h20",
        supported_controls=("observed", "none"),
        supported_target_pools=("common_warm",),
        metadata_condition="english_semantics",
        client_name="openai_vk_proxy",
        model="gpt-5.4-mini",
        scorer_format_version=SCORER_FORMAT_VERSION,
        parser_format_version=PARSER_FORMAT_VERSION,
        input_price_per_million=0.75,
        output_price_per_million=4.50,
        hyperparameters=(
            ("history_rows_per_user", MAX_HISTORY_ITEMS),
            ("candidate_scoring", "one_listwise_request_per_group"),
            ("response_parser", "complete_unique_candidate_label_ranking"),
        ),
    ),
}
