from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from beyond_click_sim.scorers import (
    LLMTargetResponseListwiseRankingScorer,
    PopularityScorer,
)
from beyond_click_sim.scorers.history.prompts import (
    TARGET_RESPONSE_LISTWISE_RANKING_SYSTEM_PROMPT,
    TARGET_RESPONSE_LISTWISE_RANKING_USER_PROMPT_TEMPLATE,
)
from runners.ood.kuairand_policy_shift.methods import (
    ENGLISH_SEMANTIC_COLUMNS,
    MAX_HISTORY_ITEMS,
    METHOD_REGISTRY,
    PARSER_FORMAT_VERSION,
    RANKING_EXPECTED_OUTPUT_TOKENS,
    RANKING_MAX_COMPLETION_TOKENS,
    SCORER_FORMAT_VERSION,
    TARGET_DESCRIPTION,
    make_scorer,
)


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="C2\nC1"),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=4,
                total_tokens=104,
            ),
        )


class _FakeClient:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_registry_contains_only_the_fixed_first_suite_methods() -> None:
    assert set(METHOD_REGISTRY) == {
        "popularity",
        "llm_openai_vk_gpt55_listwise_english",
    }


def test_popularity_uses_full_standard_source_positive_counts() -> None:
    spec = METHOD_REGISTRY["popularity"]
    scorer = make_scorer(spec, seed=999)

    assert isinstance(scorer, PopularityScorer)
    assert spec.history_scope == "full_standard_source"
    assert spec.supported_controls == ("observed",)
    assert spec.supported_target_pools == ("common_warm",)
    assert spec.supports_control("observed")
    assert not spec.supports_control("none")
    assert spec.supports_target_pool("common_warm")
    assert not spec.supports_target_pool("full_pool")

    source = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u1"],
            "item_id": ["a", "a", "a", "b"],
        }
    )
    targets = pd.Series([1, 0, 1, 1], dtype=int)
    scorer.fit(source, targets)

    candidates = pd.DataFrame({"item_id": ["a", "b", "cold"]})
    assert scorer.score(candidates).tolist() == [2.0, 1.0, 0.0]
    assert spec.manifest_config()["hyperparameters"] == {
        "cold_item_score": 0.0,
        "statistic": "sum_of_positive_standard_source_targets",
    }


def test_gpt55_spec_encodes_visible_h20_controls_and_target_pools() -> None:
    spec = METHOD_REGISTRY[
        "llm_openai_vk_gpt55_listwise_english"
    ]
    manifest = spec.manifest_config()

    assert spec.is_llm
    assert spec.is_paid
    assert spec.client_name == "openai_vk_proxy"
    assert spec.model == "gpt-5.5"
    assert spec.history_scope == "visible_h20"
    assert spec.supported_controls == ("observed", "none")
    assert spec.supported_target_pools == ("common_warm",)
    assert spec.scorer_format_version == SCORER_FORMAT_VERSION
    assert spec.parser_format_version == PARSER_FORMAT_VERSION
    assert spec.max_completion_tokens == RANKING_MAX_COMPLETION_TOKENS == 64
    assert spec.expected_output_tokens == RANKING_EXPECTED_OUTPUT_TOKENS == 32
    assert spec.score_semantics == "ranking_score"
    assert manifest["history_scope"] == "visible_h20"
    assert manifest["supported_controls"] == ["observed", "none"]
    assert manifest["supported_target_pools"] == ["common_warm"]
    assert manifest["scorer_format_version"] == SCORER_FORMAT_VERSION
    assert manifest["parser_format_version"] == PARSER_FORMAT_VERSION
    assert manifest["hyperparameters"]["history_rows_per_user"] == 20


def test_gpt55_scorer_uses_english_semantics_and_target_response_prompt() -> None:
    client = _FakeClient()
    spec = METHOD_REGISTRY[
        "llm_openai_vk_gpt55_listwise_english"
    ]
    scorer = make_scorer(spec, seed=0, client=client)

    assert isinstance(scorer, LLMTargetResponseListwiseRankingScorer)
    assert scorer.history_description_columns == (
        *ENGLISH_SEMANTIC_COLUMNS,
        "history_click",
    )
    assert scorer.candidate_description_columns == ENGLISH_SEMANTIC_COLUMNS
    assert scorer.max_history_items == MAX_HISTORY_ITEMS == 20
    assert scorer.target_description == TARGET_DESCRIPTION
    assert scorer.temperature is None
    assert scorer.max_tokens is None
    assert scorer.request_kwargs == {
        "max_completion_tokens": 64,
        "reasoning_effort": "none",
    }

    history = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_caption_en": "A source music video",
                "item_cover_text_en": "Live performance",
                "item_category_path_en": "Music > Performances",
                "history_click": 1,
            }
        ]
    )
    scorer.fit(history, pd.Series([1], dtype=int))
    candidates = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "candidate_group": "g1",
                "item_caption_en": "A candidate comedy clip",
                "item_cover_text_en": "Unexpected ending",
                "item_category_path_en": "Comedy > Sketches",
            },
            {
                "user_id": "u1",
                "candidate_group": "g1",
                "item_caption_en": "A candidate concert clip",
                "item_cover_text_en": "Encore",
                "item_category_path_en": "Music > Performances",
            },
        ]
    )
    messages = scorer._build_messages(
        user_id="u1",
        candidates=candidates,
        labels=["C1", "C2"],
    )

    expected_history = (
        "H1. English caption: A source music video; "
        "English cover text: Live performance; "
        "English category: Music > Performances; "
        "observed positive interaction after exposure (1=yes, 0=no): 1"
    )
    expected_candidates = "\n".join(
        [
            "C1. English caption: A candidate comedy clip; "
            "English cover text: Unexpected ending; "
            "English category: Comedy > Sketches",
            "C2. English caption: A candidate concert clip; "
            "English cover text: Encore; "
            "English category: Music > Performances",
        ]
    )
    assert messages == [
        {
            "role": "system",
            "content": TARGET_RESPONSE_LISTWISE_RANKING_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": TARGET_RESPONSE_LISTWISE_RANKING_USER_PROMPT_TEMPLATE.format(
                history=expected_history,
                target_description=TARGET_DESCRIPTION,
                candidates=expected_candidates,
                output_labels="C1, C2",
                candidate_count=2,
            ),
        },
    ]
    prompt = "\n".join(message["content"] for message in messages)
    assert "Positive-preference target" not in prompt
    assert "Response target:" in prompt
    assert "item_id" not in prompt
    assert "user_id" not in prompt
    candidate_prompt = prompt.split("Candidate items:", maxsplit=1)[1]
    assert "observed positive interaction" not in candidate_prompt

    scores = scorer.score(candidates)
    assert scores.tolist() == [0.0, 1.0]
    request = client.completions.calls[0]
    assert request["model"] == "gpt-5.5"
    assert request["max_completion_tokens"] == 64
    assert request["reasoning_effort"] == "none"
    assert "temperature" not in request
    assert "max_tokens" not in request


def test_gpt55_requires_a_client() -> None:
    spec = METHOD_REGISTRY[
        "llm_openai_vk_gpt55_listwise_english"
    ]

    with pytest.raises(ValueError, match="require a client"):
        make_scorer(spec, seed=0)
