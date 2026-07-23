from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from beyond_click_sim.scorers import (
    LLMInteractionListwiseRankingScorer,
    LLMInteractionYesNoScorer,
    LLMPreferenceListwiseRankingScorer,
    LLMPreferenceYesNoScorer,
    LLMRegressor,
)
from runners.in_distribution.interaction_prediction.methods import (
    openp5_style as interaction_openp5,
)
from runners.in_distribution.interaction_prediction.methods import (
    METHOD_RUNNERS as INTERACTION_METHOD_RUNNERS,
)
from runners.in_distribution.preference_prediction.methods import (
    openp5_style as preference_openp5,
)
from runners.in_distribution.preference_prediction.methods import (
    METHOD_RUNNERS as PREFERENCE_METHOD_RUNNERS,
)
from runners.in_distribution.regression_prediction.methods import (
    METHOD_RUNNERS as REGRESSION_METHOD_RUNNERS,
)


class FakeChatCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.responses.pop(0)),
                )
            ]
        )


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        completions = FakeChatCompletions(responses)
        self.chat = SimpleNamespace(completions=completions)
        self.completions = completions


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_title": ["Toy Story", "Heat"],
            "rating": [5, 2],
        }
    )


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "candidate_group": ["g1", "g1"],
            "item_title": ["Aladdin", "Casino"],
        }
    )


def _assert_neutral_openp5_call(
    client: FakeClient,
    *,
    required_text: tuple[str, ...],
) -> None:
    messages = client.completions.calls[0]["messages"]
    assert messages == [{"role": "user", "content": messages[0]["content"]}]
    prompt = messages[0]["content"]
    for text in required_text:
        assert text in prompt
    lowered = prompt.lower()
    for forbidden in (
        "simulat",
        "role-play",
        "act as",
        "next item",
        "earlier",
        "later",
        "movielens",
        "steam",
    ):
        assert forbidden not in lowered


def _assert_openp5_metadata(scorer: object) -> None:
    metadata = scorer.prompt_metadata
    assert metadata["family"] == "openp5_style"
    assert metadata["version"] == "openp5_style_iid_v1"
    assert metadata["system_role"] is False
    assert metadata["temporal_language"] is False
    assert metadata["source"].endswith("eva/templates/template.txt")


def test_openp5_style_interaction_yes_no_prompt_is_neutral() -> None:
    client = FakeClient(["C1: yes\nC2: no"])
    scorer = LLMInteractionYesNoScorer(
        client=client,
        model="fake-model",
        history_description_columns=("item_title", "rating"),
        candidate_description_columns=("item_title",),
        prompt_family="openp5_style",
    ).fit(_history(), pd.Series([1, 0]))

    assert scorer.score(_candidates()).tolist() == [1.0, 0.0]
    _assert_neutral_openp5_call(
        client,
        required_text=(
            "Considering that the user has interacted",
            "H1. item_title: Toy Story; rating: 5",
            "C1. item_title: Aladdin",
        ),
    )
    _assert_openp5_metadata(scorer)


def test_openp5_style_preference_yes_no_prompt_is_neutral() -> None:
    client = FakeClient(["C1: yes\nC2: no"])
    scorer = LLMPreferenceYesNoScorer(
        client=client,
        model="fake-model",
        history_description_columns=("item_title", "rating"),
        candidate_description_columns=("item_title",),
        target_description="The user would rate the movie at least 4 out of 5.",
        prompt_family="openp5_style",
    ).fit(_history(), pd.Series([1, 0]))

    assert scorer.score(_candidates()).tolist() == [1.0, 0.0]
    _assert_neutral_openp5_call(
        client,
        required_text=(
            "Considering the user's observed feedback history",
            "Positive-preference target:",
            "rate the movie at least 4 out of 5",
        ),
    )
    _assert_openp5_metadata(scorer)


def test_openp5_style_interaction_listwise_prompt_is_neutral() -> None:
    client = FakeClient(["C2\nC1"])
    scorer = LLMInteractionListwiseRankingScorer(
        client=client,
        model="fake-model",
        history_description_columns=("item_title", "rating"),
        candidate_description_columns=("item_title",),
        prompt_family="openp5_style",
    ).fit(_history(), pd.Series([1, 0]))

    assert scorer.score(_candidates()).tolist() == [0.0, 1.0]
    _assert_neutral_openp5_call(
        client,
        required_text=(
            "Considering that the user has interacted",
            "Rank all candidate items",
            "exactly 2 non-empty lines",
        ),
    )
    _assert_openp5_metadata(scorer)


def test_openp5_style_preference_listwise_prompt_is_neutral() -> None:
    client = FakeClient(["C2\nC1"])
    scorer = LLMPreferenceListwiseRankingScorer(
        client=client,
        model="fake-model",
        history_description_columns=("item_title", "rating"),
        candidate_description_columns=("item_title",),
        target_description="The user would rate the movie at least 4 out of 5.",
        prompt_family="openp5_style",
    ).fit(_history(), pd.Series([1, 0]))

    assert scorer.score(_candidates()).tolist() == [0.0, 1.0]
    _assert_neutral_openp5_call(
        client,
        required_text=(
            "Considering the user's observed feedback history",
            "Positive-preference target:",
            "Rank all candidate items",
        ),
    )
    _assert_openp5_metadata(scorer)


def test_openp5_style_regression_prompt_is_neutral() -> None:
    client = FakeClient(["4"])
    scorer = LLMRegressor(
        client=client,
        model="fake-model",
        history_description_columns=("item_title", "rating"),
        candidate_description_columns=("item_title",),
        target_description="Predict the rating from 1 to 5.",
        output_instructions="Return exactly one integer from 1 to 5.",
        valid_values=(1, 2, 3, 4, 5),
        prompt_family="openp5_style",
    ).fit(_history(), pd.Series([5, 2]))

    X = _candidates().head(1)
    assert scorer.score(X).tolist() == [4.0]
    _assert_neutral_openp5_call(
        client,
        required_text=(
            "Considering the user's observed response history",
            "Candidate item:",
            "Predict the rating from 1 to 5.",
        ),
    )
    _assert_openp5_metadata(scorer)


def test_prompt_family_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="prompt_family"):
        LLMInteractionYesNoScorer(
            client=FakeClient([]),
            model="fake-model",
            item_description_columns=("item_title",),
            prompt_family="unknown",
        )


def test_openp5_style_smoke_methods_are_registered() -> None:
    assert (
        "llm_yes_no_openp5_style_ollama_qwen3_30b_a3b_smoke5"
        in INTERACTION_METHOD_RUNNERS
    )
    assert (
        "llm_listwise_ranking_openp5_style_ollama_qwen3_30b_a3b_smoke5"
        in INTERACTION_METHOD_RUNNERS
    )
    assert (
        "llm_preference_yes_no_openp5_style_ollama_qwen3_30b_a3b_smoke5"
        in PREFERENCE_METHOD_RUNNERS
    )
    assert (
        "llm_preference_listwise_ranking_openp5_style_"
        "ollama_qwen3_30b_a3b_smoke5"
        in PREFERENCE_METHOD_RUNNERS
    )
    assert (
        "llm_regressor_openp5_style_ollama_qwen3_30b_a3b_smoke5"
        in REGRESSION_METHOD_RUNNERS
    )


def test_openp5_style_server_methods_are_registered() -> None:
    interaction_names = {
        "llm_yes_no_openp5_style_litellm_qwen3_8b_with_item_stats_smoke",
        "llm_yes_no_openp5_style_litellm_qwen3_8b_with_item_stats_full",
        "llm_yes_no_openp5_style_litellm_qwen36_27b_with_item_stats_smoke",
        "llm_yes_no_openp5_style_litellm_qwen36_27b_with_item_stats_full",
        (
            "llm_listwise_ranking_openp5_style_litellm_qwen3_8b_"
            "with_item_stats_smoke"
        ),
        (
            "llm_listwise_ranking_openp5_style_litellm_qwen3_8b_"
            "with_item_stats_full"
        ),
        (
            "llm_listwise_ranking_openp5_style_litellm_qwen36_27b_"
            "with_item_stats_smoke"
        ),
        (
            "llm_listwise_ranking_openp5_style_litellm_qwen36_27b_"
            "with_item_stats_full"
        ),
    }
    preference_names = {
        "llm_preference_yes_no_openp5_style_litellm_qwen3_8b_smoke",
        "llm_preference_yes_no_openp5_style_litellm_qwen3_8b_full",
        "llm_preference_yes_no_openp5_style_litellm_qwen36_27b_smoke",
        "llm_preference_yes_no_openp5_style_litellm_qwen36_27b_full",
        (
            "llm_preference_listwise_ranking_openp5_style_litellm_"
            "qwen3_8b_smoke"
        ),
        (
            "llm_preference_listwise_ranking_openp5_style_litellm_"
            "qwen3_8b_full"
        ),
        (
            "llm_preference_listwise_ranking_openp5_style_litellm_"
            "qwen36_27b_smoke"
        ),
        (
            "llm_preference_listwise_ranking_openp5_style_litellm_"
            "qwen36_27b_full"
        ),
    }

    assert interaction_names <= INTERACTION_METHOD_RUNNERS.keys()
    assert preference_names <= PREFERENCE_METHOD_RUNNERS.keys()


def test_openp5_style_server_wrappers_match_history_visibility(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_method(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(interaction_openp5, "run_yes_no_method", fake_run_method)
    monkeypatch.setattr(interaction_openp5, "run_listwise_method", fake_run_method)
    monkeypatch.setattr(preference_openp5, "run_grouped_yes_no_method", fake_run_method)
    monkeypatch.setattr(preference_openp5, "run_listwise_method", fake_run_method)
    monkeypatch.setattr(
        interaction_openp5,
        "_serving_metadata",
        lambda: {"backend": "test"},
    )
    monkeypatch.setattr(
        interaction_openp5,
        "_source_metadata",
        lambda: {"source": "test"},
    )
    monkeypatch.setattr(
        preference_openp5,
        "_serving_metadata",
        lambda: {"backend": "test"},
    )
    monkeypatch.setattr(
        preference_openp5,
        "_source_metadata",
        lambda: {"source": "test"},
    )
    task = SimpleNamespace(manifest={"dataset": "ml-1m"})

    interaction_openp5.run_yes_no_litellm_qwen3_8b_with_item_stats_full(
        task,
        tmp_path,
    )
    interaction_openp5.run_listwise_litellm_qwen3_8b_with_item_stats_full(
        task,
        tmp_path,
    )
    preference_openp5.run_yes_no_litellm_qwen36_27b_full(task, tmp_path)
    preference_openp5.run_listwise_litellm_qwen36_27b_full(task, tmp_path)

    assert all(call["prompt_family"] == "openp5_style" for call in calls)
    assert [call.get("use_item_stats") for call in calls] == [
        True,
        True,
        None,
        False,
    ]
    assert [call["model"] for call in calls] == [
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3.6-27B",
        "Qwen/Qwen3.6-27B",
    ]
