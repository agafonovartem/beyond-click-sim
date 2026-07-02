from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from beyond_click_sim.scorers import LLMInteractionYesNoScorer, LLMRegressor
from beyond_click_sim.scorers.constant import select_user_history_positions
from beyond_click_sim.scorers.llm import (
    parse_regression_value_response,
    parse_single_yes_no_response,
    parse_yes_no_response,
)


class FakeChatCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=response),
                )
            ]
        )


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        completions = FakeChatCompletions(responses)
        self.chat = SimpleNamespace(completions=completions)
        self.completions = completions


def test_parse_yes_no_response_accepts_case_and_whitespace() -> None:
    parsed = parse_yes_no_response(
        "  C1 : YES  \n\nC2: no\n",
        labels=["C1", "C2"],
    )

    assert parsed == {"C1": 1.0, "C2": 0.0}


def test_parse_yes_no_response_requires_all_labels() -> None:
    with pytest.raises(ValueError, match="Missing candidate labels"):
        parse_yes_no_response("C1: yes", labels=["C1", "C2"])


def test_parse_yes_no_response_rejects_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="Duplicate candidate label"):
        parse_yes_no_response("C1: yes\nC1: no", labels=["C1"])


def test_parse_yes_no_response_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="Unknown candidate label"):
        parse_yes_no_response("C1: yes\nC3: no", labels=["C1", "C2"])


def test_parse_yes_no_response_rejects_invalid_answers() -> None:
    with pytest.raises(ValueError, match="Invalid yes/no answer"):
        parse_yes_no_response("C1: maybe", labels=["C1"])


def test_llm_interaction_scorer_scores_candidate_groups() -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_title": ["Toy Story", "Aladdin", "Heat"],
            "item_genre": ["Animation", "Animation", "Crime"],
        }
    )
    y_train = pd.Series([0, 1, 0], name="target")
    X_test = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "candidate_group": ["g1", "g1", "g2"],
            "item_title": ["Lion King", "Godfather", "Portal 2"],
            "item_genre": ["Animation", "Crime", "Puzzle"],
        },
        index=["a", "b", "c"],
    )
    client = FakeClient(["C1: yes\nC2: no", "C1: no"])

    scorer = LLMInteractionYesNoScorer(
        client=client,
        model="fake-model",
        item_description_columns=("item_title", "item_genre"),
        max_history_items=2,
        temperature=0.1,
        max_tokens=64,
    )
    scorer.fit(X_train, y_train)
    scores = scorer.score(X_test)

    assert scores.tolist() == [1.0, 0.0, 0.0]
    assert list(scores.index) == ["a", "b", "c"]
    assert scores.name == "score"
    assert scorer.history_by_user_ == {
        "u1": [
            "H1. item_title: Aladdin; item_genre: Animation",
            "H2. item_title: Heat; item_genre: Crime",
        ]
    }
    assert len(client.completions.calls) == 2

    first_call = client.completions.calls[0]
    assert first_call["model"] == "fake-model"
    assert first_call["temperature"] == 0.1
    assert first_call["max_tokens"] == 64
    messages = first_call["messages"]
    user_prompt = messages[1]["content"]
    assert "Toy Story" not in user_prompt
    assert "H1. item_title: Aladdin" in user_prompt
    assert "H2. item_title: Heat" in user_prompt
    assert "C1. item_title: Lion King; item_genre: Animation" in user_prompt
    assert "C2. item_title: Godfather; item_genre: Crime" in user_prompt

    second_prompt = client.completions.calls[1]["messages"][1]["content"]
    assert "- No interaction history available." in second_prompt


def test_llm_regressor_passes_extra_body() -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_title": ["Toy Story"],
            "rating": [5],
        }
    )
    X_test = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_title": ["Lion King"],
        },
        index=["a"],
    )
    client = FakeClient(["4"])

    scorer = LLMRegressor(
        client=client,
        model="fake-model",
        history_description_columns=("item_title", "rating"),
        candidate_description_columns=("item_title",),
        target_description="Predict rating.",
        output_instructions="Return exactly one integer.",
        valid_values=(1, 2, 3, 4, 5),
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ).fit(X_train, pd.Series([5]))

    scores = scorer.score(X_test)

    assert scores.to_dict() == {"a": 4.0}
    assert client.completions.calls[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_llm_interaction_scorer_can_keep_full_history() -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_title": ["Toy Story", "Aladdin", "Heat"],
        }
    )

    scorer = LLMInteractionYesNoScorer(
        client=FakeClient([]),
        model="fake-model",
        item_description_columns=("item_title",),
        max_history_items=None,
    )
    scorer.fit(X_train, pd.Series([1, 0, 1]))

    assert scorer.history_by_user_ == {
        "u1": [
            "H1. item_title: Toy Story",
            "H2. item_title: Aladdin",
            "H3. item_title: Heat",
        ]
    }


def test_llm_interaction_scorer_prompt_lists_all_candidate_labels() -> None:
    X_train = pd.DataFrame({"user_id": ["u1"], "item_title": ["Toy Story"]})
    X_test = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "candidate_group": ["g1", "g1", "g1"],
            "item_title": ["Aladdin", "Heat", "Portal 2"],
        }
    )
    client = FakeClient(["C1: yes\nC2: no\nC3: no"])

    scorer = LLMInteractionYesNoScorer(
        client=client,
        model="fake-model",
        item_description_columns=("item_title",),
    )
    scorer.fit(X_train, pd.Series([1]))
    scorer.score(X_test)

    user_prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "C1. item_title: Aladdin" in user_prompt
    assert "C2. item_title: Heat" in user_prompt
    assert "C3. item_title: Portal 2" in user_prompt
    assert "C1:\nC2:\nC3:" in user_prompt


def test_llm_interaction_scorer_separates_history_and_candidate_columns() -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_title": ["Toy Story"],
            "history_rating": [5],
        }
    )
    X_test = pd.DataFrame(
        {
            "user_id": ["u1"],
            "candidate_group": ["g1"],
            "item_title": ["Lion King"],
            "candidate_target_rating": [999],
        }
    )
    client = FakeClient(["C1: yes"])

    scorer = LLMInteractionYesNoScorer(
        client=client,
        model="fake-model",
        history_description_columns=("item_title", "history_rating"),
        candidate_description_columns=("item_title",),
    )
    scorer.fit(X_train, pd.Series([1]))
    scorer.score(X_test)

    user_prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "H1. item_title: Toy Story; history_rating: 5" in user_prompt
    assert "C1. item_title: Lion King" in user_prompt
    assert "candidate_target_rating" not in user_prompt
    assert "999" not in user_prompt


def test_llm_interaction_scorer_formats_labels_and_numeric_values() -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_title": ["Toy Story"],
            "rating": [5.0],
            "item_rating_mean": [4.153],
            "item_rating_count": [2077],
            "missing_stat": [float("nan")],
        }
    )
    X_test = pd.DataFrame(
        {
            "user_id": ["u1"],
            "candidate_group": ["g1"],
            "item_title": ["Lion King"],
            "item_rating_mean": [3.333],
            "item_rating_count": [0],
            "missing_stat": [float("nan")],
        }
    )
    client = FakeClient(["C1: yes"])

    scorer = LLMInteractionYesNoScorer(
        client=client,
        model="fake-model",
        history_description_columns=(
            "item_title",
            "rating",
            "item_rating_mean",
            "item_rating_count",
            "missing_stat",
        ),
        candidate_description_columns=(
            "item_title",
            "item_rating_mean",
            "item_rating_count",
            "missing_stat",
        ),
        column_labels={
            "rating": "user rating",
            "item_rating_mean": "average rating",
            "item_rating_count": "number of prior reviews",
        },
    )
    scorer.fit(X_train, pd.Series([1]))
    scorer.score(X_test)

    user_prompt = client.completions.calls[0]["messages"][1]["content"]
    assert (
        "H1. item_title: Toy Story; user rating: 5; average rating: 4.15; "
        "number of prior reviews: 2077"
    ) in user_prompt
    assert (
        "C1. item_title: Lion King; average rating: 3.33; "
        "number of prior reviews: 0"
    ) in user_prompt
    assert "missing_stat" not in user_prompt


def test_llm_interaction_scorer_requires_fit_before_score() -> None:
    scorer = LLMInteractionYesNoScorer(
        client=FakeClient(["C1: yes"]),
        model="fake-model",
        item_description_columns=("item_title",),
    )

    with pytest.raises(RuntimeError, match="not fitted"):
        scorer.score(
            pd.DataFrame(
                {
                    "user_id": ["u1"],
                    "candidate_group": ["g1"],
                    "item_title": ["Toy Story"],
                }
            )
        )


def test_llm_interaction_scorer_requires_fit_columns() -> None:
    scorer = LLMInteractionYesNoScorer(
        client=FakeClient([]),
        model="fake-model",
        item_description_columns=("item_title",),
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        scorer.fit(pd.DataFrame({"user_id": ["u1"]}), pd.Series([1]))


def test_llm_interaction_scorer_requires_same_fit_lengths() -> None:
    scorer = LLMInteractionYesNoScorer(
        client=FakeClient([]),
        model="fake-model",
        item_description_columns=("item_title",),
    )

    with pytest.raises(ValueError, match="same length"):
        scorer.fit(
            pd.DataFrame({"user_id": ["u1"], "item_title": ["Toy Story"]}),
            pd.Series([1, 0]),
        )


def test_llm_interaction_scorer_requires_candidate_group_column() -> None:
    scorer = LLMInteractionYesNoScorer(
        client=FakeClient(["C1: yes"]),
        model="fake-model",
        item_description_columns=("item_title",),
    )
    scorer.fit(
        pd.DataFrame({"user_id": ["u1"], "item_title": ["Toy Story"]}),
        pd.Series([1]),
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        scorer.score(pd.DataFrame({"user_id": ["u1"], "item_title": ["Aladdin"]}))


def test_llm_interaction_scorer_rejects_multi_user_candidate_group() -> None:
    scorer = LLMInteractionYesNoScorer(
        client=FakeClient(["C1: yes\nC2: no"]),
        model="fake-model",
        item_description_columns=("item_title",),
    )
    scorer.fit(
        pd.DataFrame({"user_id": ["u1"], "item_title": ["Toy Story"]}),
        pd.Series([1]),
    )

    X_test = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "candidate_group": ["g1", "g1"],
            "item_title": ["Aladdin", "Portal 2"],
        }
    )

    with pytest.raises(ValueError, match="exactly one user"):
        scorer.score(X_test)


def test_parse_regression_value_response_accepts_valid_integer() -> None:
    assert parse_regression_value_response(" 4 \n", valid_values=(1, 2, 3, 4, 5)) == 4.0


@pytest.mark.parametrize("text", ["4.0", "3.5", "rating: 4", "4/5", ""])
def test_parse_regression_value_response_rejects_non_bare_integer(text: str) -> None:
    with pytest.raises(ValueError, match="bare integer"):
        parse_regression_value_response(text, valid_values=(1, 2, 3, 4, 5))


@pytest.mark.parametrize("text", ["0", "6"])
def test_parse_regression_value_response_rejects_out_of_range_integer(text: str) -> None:
    with pytest.raises(ValueError, match="outside valid values"):
        parse_regression_value_response(text, valid_values=(1, 2, 3, 4, 5))


def test_llm_interaction_scorer_forwards_extra_body_to_create() -> None:
    X_train = pd.DataFrame({"user_id": ["u1"], "item_title": ["Toy Story"]})
    X_test = pd.DataFrame(
        {"user_id": ["u1"], "candidate_group": ["g1"], "item_title": ["Lion King"]}
    )
    client = FakeClient(["C1: yes"])

    scorer = LLMInteractionYesNoScorer(
        client=client,
        model="fake-model",
        item_description_columns=("item_title",),
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    scorer.fit(X_train, pd.Series([1]))
    scorer.score(X_test)

    assert client.completions.calls[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_llm_interaction_scorer_omits_extra_body_when_none() -> None:
    X_train = pd.DataFrame({"user_id": ["u1"], "item_title": ["Toy Story"]})
    X_test = pd.DataFrame(
        {"user_id": ["u1"], "candidate_group": ["g1"], "item_title": ["Lion King"]}
    )
    client = FakeClient(["C1: yes"])

    scorer = LLMInteractionYesNoScorer(
        client=client,
        model="fake-model",
        item_description_columns=("item_title",),
        extra_body=None,
    )
    scorer.fit(X_train, pd.Series([1]))
    scorer.score(X_test)

    assert "extra_body" not in client.completions.calls[0]
    X_train = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_title": ["Toy Story", "Aladdin", "Heat"],
            "item_genres": ["Animation", "Animation", "Crime"],
            "rating": [5, 4, 2],
        }
    )
    X_test = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "item_title": ["Lion King", "Portal 2"],
            "item_genres": ["Animation", "Puzzle"],
        },
        index=["a", "b"],
    )
    client = FakeClient(["4", "3"])

    scorer = LLMRegressor(
        client=client,
        model="fake-model",
        history_description_columns=("item_title", "item_genres", "rating"),
        candidate_description_columns=("item_title", "item_genres"),
        target_description="Predict the integer MovieLens rating.",
        output_instructions="Return exactly one integer: 1, 2, 3, 4, or 5. Return no other text.",
        valid_values=(1, 2, 3, 4, 5),
        max_history_items=2,
        temperature=0.1,
        max_tokens=32,
    ).fit(X_train, pd.Series([5, 4, 2]))

    scores = scorer.score(X_test)

    assert scores.tolist() == [4.0, 3.0]
    assert list(scores.index) == ["a", "b"]
    assert scores.name == "score"
    assert scorer.history_by_user_ == {
        "u1": [
            "H1. item_title: Aladdin; item_genres: Animation; rating: 4",
            "H2. item_title: Heat; item_genres: Crime; rating: 2",
        ]
    }
    assert len(client.completions.calls) == 2

    first_call = client.completions.calls[0]
    assert first_call["model"] == "fake-model"
    assert first_call["temperature"] == 0.1
    assert first_call["max_tokens"] == 32
    first_prompt = first_call["messages"][1]["content"]
    assert "Toy Story" not in first_prompt
    assert "H1. item_title: Aladdin" in first_prompt
    assert "Candidate. item_title: Lion King; item_genres: Animation" in first_prompt

    second_prompt = client.completions.calls[1]["messages"][1]["content"]
    assert "- No interaction history available." in second_prompt


def test_llm_regressor_uses_shared_history_window_selection() -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u1", "u1", "u2", "u1"],
            "item_title": ["A", "B", "C", "D", "E", "F"],
            "item_genres": ["g"] * 6,
            "rating": [5, 1, 2, 4, 3, 1],
        }
    )
    client = FakeClient([])

    scorer = LLMRegressor(
        client=client,
        model="fake-model",
        history_description_columns=("item_title", "item_genres", "rating"),
        candidate_description_columns=("item_title", "item_genres"),
        target_description="Predict the integer MovieLens rating.",
        output_instructions="Return exactly one integer: 1, 2, 3, 4, or 5. Return no other text.",
        valid_values=(1, 2, 3, 4, 5),
        max_history_items=3,
    ).fit(X_train, pd.Series([5, 1, 2, 4, 3, 1]))

    selected = select_user_history_positions(
        X_train,
        user_column="user_id",
        max_history_items=3,
    )
    assert selected["u1"] == [2, 3, 5]
    assert scorer.history_by_user_["u1"] == [
        "H1. item_title: C; item_genres: g; rating: 2",
        "H2. item_title: D; item_genres: g; rating: 4",
        "H3. item_title: F; item_genres: g; rating: 1",
    ]


def test_llm_regressor_prompt_does_not_leak_candidate_target_or_rating() -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_title": ["Toy Story"],
            "item_genres": ["Animation"],
            "rating": [5],
        }
    )
    X_test = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_title": ["Lion King"],
            "item_genres": ["Animation"],
            "rating": [999],
            "target": [999],
        }
    )
    client = FakeClient(["4"])

    scorer = LLMRegressor(
        client=client,
        model="fake-model",
        history_description_columns=("item_title", "item_genres", "rating"),
        candidate_description_columns=("item_title", "item_genres"),
        target_description="Predict the integer MovieLens rating.",
        output_instructions="Return exactly one integer: 1, 2, 3, 4, or 5. Return no other text.",
        valid_values=(1, 2, 3, 4, 5),
    ).fit(X_train, pd.Series([5]))
    scorer.score(X_test)

    user_prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "H1. item_title: Toy Story; item_genres: Animation; rating: 5" in user_prompt
    assert "Candidate. item_title: Lion King; item_genres: Animation" in user_prompt
    assert "target" not in user_prompt
    assert "rating: 999" not in user_prompt
    assert "999" not in user_prompt


def test_llm_regressor_formats_labels_and_numeric_values() -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_title": ["Toy Story"],
            "rating": [5.0],
            "item_rating_mean": [4.153],
            "item_rating_count": [2077],
        }
    )
    X_test = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_title": ["Lion King"],
            "item_rating_mean": [3.333],
            "item_rating_count": [0],
        }
    )
    client = FakeClient(["4"])

    scorer = LLMRegressor(
        client=client,
        model="fake-model",
        history_description_columns=(
            "item_title",
            "rating",
            "item_rating_mean",
            "item_rating_count",
        ),
        candidate_description_columns=(
            "item_title",
            "item_rating_mean",
            "item_rating_count",
        ),
        column_labels={
            "rating": "user rating",
            "item_rating_mean": "average rating",
            "item_rating_count": "number of prior reviews",
        },
        target_description="Predict a rating.",
        output_instructions="Return exactly one integer.",
        valid_values=(1, 2, 3, 4, 5),
    ).fit(X_train, pd.Series([5]))
    scorer.score(X_test)

    user_prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "user rating: 5" in user_prompt
    assert "average rating: 4.15" in user_prompt
    assert "Candidate. item_title: Lion King; average rating: 3.33" in user_prompt


def test_llm_regressor_requires_fit_before_score() -> None:
    scorer = LLMRegressor(
        client=FakeClient(["4"]),
        model="fake-model",
        item_description_columns=("item_title",),
        target_description="Predict a rating.",
        output_instructions="Return exactly one integer.",
        valid_values=(1, 2, 3, 4, 5),
    )

    with pytest.raises(RuntimeError, match="not fitted"):
        scorer.score(pd.DataFrame({"user_id": ["u1"], "item_title": ["Toy Story"]}))


def test_llm_regressor_rejects_invalid_config_and_inputs() -> None:
    with pytest.raises(ValueError, match="valid_values"):
        LLMRegressor(
            client=FakeClient([]),
            model="fake-model",
            item_description_columns=("item_title",),
            target_description="Predict a rating.",
            output_instructions="Return exactly one integer.",
            valid_values=(),
        )

    scorer = LLMRegressor(
        client=FakeClient([]),
        model="fake-model",
        item_description_columns=("item_title",),
        target_description="Predict a rating.",
        output_instructions="Return exactly one integer.",
        valid_values=(1, 2, 3, 4, 5),
    )
    with pytest.raises(ValueError, match="same length"):
        scorer.fit(
            pd.DataFrame({"user_id": ["u1"], "item_title": ["Toy Story"]}),
            pd.Series([1, 2]),
        )
    with pytest.raises(ValueError, match="Missing required columns"):
        scorer.fit(pd.DataFrame({"user_id": ["u1"]}), pd.Series([1]))


# ---------------------------------------------------------------------------
# parse_single_yes_no_response
# ---------------------------------------------------------------------------

def test_parse_single_yes_no_response_accepts_yes_and_no() -> None:
    assert parse_single_yes_no_response("yes") == 1.0
    assert parse_single_yes_no_response("no") == 0.0


def test_parse_single_yes_no_response_is_case_insensitive() -> None:
    assert parse_single_yes_no_response("YES") == 1.0
    assert parse_single_yes_no_response("No") == 0.0


def test_parse_single_yes_no_response_strips_whitespace() -> None:
    assert parse_single_yes_no_response("  yes  \n") == 1.0


def test_parse_single_yes_no_response_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="bare 'yes' or 'no'"):
        parse_single_yes_no_response("maybe")


# ---------------------------------------------------------------------------
# LLMInteractionYesNoScorer — itemwise prompt style
# ---------------------------------------------------------------------------

def test_llm_interaction_scorer_rejects_invalid_prompt_style() -> None:
    with pytest.raises(ValueError, match="prompt_style"):
        LLMInteractionYesNoScorer(
            client=FakeClient([]),
            model="fake-model",
            item_description_columns=("item_title",),
            prompt_style="unknown",
        )


def test_llm_interaction_scorer_itemwise_uses_new_prompt_templates() -> None:
    X_train = pd.DataFrame({"user_id": ["u1"], "item_title": ["Toy Story"]})
    X_test = pd.DataFrame(
        {
            "user_id": ["u1"],
            "candidate_group": ["g1"],
            "item_title": ["Lion King"],
        }
    )
    client = FakeClient(["yes"])

    scorer = LLMInteractionYesNoScorer(
        client=client,
        model="fake-model",
        item_description_columns=("item_title",),
        prompt_style="itemwise",
    )
    scorer.fit(X_train, pd.Series([1]))
    scores = scorer.score(X_test)

    assert scores.tolist() == [1.0]
    messages = client.completions.calls[0]["messages"]
    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]
    assert "exactly one word: yes or no" in system_prompt
    assert "Recommended item:" in user_prompt
    assert "Would the user interact" in user_prompt
    assert "C1" not in user_prompt


def test_llm_interaction_scorer_itemwise_formats_candidate_without_label_prefix() -> None:
    X_train = pd.DataFrame({"user_id": ["u1"], "item_title": ["Toy Story"], "item_genre": ["Animation"]})
    X_test = pd.DataFrame(
        {
            "user_id": ["u1"],
            "candidate_group": ["g1"],
            "item_title": ["Lion King"],
            "item_genre": ["Animation"],
        }
    )
    client = FakeClient(["yes"])

    scorer = LLMInteractionYesNoScorer(
        client=client,
        model="fake-model",
        item_description_columns=("item_title", "item_genre"),
        prompt_style="itemwise",
    )
    scorer.fit(X_train, pd.Series([1]))
    scorer.score(X_test)

    user_prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "C1." not in user_prompt
    assert "item_title: Lion King; item_genre: Animation" in user_prompt


def test_llm_interaction_scorer_itemwise_scores_each_item_separately() -> None:
    X_train = pd.DataFrame({"user_id": ["u1"], "item_title": ["Toy Story"]})
    X_test = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "candidate_group": ["g1", "g2"],
            "item_title": ["Lion King", "Godfather"],
        },
        index=["a", "b"],
    )
    client = FakeClient(["yes", "no"])

    scorer = LLMInteractionYesNoScorer(
        client=client,
        model="fake-model",
        item_description_columns=("item_title",),
        prompt_style="itemwise",
    )
    scorer.fit(X_train, pd.Series([1]))
    scores = scorer.score(X_test)

    assert scores.tolist() == [1.0, 0.0]
    assert list(scores.index) == ["a", "b"]
    assert len(client.completions.calls) == 2


def test_llm_interaction_scorer_itemwise_rejects_group_size_gt_1() -> None:
    X_train = pd.DataFrame({"user_id": ["u1"], "item_title": ["Toy Story"]})
    X_test = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "candidate_group": ["g1", "g1"],
            "item_title": ["Lion King", "Godfather"],
        }
    )
    client = FakeClient(["yes"])

    scorer = LLMInteractionYesNoScorer(
        client=client,
        model="fake-model",
        item_description_columns=("item_title",),
        prompt_style="itemwise",
    )
    scorer.fit(X_train, pd.Series([1]))

    with pytest.raises(ValueError, match="exactly one row"):
        scorer.score(X_test)
