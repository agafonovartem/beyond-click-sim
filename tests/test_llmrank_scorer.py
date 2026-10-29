from __future__ import annotations

import random
import re
from threading import Lock
from types import SimpleNamespace

import pandas as pd
import pytest

from beyond_click_sim.scorers import LLMInteractionListwiseRankingScorer, LLMRankScorer
from beyond_click_sim.scorers.history import LLMRankScorer as HistoryLLMRankScorer
from runners.in_distribution.interaction_prediction.methods.llm_yes_no import (
    _score_groups,
)


class FakeClient:
    """Return rankings by item identity, regardless of presentation order."""

    def __init__(self, replies=None, usage=None):
        self.chat = SimpleNamespace(completions=self)
        self.calls = []
        self.replies = list(replies) if replies is not None else None
        self.usage = usage
        self.lock = Lock()

    def create(self, **kwargs):
        with self.lock:
            self.calls.append(kwargs)
            reply = self.replies.pop(0) if self.replies is not None else None
        if isinstance(reply, Exception):
            raise reply
        candidates = dict(
            re.findall(
                r"^(C\d+)\. item_title: ([^;\n]+)",
                kwargs["messages"][1]["content"],
                re.M,
            )
        )
        if isinstance(reply, list):
            inverse = {title: label for label, title in candidates.items()}
            text = "\n".join(inverse[title] for title in reply)
        elif reply is None:
            text = "\n".join(sorted(candidates, key=candidates.get))
        else:
            text = reply
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=self.usage,
        )


def make_scorer(client=None, **kwargs):
    scorer = LLMRankScorer(
        client=client or FakeClient(),
        model="fake",
        history_description_columns=("item_title", "rating"),
        candidate_description_columns=("item_title",),
        **kwargs,
    )
    train = pd.DataFrame(
        {
            "user_id": ["u"] * 25,
            "item_id": [f"h{i}" for i in range(25)],
            "item_title": [f"History {i}" for i in range(25)],
            "rating": [5] * 25,
        }
    )
    return scorer.fit(train, pd.Series([1] * 25))


def candidates(n=3):
    return pd.DataFrame(
        {
            "user_id": ["u"] * n,
            "item_id": [f"i{i}" for i in range(n)],
            "candidate_group": ["g"] * n,
            "item_title": [chr(65 + i) for i in range(n)],
        },
        index=[10 + 7 * i for i in range(n)],
    )


def permutations(records):
    return [
        (record["request_key"], [r["permutation"] for r in record["rounds"]])
        for record in records
    ]


def test_public_contract_recency_history_and_original_item_score_mapping():
    client = FakeClient(replies=[["B", "A", "C"], ["A", "C", "B"], ["B", "C", "A"]])
    scorer = make_scorer(client)
    X = candidates()
    original = X.copy(deep=True)
    history = {user: list(rows) for user, rows in scorer.history_by_user_.items()}
    global_rng = random.getstate()

    result = scorer.score(X)

    assert HistoryLLMRankScorer is LLMRankScorer
    assert isinstance(scorer, LLMInteractionListwiseRankingScorer)
    assert result.tolist() == [3.0, 4.0, 2.0]
    assert result.name == "score"
    assert result.index.equals(X.index)
    pd.testing.assert_frame_equal(X, original)
    assert scorer.history_by_user_ == history
    assert random.getstate() == global_rng
    assert len(client.calls) == 3
    for call in client.calls:
        assert (call["model"], call["temperature"], call["max_tokens"]) == (
            "fake",
            0,
            512,
        )
        prompt = call["messages"][1]["content"]
        assert "H1. item_title: History 5; rating: 5" in prompt
        assert "H20. item_title: History 24; rating: 5" in prompt
        assert "most recent interaction in this history is H20." in prompt
        assert "item_title: History 4;" not in prompt
        assert "chronological order (oldest to newest)" in prompt
        assert "Do not output item names or history items." in prompt
    records = scorer.drain_diagnostics()
    assert len(records) == 1
    assert records[0]["completed"] is True
    assert records[0]["scores"] == [3, 4, 2]
    assert [r["ranked_item_ids"] for r in records[0]["rounds"]] == [
        ["i1", "i0", "i2"],
        ["i0", "i2", "i1"],
        ["i1", "i2", "i0"],
    ]
    assert scorer.drain_diagnostics() == []
    assert scorer.prompt_metadata["version"] == "llmrank_recency_bootstrap_v1"


def test_aggregation_preserves_ties():
    scorer = make_scorer(
        FakeClient(
            replies=[
                ["A", "B", "C"],
                ["B", "C", "A"],
                ["C", "A", "B"],
            ]
        )
    )
    assert scorer.score(candidates()).tolist() == [3, 3, 3]


@pytest.mark.parametrize("n", [1, 2, 4])
def test_variable_group_size_and_short_history(n):
    scorer = make_scorer(max_history_items=2)
    assert scorer.score(candidates(n)).tolist() == [3 * (n - 1 - i) for i in range(n)]
    assert (
        "most recent interaction in this history is H2."
        in scorer.client.calls[0]["messages"][1]["content"]
    )
    for record in scorer.drain_diagnostics():
        for round_record in record["rounds"]:
            assert sorted(round_record["permutation"]) == list(range(n))
            assert set(round_record["label_to_item_id"].values()) == set(
                candidates(n).item_id
            )


def test_empty_input_makes_no_requests():
    scorer = make_scorer()
    result = scorer.score(candidates(0))
    assert result.empty and result.name == "score"
    assert scorer.client.calls == []


def test_reproducibility_across_calls_seeds_and_thread_scheduling():
    X = pd.concat(
        [candidates().assign(candidate_group=f"g{i}") for i in range(4)],
        ignore_index=True,
    )
    serial = make_scorer(bootstrap_seed=42)
    parallel = make_scorer(bootstrap_seed=42)
    serial_scores, serial_errors = _score_groups(
        serial,
        X,
        candidate_group_column="candidate_group",
        max_attempts=1,
        max_workers=1,
    )
    parallel_scores, parallel_errors = _score_groups(
        parallel,
        X,
        candidate_group_column="candidate_group",
        max_attempts=1,
        max_workers=4,
    )
    pd.testing.assert_series_equal(serial_scores, parallel_scores)
    assert not serial_errors and not parallel_errors
    expected = permutations(serial.drain_diagnostics())
    assert expected == permutations(parallel.drain_diagnostics())
    serial.score(X)
    assert expected == permutations(serial.drain_diagnostics())
    different = make_scorer(bootstrap_seed=43)
    different.score(X)
    assert expected != permutations(different.drain_diagnostics())
    # Round draws may repeat; they must not all share a single global shuffle.
    assert any(len({tuple(p) for p in draws}) > 1 for _, draws in expected)


@pytest.mark.parametrize(
    "invalid",
    [
        "",
        "C1",
        "C1\nC1\nC2",
        "C1\nC2\nC99",
        RuntimeError("request failed"),
    ],
)
def test_request_and_parser_failures_retry_whole_ensemble(invalid):
    client = FakeClient(
        replies=[
            ["C", "B", "A"],
            ["C", "B", "A"],
            invalid,
            ["A", "B", "C"],
            ["A", "B", "C"],
            ["A", "B", "C"],
        ]
    )
    scorer = make_scorer(client)
    result, errors = _score_groups(
        scorer, candidates(), candidate_group_column="candidate_group", max_attempts=5
    )
    assert result.tolist() == [6, 3, 0]  # No partial scores from the discarded attempt.
    assert not errors and len(client.calls) == 6
    records = scorer.drain_diagnostics()
    assert [r["attempt"] for r in records] == [1, 2]
    assert [r["completed"] for r in records] == [False, True]
    assert permutations(records)[0][1] == permutations(records)[1][1]
    assert "error" in records[0]["rounds"][-1]


def test_exhausted_groups_are_missing_and_fifteen_requests_is_the_bound():
    client = FakeClient(replies=["C1\nC2\nC3", "C1\nC2\nC3", "C1"] * 5)
    scorer = make_scorer(client)
    result, errors = _score_groups(
        scorer, candidates(), candidate_group_column="candidate_group", max_attempts=5
    )
    assert result.isna().all()
    assert len(errors) == 1 and errors[0]["attempts"] == 5
    assert len(client.calls) == 15
    records = scorer.drain_diagnostics()
    assert len(records) == 5 and not any(r["completed"] for r in records)
    assert all("scores" not in record for record in records)


@pytest.mark.parametrize(
    "case",
    [
        "unknown_user",
        "empty_history",
        "mixed_users",
        "duplicate_items",
        "null_group",
        "duplicate_index",
    ],
)
def test_invalid_inputs_fail_without_requests(case):
    scorer = make_scorer(max_history_items=0 if case == "empty_history" else 20)
    X = candidates()
    if case == "unknown_user":
        X["user_id"] = "missing"
    elif case == "mixed_users":
        X.loc[X.index[0], "user_id"] = "other"
    elif case == "duplicate_items":
        X["item_id"] = "same"
    elif case == "null_group":
        X.loc[X.index[0], "candidate_group"] = None
    elif case == "duplicate_index":
        X.index = [0, 0, 1]
    with pytest.raises(ValueError):
        scorer.score(X)
    assert not scorer.client.calls


def test_usage_is_preserved_and_targets_do_not_affect_generation():
    client = FakeClient(
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3, total_tokens=15)
    )
    scorer = make_scorer(client)
    first = scorer.score(candidates().assign(target=[1, 0, 0]))
    second = scorer.score(candidates().assign(target=[0, 1, 1]))
    pd.testing.assert_series_equal(first, second)
    assert len(client.calls) == 6
    assert client.calls[:3] == client.calls[3:]
    assert all(
        r["usage"] == {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
        for record in scorer.drain_diagnostics()
        for r in record["rounds"]
    )


@pytest.mark.parametrize("seed", [True, "0", 1.5])
def test_invalid_seed(seed):
    with pytest.raises(ValueError, match="bootstrap_seed"):
        make_scorer(bootstrap_seed=seed)


def test_neutral_iid_prompt_is_not_a_temporal_variant():
    with pytest.raises(ValueError, match="dedicated"):
        make_scorer(prompt_family="openp5_style")
