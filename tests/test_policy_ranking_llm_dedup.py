"""Tests for cross-policy LLM deduplication and run_count aggregation.

These tests cover the efficiency and noise-reduction mechanisms added to the
Q3 policy ranking LLM runner:

  1. majority_vote tie-break function
  2. _score_one_group with deliberate repetition (run_count > 1)
  3. _score_one_group with custom tie-break
  4. _score_one_group partial run failure
  5. Itemwise deduplication logic (unique (user, item) pairs)
  6. Batch deduplication logic (ordered and unordered modes)

The dedup logic lives inside _run(), which requires a full Task object.
Instead of calling _run(), we test the dedup step by replicating its
DataFrame transformations here, then pass the result to _score_groups()
directly to count LLM calls.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from beyond_click_sim.scorers import LLMInteractionYesNoScorer
from runners.in_distribution.policy_ranking_agreement.methods.llm_yes_no import (
    _score_groups,
    _score_one_group,
    majority_vote,
)


# ---------------------------------------------------------------------------
# Helpers (same FakeClient pattern as test_llm_scorers.py)
# ---------------------------------------------------------------------------

class FakeChatCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response))]
        )


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        completions = FakeChatCompletions(responses)
        self.chat = SimpleNamespace(completions=completions)
        self.completions = completions


def _fitted_itemwise_scorer(client: FakeClient) -> LLMInteractionYesNoScorer:
    """Return a fitted itemwise scorer with minimal train data."""
    X_train = pd.DataFrame({"user_id": ["u1"], "item_title": ["History"]})
    scorer = LLMInteractionYesNoScorer(
        client=client,
        model="fake",
        item_description_columns=("item_title",),
        prompt_style="itemwise",
    )
    scorer.fit(X_train, pd.Series([1]))
    return scorer


def _itemwise_group(user_id: str, item_title: str, idx=0) -> pd.DataFrame:
    """One-row group suitable for the itemwise scorer."""
    return pd.DataFrame(
        {
            "user_id": [user_id],
            "candidate_group": [f"g{idx}"],
            "item_title": [item_title],
        },
        index=[idx],
    )


# ---------------------------------------------------------------------------
# majority_vote
# ---------------------------------------------------------------------------

def test_majority_vote_yes_wins() -> None:
    assert majority_vote([1.0, 0.0, 1.0]) == 1.0


def test_majority_vote_no_wins() -> None:
    assert majority_vote([0.0, 1.0, 0.0]) == 0.0


def test_majority_vote_tie_goes_to_no() -> None:
    assert majority_vote([1.0, 0.0]) == 0.0


def test_majority_vote_single_yes() -> None:
    assert majority_vote([1.0]) == 1.0


def test_majority_vote_single_no() -> None:
    assert majority_vote([0.0]) == 0.0


# ---------------------------------------------------------------------------
# _score_one_group — run_count behaviour
# ---------------------------------------------------------------------------

def test_run_count_1_is_identity() -> None:
    """run_count=1 returns the single result unchanged; one LLM call."""
    client = FakeClient(["yes"])
    scorer = _fitted_itemwise_scorer(client)
    group = _itemwise_group("u1", "Portal")

    result, error = _score_one_group(
        scorer, "g0", group, max_attempts=1, run_count=1, tie_break_fn=majority_vote
    )

    assert error is None
    assert result.iloc[0] == 1.0
    assert len(client.completions.calls) == 1


def test_run_count_3_majority_yes() -> None:
    """3 runs [yes, no, yes] → majority_vote → 1.0."""
    client = FakeClient(["yes", "no", "yes"])
    scorer = _fitted_itemwise_scorer(client)
    group = _itemwise_group("u1", "Portal")

    result, error = _score_one_group(
        scorer, "g0", group, max_attempts=1, run_count=3, tie_break_fn=majority_vote
    )

    assert error is None
    assert result.iloc[0] == 1.0
    assert len(client.completions.calls) == 3


def test_run_count_3_majority_no() -> None:
    """3 runs [yes, no, no] → majority_vote → 0.0."""
    client = FakeClient(["yes", "no", "no"])
    scorer = _fitted_itemwise_scorer(client)
    group = _itemwise_group("u1", "Portal")

    result, error = _score_one_group(
        scorer, "g0", group, max_attempts=1, run_count=3, tie_break_fn=majority_vote
    )

    assert error is None
    assert result.iloc[0] == 0.0


def test_custom_tie_break_receives_all_run_scores() -> None:
    """Custom tie_break_fn receives one score per successful run."""
    received: list[list[float]] = []

    def capture_tie_break(scores: list[float]) -> float:
        received.append(scores)
        return max(scores)

    client = FakeClient(["yes", "no", "yes"])
    scorer = _fitted_itemwise_scorer(client)
    group = _itemwise_group("u1", "Portal")

    _score_one_group(
        scorer, "g0", group,
        max_attempts=1,
        run_count=3,
        tie_break_fn=capture_tie_break,
    )

    assert len(received) == 1
    assert sorted(received[0]) == [0.0, 1.0, 1.0]


def test_run_count_partial_failure_uses_successful_runs() -> None:
    """When some runs fail, the successful ones are aggregated (not NaN)."""
    class FailOnSecondCall:
        call_count = 0
        def __init__(self, responses):
            self.responses = responses
            self.calls: list = []
        def create(self, **kwargs):
            self.call_count += 1
            self.calls.append(kwargs)
            if self.call_count == 2:
                raise RuntimeError("simulated LLM error")
            response = self.responses.pop(0)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=response))]
            )

    failclient = FailOnSecondCall(["yes", "yes"])  # runs 1 and 3 succeed
    X_train = pd.DataFrame({"user_id": ["u1"], "item_title": ["History"]})
    scorer = LLMInteractionYesNoScorer(
        client=SimpleNamespace(chat=SimpleNamespace(completions=failclient)),
        model="fake",
        item_description_columns=("item_title",),
        prompt_style="itemwise",
    )
    scorer.fit(X_train, pd.Series([1]))
    group = _itemwise_group("u1", "Portal")

    result, error = _score_one_group(
        scorer, "g0", group,
        max_attempts=1,
        run_count=3,
        tie_break_fn=majority_vote,
    )

    # 2 of 3 runs succeeded with "yes" → majority_vote([1.0, 1.0]) = 1.0
    assert error is None
    assert result.iloc[0] == 1.0
    assert failclient.call_count == 3


def test_all_runs_fail_returns_nan_and_error() -> None:
    """When every run exhausts all attempts, the group returns NaN + error dict."""
    class AlwaysFails:
        calls = 0
        def create(self, **kwargs):
            self.calls += 1
            raise RuntimeError("always fails")

    badclient = AlwaysFails()
    X_train = pd.DataFrame({"user_id": ["u1"], "item_title": ["History"]})
    scorer = LLMInteractionYesNoScorer(
        client=SimpleNamespace(chat=SimpleNamespace(completions=badclient)),
        model="fake",
        item_description_columns=("item_title",),
        prompt_style="itemwise",
    )
    scorer.fit(X_train, pd.Series([1]))
    group = _itemwise_group("u1", "Portal")

    result, error = _score_one_group(
        scorer, "g0", group,
        max_attempts=2,
        run_count=3,
        tie_break_fn=majority_vote,
    )

    assert error is not None
    assert error["attempts"] == 6  # 3 runs × 2 attempts
    assert result.isna().all()


# ---------------------------------------------------------------------------
# Itemwise deduplication logic
# ---------------------------------------------------------------------------

def _apply_itemwise_dedup(X_test: pd.DataFrame) -> pd.DataFrame:
    """Replicate the itemwise dedup step from _run()."""
    X_test = X_test.copy()
    X_test["_score_key_"] = (
        X_test["user_id"].astype(str) + "::" + X_test["item_id"].astype(str)
    )
    X_to_score = X_test.drop_duplicates(subset=["_score_key_"]).copy()
    X_to_score["_llm_group_"] = X_to_score["_score_key_"]
    return X_test, X_to_score


def test_itemwise_dedup_reduces_cross_policy_duplicates() -> None:
    """Two policies both recommending item A → X_to_score has one row for (u1, A)."""
    X_test = pd.DataFrame({
        "user_id":   ["u1", "u1", "u1", "u1"],
        "item_id":   ["A",  "B",  "A",  "B"],
        "item_title":["Alpha","Beta","Alpha","Beta"],
        "policy":    ["PX", "PX", "PY", "PY"],
        "_llm_group_": ["u1::PX::A","u1::PX::B","u1::PY::A","u1::PY::B"],
    })
    _, X_to_score = _apply_itemwise_dedup(X_test)

    # 4 rows → 2 unique (user, item) pairs
    assert len(X_to_score) == 2
    assert set(X_to_score["_llm_group_"]) == {"u1::A", "u1::B"}


def test_itemwise_dedup_calls_each_pair_once() -> None:
    """With dedup, the scorer is called once per unique (user, item), not per policy."""
    X_test = pd.DataFrame({
        "user_id":    ["u1",       "u1",       "u1",       "u1"],
        "item_id":    ["A",        "B",        "A",        "B"],
        "item_title": ["Alpha",    "Beta",     "Alpha",    "Beta"],
        "policy":     ["PX",       "PX",       "PY",       "PY"],
        "_llm_group_": ["u1::PX::A","u1::PX::B","u1::PY::A","u1::PY::B"],
    })
    X_test_with_key, X_to_score = _apply_itemwise_dedup(X_test)

    client = FakeClient(["yes", "no"])  # exactly 2 responses
    X_train = pd.DataFrame({"user_id": ["u1"], "item_title": ["History"]})
    scorer = LLMInteractionYesNoScorer(
        client=client,
        model="fake",
        history_description_columns=("item_title",),
        candidate_description_columns=("item_title",),
        candidate_group_column="_llm_group_",
        prompt_style="itemwise",
    )
    scorer.fit(X_train, pd.Series([1]))

    unique_scores, errors = _score_groups(
        scorer, X_to_score,
        candidate_group_column="_llm_group_",
        max_attempts=1,
        max_workers=1,
    )

    # Only 2 LLM calls (not 4)
    assert len(client.completions.calls) == 2
    assert len(errors) == 0


def test_itemwise_dedup_score_broadcast_to_all_policies() -> None:
    """Scores from unique pairs propagate back to all policy rows identically."""
    X_test = pd.DataFrame({
        "user_id":    ["u1",        "u1",        "u1",        "u1"],
        "item_id":    ["A",         "B",         "A",         "B"],
        "item_title": ["Alpha",     "Beta",      "Alpha",     "Beta"],
        "policy":     ["PX",        "PX",        "PY",        "PY"],
        "_llm_group_": ["u1::PX::A","u1::PX::B","u1::PY::A","u1::PY::B"],
    }, index=[10, 11, 12, 13])
    X_test_with_key, X_to_score = _apply_itemwise_dedup(X_test)

    client = FakeClient(["yes", "no"])
    X_train = pd.DataFrame({"user_id": ["u1"], "item_title": ["History"]})
    scorer = LLMInteractionYesNoScorer(
        client=client,
        model="fake",
        history_description_columns=("item_title",),
        candidate_description_columns=("item_title",),
        candidate_group_column="_llm_group_",
        prompt_style="itemwise",
    )
    scorer.fit(X_train, pd.Series([1]))

    unique_scores, _ = _score_groups(
        scorer, X_to_score,
        candidate_group_column="_llm_group_",
        max_attempts=1,
        max_workers=1,
    )

    # Join back
    score_key_to_score = dict(zip(X_to_score["_score_key_"].tolist(), unique_scores.tolist()))
    scores = X_test_with_key["_score_key_"].map(score_key_to_score)

    # item A → yes(1.0) for both PX and PY
    # item B → no(0.0) for both PX and PY
    assert scores[10] == scores[12]  # both item A rows same score
    assert scores[11] == scores[13]  # both item B rows same score
    assert scores[10] != scores[11]  # A and B got different answers


# ---------------------------------------------------------------------------
# Batch deduplication logic
# ---------------------------------------------------------------------------

def _compute_batch_sigs(X_test: pd.DataFrame, batch_dedup_mode: str) -> dict[str, str]:
    """Replicate the batch signature computation from _run()."""
    group_sigs: dict[str, str] = {}
    for gid, grp in X_test.groupby("_llm_group_", sort=False):
        uid = str(grp["user_id"].iloc[0])
        if batch_dedup_mode == "ordered":
            items = tuple(grp.sort_values("rank")["item_id"].tolist())
        else:
            items = tuple(sorted(grp["item_id"].astype(str).tolist()))
        group_sigs[str(gid)] = f"{uid}::{items}"
    return group_sigs


def test_batch_dedup_identical_ordered_batches_have_same_sig() -> None:
    """Two policies with the same items in the same rank order share one batch sig."""
    X_test = pd.DataFrame({
        "user_id":   ["u1","u1","u1","u1"],
        "item_id":   ["A",  "B", "A", "B"],
        "rank":      [1,    2,   1,   2],
        "policy":    ["PX","PX","PY","PY"],
        "_llm_group_": ["u1::PX","u1::PX","u1::PY","u1::PY"],
    })
    group_sigs = _compute_batch_sigs(X_test, "ordered")

    assert len(set(group_sigs.values())) == 1  # same signature → deduplicated to 1 call


def test_batch_dedup_ordered_treats_permutations_as_different() -> None:
    """Different rank orders → different signatures in ordered mode."""
    X_test = pd.DataFrame({
        "user_id":   ["u1","u1","u1","u1"],
        "item_id":   ["A",  "B", "B", "A"],
        "rank":      [1,    2,   1,   2],
        "policy":    ["PX","PX","PY","PY"],
        "_llm_group_": ["u1::PX","u1::PX","u1::PY","u1::PY"],
    })
    group_sigs = _compute_batch_sigs(X_test, "ordered")

    assert len(set(group_sigs.values())) == 2  # PX=[A,B], PY=[B,A] — different


def test_batch_dedup_unordered_treats_permutations_as_same() -> None:
    """Different rank orders but same item set → same signature in unordered mode."""
    X_test = pd.DataFrame({
        "user_id":   ["u1","u1","u1","u1"],
        "item_id":   ["A",  "B", "B", "A"],
        "rank":      [1,    2,   1,   2],
        "policy":    ["PX","PX","PY","PY"],
        "_llm_group_": ["u1::PX","u1::PX","u1::PY","u1::PY"],
    })
    group_sigs = _compute_batch_sigs(X_test, "unordered")

    assert len(set(group_sigs.values())) == 1  # frozenset({A,B}) is the same


def test_batch_dedup_different_item_sets_always_different() -> None:
    """Different item sets → different signatures in both modes."""
    X_test = pd.DataFrame({
        "user_id":   ["u1","u1","u1","u1"],
        "item_id":   ["A",  "B", "A", "C"],  # PX=[A,B], PY=[A,C]
        "rank":      [1,    2,   1,   2],
        "policy":    ["PX","PX","PY","PY"],
        "_llm_group_": ["u1::PX","u1::PX","u1::PY","u1::PY"],
    })
    for mode in ("ordered", "unordered"):
        group_sigs = _compute_batch_sigs(X_test, mode)
        assert len(set(group_sigs.values())) == 2, f"failed in mode={mode!r}"
