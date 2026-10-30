from fractions import Fraction

import pandas as pd
import pytest

from beyond_click_sim.scorers.dyta4rec.config import DyTA4RecConfig
from beyond_click_sim.scorers.dyta4rec.fusion import weights, fuse
from beyond_click_sim.scorers.dyta4rec.profiles import prepare_evidence
from test_recagent_scorer import training


@pytest.mark.parametrize(
    "cluster,sequence,expected",
    [
        ("yes", "yes", (1, 1, 1)),
        ("yes", "no", (0.5, 1, 1)),
        ("no", "yes", (1, 1, 0)),
        ("no", "no", (1, 1, 0)),
        ("other", "other", (1, 1, 0)),
    ],
)
def test_asymmetric_weights(cluster, sequence, expected):
    assert weights(cluster, sequence) == expected


@pytest.mark.parametrize("strategy", ["rrf", "bc"])
@pytest.mark.parametrize("n", [1, 3, 10, 20])
def test_hand_computed_mapping_and_exact_ties(strategy, n):
    labels = [f"C{i}" for i in range(1, n + 1)]
    branches = {"sequence": labels, "profile": labels[::-1], "cluster": labels}
    result = fuse(
        branches,
        labels=labels,
        cluster_flag="no",
        sequence_flag="no",
        strategy=strategy,
    )
    if strategy == "rrf":
        expected = [
            float(Fraction(1, 60 + i) + Fraction(1, 60 + n + 1 - i))
            for i in range(1, n + 1)
        ]
    else:
        expected = [n + 3.0] * n
    assert result["scores"] == expected
    assert result["scores"][0] == result["scores"][-1]
    assert result["ranks"]["profile"]["C1"] == n
    if n > 1:
        assert result["tie_groups"]


def test_half_weight_and_unusual_bc_offset():
    branches = {b: ["C1", "C2", "C3"] for b in ("sequence", "profile", "cluster")}
    result = fuse(
        branches,
        labels=branches["sequence"],
        cluster_flag="yes",
        sequence_flag="other",
        strategy="bc",
    )
    assert result["scores"] == [10, 7.5, 5]
    assert result["active_weight"] == 2.5
    with pytest.raises(ValueError):
        fuse(
            branches,
            labels=branches["sequence"],
            cluster_flag="high",
            sequence_flag="no",
        )


def test_statistics_population_not_requested_cohort_and_empty_rating_categories():
    X, y, events = training(3)
    X["item_id"] = ["a", "a", "b"]
    events["item_id"] = X.item_id
    X["item_title"] = ["A", "A", "B"]
    X["item_genres"] = ['["Comedy", "Drama"]', '["Comedy", "Drama"]', '["Drama"]']
    X["rating"] = [1, 5, 3]
    users, population = prepare_evidence(
        X, y, events, DyTA4RecConfig(), sampled_column="is_sampled"
    )
    stats = users["u"]["statistics"]
    assert population == {
        "a": {"training_interaction_count": 2, "training_mean_rating": 3},
        "b": {"training_interaction_count": 1, "training_mean_rating": 3},
    }
    assert stats["observed_events"] == 3 and stats["distinct_items"] == 2
    assert stats["genre_statistics"]["Comedy"] == {
        "observed_events": 2,
        "mean_own_rating": 3,
    }
    assert stats["genre_statistics"]["Drama"] == {
        "observed_events": 3,
        "mean_own_rating": 3,
    }
    assert stats["population_count_over_own_events"] == {"mean": 5 / 3, "median": 2}
    assert (
        stats["population_associations_by_own_rating"]["2"]["mean_training_count"]
        is None
    )
    assert stats["rating_histogram"] == {"1": 1, "2": 0, "3": 1, "4": 0, "5": 1}


def test_stable_event_identity_ties_repeated_items_and_unavailable_demographics():
    X, y, events = training(3)
    X["user_age"] = pd.NA
    events["timestamp"] = 0
    users, _ = prepare_evidence(
        X.iloc[::-1],
        y.iloc[::-1],
        events.iloc[::-1],
        DyTA4RecConfig(),
        sampled_column="is_sampled",
    )
    assert [e["event_id"] for e in users["u"]["events"]] == [
        "event0",
        "event1",
        "event2",
    ]
    assert users["u"]["demographics"]["age_category"] == "unavailable"


@pytest.mark.parametrize(
    "case",
    ["sampled", "negative", "rating", "missing_demographics", "timestamp", "mapping"],
)
def test_invalid_training_rejected(case):
    X, y, events = training(3)
    if case == "sampled":
        X["is_sampled"] = True
    if case == "negative":
        y.iloc[0] = 0
    if case == "rating":
        X.loc[X.index[0], "rating"] = 6
    if case == "missing_demographics":
        X = X.drop(columns="user_gender")
    if case == "timestamp":
        events.loc[events.index[0], "timestamp"] = float("nan")
    if case == "mapping":
        events.loc[events.index[0], "item_id"] = "wrong"
    with pytest.raises(ValueError):
        prepare_evidence(X, y, events, DyTA4RecConfig(), sampled_column="is_sampled")
