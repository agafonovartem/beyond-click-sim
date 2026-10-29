from __future__ import annotations

import math
from itertools import permutations

import pandas as pd
import pytest

from beyond_click_sim.evaluation import (
    grouped_ranking_metrics,
    user_grouped_ranking_metrics,
)


def test_grouped_ranking_metrics_compute_hit_rate_and_ndcg_without_ties() -> None:
    y_true = pd.Series([1, 0, 1, 0, 1])
    scores = pd.Series([0.9, 0.8, 0.1, 0.4, 0.3])
    groups = pd.Series(["g1", "g1", "g1", "g2", "g2"])

    metrics = grouped_ranking_metrics(y_true, scores, groups, ks=(1, 2))

    discount_2 = 1 / math.log2(3)
    g1_ndcg_at_2 = 1 / (1 + discount_2)
    g2_ndcg_at_2 = discount_2
    assert metrics["hit_rate@1"] == pytest.approx(0.5)
    assert metrics["ndcg@1"] == pytest.approx(0.5)
    assert metrics["hit_rate@2"] == pytest.approx(1.0)
    assert metrics["ndcg@2"] == pytest.approx((g1_ndcg_at_2 + g2_ndcg_at_2) / 2)
    assert metrics["n_groups"] == 2
    assert metrics["n"] == 5
    assert metrics["n_positive"] == 3
    assert metrics["groups_with_score_ties"] == 0
    assert metrics["tie_policy"] == "average"


def test_grouped_ranking_metrics_average_ties() -> None:
    y_true = pd.Series([1, 0])
    scores = pd.Series([1.0, 1.0])
    groups = pd.Series(["g1", "g1"])

    metrics = grouped_ranking_metrics(y_true, scores, groups, ks=(1, 2))

    expected_ndcg_at_2 = 0.5 * 1.0 + 0.5 * (1 / math.log2(3))
    assert metrics["hit_rate@1"] == pytest.approx(0.5)
    assert metrics["ndcg@1"] == pytest.approx(0.5)
    assert metrics["hit_rate@2"] == pytest.approx(1.0)
    assert metrics["ndcg@2"] == pytest.approx(expected_ndcg_at_2)
    assert metrics["groups_with_score_ties"] == 1
    assert metrics["groups_with_score_ties_fraction"] == pytest.approx(1.0)


def test_user_grouped_ranking_metrics_average_groups_then_users() -> None:
    y_true = pd.Series([1, 0, 0, 1, 1, 0])
    scores = pd.Series([0.9, 0.1, 0.8, 0.2, 0.7, 0.1])
    groups = pd.Series(["u1-g1", "u1-g1", "u1-g2", "u1-g2", "u2-g1", "u2-g1"])
    users = pd.Series(["u1", "u1", "u1", "u1", "u2", "u2"])

    group_metrics = grouped_ranking_metrics(y_true, scores, groups, ks=(1,))
    user_group_metrics = user_grouped_ranking_metrics(
        y_true, scores, groups, users, ks=(1,)
    )

    assert group_metrics["ndcg@1"] == pytest.approx((1.0 + 0.0 + 1.0) / 3)
    assert user_group_metrics["ndcg@1"] == pytest.approx((((1.0 + 0.0) / 2) + 1.0) / 2)
    assert user_group_metrics["ndcg@1"] != pytest.approx(group_metrics["ndcg@1"])
    assert user_group_metrics["hit_rate@1"] == pytest.approx(
        user_group_metrics["ndcg@1"]
    )
    assert user_group_metrics["n_users"] == 2
    assert user_group_metrics["n_groups"] == 3


def test_user_grouped_ranking_matches_group_macro_for_one_group_per_user() -> None:
    y_true = pd.Series([1, 0, 0, 1])
    scores = pd.Series([0.9, 0.1, 0.8, 0.2])
    groups = pd.Series(["u1-g1", "u1-g1", "u2-g1", "u2-g1"])
    users = pd.Series(["u1", "u1", "u2", "u2"])

    group_metrics = grouped_ranking_metrics(y_true, scores, groups, ks=(1, 2))
    user_group_metrics = user_grouped_ranking_metrics(
        y_true, scores, groups, users, ks=(1, 2)
    )

    for metric in ["hit_rate@1", "ndcg@1", "hit_rate@2", "ndcg@2"]:
        assert user_group_metrics[metric] == pytest.approx(group_metrics[metric])


def test_grouped_ranking_metrics_handle_k_larger_than_group_size() -> None:
    y_true = pd.Series([1, 0])
    scores = pd.Series([0.9, 0.1])
    groups = pd.Series(["g1", "g1"])

    metrics = grouped_ranking_metrics(y_true, scores, groups, ks=(1, 3))

    assert metrics["hit_rate@3"] == pytest.approx(1.0)
    assert metrics["ndcg@3"] == pytest.approx(1.0)
    assert metrics["groups_with_size_lte@1"] == 0
    assert metrics["groups_with_size_lte@3"] == 1


def test_ranking_metrics_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="empty inputs"):
        grouped_ranking_metrics(
            pd.Series([], dtype=int),
            pd.Series([], dtype=float),
            pd.Series([], dtype=str),
        )

    with pytest.raises(ValueError, match="same length"):
        grouped_ranking_metrics(
            pd.Series([1]), pd.Series([0.1]), pd.Series(["g1", "g2"])
        )

    with pytest.raises(ValueError, match="NaN values"):
        grouped_ranking_metrics(
            pd.Series([1, 0]),
            pd.Series([0.1, float("nan")]),
            pd.Series(["g1", "g1"]),
        )

    with pytest.raises(ValueError, match="Unsupported tie_policy"):
        grouped_ranking_metrics(
            pd.Series([1, 0]),
            pd.Series([0.1, 0.2]),
            pd.Series(["g1", "g1"]),
            tie_policy="stable",  # type: ignore[arg-type]
        )


def test_recall_at_five_counts_multiple_positives_and_caps_at_group_size() -> None:
    metrics = grouped_ranking_metrics(
        pd.Series([1, 0, 0, 0, 0, 0, 1, 0]),
        pd.Series(list(reversed(range(8)))),
        pd.Series(["g"] * 8),
        ks=(5, 10),
    )
    assert metrics["hit_rate@5"] == 1.0
    assert metrics["recall@5"] == 0.5
    assert metrics["recall@10"] == 1.0


@pytest.mark.parametrize("k", [1, 2, 3, 5, 10])
def test_recall_matches_enumerated_orders_within_ties(k) -> None:
    # A fixed positive, four tied items straddling the cutoff, then a positive.
    tied = [1, 0, 1, 0]
    expected = sum(
        sum(([1] + list(order) + [1])[:k]) / 4 for order in permutations(tied)
    ) / math.factorial(len(tied))
    metrics = grouped_ranking_metrics(
        pd.Series([1, *tied, 1]),
        pd.Series([3, 2, 2, 2, 2, 1]),
        pd.Series(["g"] * 6),
        ks=(k,),
    )
    assert metrics[f"recall@{k}"] == pytest.approx(expected)


def test_recall_with_no_relevant_items_is_zero() -> None:
    metrics = grouped_ranking_metrics(
        pd.Series([0, 0]), pd.Series([1, 1]), pd.Series(["g", "g"]), ks=(5,)
    )
    assert metrics["recall@5"] == 0.0


def test_recall_averages_groups_within_users_and_retains_failed_groups() -> None:
    from runners.in_distribution.interaction_prediction.methods.common import (
        ranking_metrics_with_failed_groups_as_zero,
    )

    X = pd.DataFrame(
        {
            "user_id": ["u"] * 12 + ["v"] * 6,
            "candidate_group": ["a"] * 6 + ["b"] * 6 + ["c"] * 6,
        }
    )
    y = pd.Series([1, 0, 0, 0, 0, 1] * 3)
    scores = pd.Series([5, 4, 3, 2, 1, 0] + [float("nan")] * 6 + [5, 3, 2, 1, 0, 4])
    metrics = ranking_metrics_with_failed_groups_as_zero(
        X=X, y=y, scores=scores, candidate_group_column="candidate_group", ks=(5,)
    )
    assert metrics["macro_by_group"]["recall@5"] == pytest.approx((0.5 + 0 + 1) / 3)
    assert metrics["macro_by_user_group_mean"]["recall@5"] == pytest.approx(
        ((0.5 + 0) / 2 + 1) / 2
    )
    valid = scores.notna()
    parsed = user_grouped_ranking_metrics(
        y[valid],
        scores[valid],
        X.loc[valid, "candidate_group"],
        X.loc[valid, "user_id"],
        ks=(5,),
    )
    assert parsed["recall@5"] == pytest.approx((0.5 + 1) / 2)
    all_failed = ranking_metrics_with_failed_groups_as_zero(
        X=X,
        y=y,
        scores=scores * float("nan"),
        candidate_group_column="candidate_group",
        ks=(5,),
    )
    assert all_failed["macro_by_user_group_mean"]["recall@5"] == 0.0
