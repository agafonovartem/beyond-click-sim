from __future__ import annotations

import math

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
    user_group_metrics = user_grouped_ranking_metrics(y_true, scores, groups, users, ks=(1,))

    assert group_metrics["ndcg@1"] == pytest.approx((1.0 + 0.0 + 1.0) / 3)
    assert user_group_metrics["ndcg@1"] == pytest.approx((((1.0 + 0.0) / 2) + 1.0) / 2)
    assert user_group_metrics["ndcg@1"] != pytest.approx(group_metrics["ndcg@1"])
    assert user_group_metrics["hit_rate@1"] == pytest.approx(user_group_metrics["ndcg@1"])
    assert user_group_metrics["n_users"] == 2
    assert user_group_metrics["n_groups"] == 3


def test_user_grouped_ranking_matches_group_macro_for_one_group_per_user() -> None:
    y_true = pd.Series([1, 0, 0, 1])
    scores = pd.Series([0.9, 0.1, 0.8, 0.2])
    groups = pd.Series(["u1-g1", "u1-g1", "u2-g1", "u2-g1"])
    users = pd.Series(["u1", "u1", "u2", "u2"])

    group_metrics = grouped_ranking_metrics(y_true, scores, groups, ks=(1, 2))
    user_group_metrics = user_grouped_ranking_metrics(y_true, scores, groups, users, ks=(1, 2))

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
        grouped_ranking_metrics(pd.Series([1]), pd.Series([0.1]), pd.Series(["g1", "g2"]))

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
