from __future__ import annotations

import math

import pandas as pd
import pytest

from beyond_click_sim.evaluation.policy_ranking import evaluate_policy_recommendations


def test_evaluate_policy_recommendations_computes_expected_metrics() -> None:
    recs = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u2"],
            "item_id": ["i1", "i2", "i3", "i4", "i5", "i6"],
            "rank": [1, 2, 3, 1, 2, 3],
            "policy": ["PopularityPolicy"] * 6,
        }
    )
    targets = pd.Series([1, 0, 0, 0, 1, 0], dtype=float)

    result = evaluate_policy_recommendations(
        recs,
        targets=targets,
        policy_name="PopularityPolicy",
        k=3,
        ks=(1, 3),
        fit_recommend_seconds=1.23,
    )

    assert result["policy"] == "PopularityPolicy"
    assert result["k"] == 3
    assert result["n_users"] == 2
    assert result["n_recommendations"] == 6
    assert result["mean_hit_rate"] == pytest.approx(2.0 / 6.0)
    assert result["headline_metric"] == "macro_by_user_group_mean.ndcg@3"
    assert result["fit_recommend_seconds"] == pytest.approx(1.23)

    ranking = result["ranking"]
    macro_by_user = ranking["macro_by_user_group_mean"]
    assert macro_by_user["hit_rate@1"] == pytest.approx(0.5)
    expected_ndcg_at_3 = (1.0 + (1.0 / math.log2(3))) / 2.0
    assert macro_by_user["ndcg@3"] == pytest.approx(expected_ndcg_at_3)
