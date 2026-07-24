from __future__ import annotations

import pandas as pd
import pytest

from beyond_click_sim.scorers import HistoryTextAffinityScorer


def _history() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "description": [
                "荒野大镖客 游戏技巧",
                "婚礼 姐姐 演讲",
                "婚礼 家庭 祝福",
                "电子游戏 攻略",
            ],
            "category": ["游戏", "情感", "情感", "游戏"],
        }
    )
    y = pd.Series([1, 0, 1, 0], name="target")
    return X, y


def _scorer() -> HistoryTextAffinityScorer:
    return HistoryTextAffinityScorer(
        text_columns=("description", "category"),
        ngram_range=(1, 3),
    )


def test_text_affinity_uses_signed_per_user_feedback() -> None:
    X, y = _history()
    candidates = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "description": [
                "电子游戏 技巧",
                "婚礼 祝福",
                "电子游戏 技巧",
                "婚礼 祝福",
            ],
            "category": ["游戏", "情感", "游戏", "情感"],
        }
    )

    scores = _scorer().fit(X, y).score(candidates)

    assert scores.iloc[0] > scores.iloc[1]
    assert scores.iloc[3] > scores.iloc[2]


def test_text_affinity_preserves_candidate_index() -> None:
    X, y = _history()
    candidates = X.iloc[[0, 2]].copy()
    candidates.index = [13, 9]

    scores = _scorer().fit(X, y).score(candidates)

    assert scores.index.tolist() == [13, 9]
    assert scores.name == "score"


def test_text_affinity_skips_constant_label_profiles_but_keeps_source_text() -> None:
    X, y = _history()
    y.loc[X["user_id"] == "u1"] = 1
    X.loc[X["user_id"] == "u1", "description"] = [
        "sourceonlytoken",
        "sourceonlytoken",
    ]

    scorer = HistoryTextAffinityScorer(
        text_columns=("description", "category"),
        analyzer="word",
        ngram_range=(1, 1),
    ).fit(X, y)

    assert set(scorer.profile_by_user_) == {"u2"}
    assert "sourceonlytoken" in scorer.vectorizer_.vocabulary_
    assert scorer.fit_diagnostics_ == {
        "n_source_rows": 4,
        "n_source_users": 2,
        "n_profile_users": 1,
        "n_skipped_constant_label_users": 1,
        "profile_user_fraction": 0.5,
        "vocabulary_size": 8,
    }


def test_text_affinity_rejects_unknown_users() -> None:
    X, y = _history()
    candidates = pd.DataFrame(
        {
            "user_id": ["cold"],
            "description": ["游戏"],
            "category": ["游戏"],
        }
    )

    with pytest.raises(ValueError, match="without profiles"):
        _scorer().fit(X, y).score(candidates)


def test_text_affinity_can_zero_score_users_without_profiles() -> None:
    X, y = _history()
    y.loc[X["user_id"] == "u1"] = 1
    candidates = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "description": ["游戏", "游戏"],
            "category": ["游戏", "游戏"],
        }
    )
    scorer = HistoryTextAffinityScorer(
        text_columns=("description", "category"),
        ngram_range=(1, 3),
        cold_score_policy="zero",
    ).fit(X, y)

    scores = scorer.score(candidates)

    assert scores.iloc[0] == 0.0
    assert scorer.coverage_diagnostics_ == {
        "n_target_rows": 2,
        "n_scored_profile_rows": 1,
        "n_zero_fallback_rows": 1,
        "n_target_users": 2,
        "n_users_without_profiles": 1,
        "profile_row_fraction": 0.5,
    }
