from __future__ import annotations

import pandas as pd
import pytest

from beyond_click_sim.tasks import PostSplitUserSampler


def test_post_split_user_sampler_limits_eval_users_after_split() -> None:
    rows = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "item_id": ["i1", "i2", "i3", "i4", "i5", "i6"],
        }
    )
    train = pd.DataFrame({"user_id": ["u1", "u2"]})
    sampler = PostSplitUserSampler(n_users=1, seed=0)

    sampled, summary = sampler.sample(rows, train=train)

    assert sampled["user_id"].nunique() == 1
    assert set(sampled["user_id"]) <= {"u1", "u2"}
    assert len(sampled) == 2
    assert summary == {
        "eligible_users": 2,
        "selected_users": 1,
        "rows_before": 6,
        "rows_after": 2,
    }


def test_post_split_user_sampler_can_keep_users_without_train_history() -> None:
    rows = pd.DataFrame({"user_id": ["u1", "u2"], "item_id": ["i1", "i2"]})
    train = pd.DataFrame({"user_id": ["u1"]})
    sampler = PostSplitUserSampler(
        n_users=None,
        seed=0,
        require_train_history=False,
    )

    sampled, summary = sampler.sample(rows, train=train)

    assert sampled["user_id"].tolist() == ["u1", "u2"]
    assert summary["eligible_users"] == 2
    assert summary["selected_users"] == 2


def test_post_split_user_sampler_can_limit_rows_per_user_deterministically() -> None:
    rows = pd.DataFrame(
        [
            {"interaction_id": f"u1-r{i}", "user_id": "u1", "item_id": f"i{i}"}
            for i in range(6)
        ]
        + [
            {"interaction_id": f"u2-r{i}", "user_id": "u2", "item_id": f"j{i}"}
            for i in range(2)
        ]
        + [
            {"interaction_id": f"u3-r{i}", "user_id": "u3", "item_id": f"k{i}"}
            for i in range(6)
        ]
    )
    shuffled = rows.sample(frac=1, random_state=123).reset_index(drop=True)
    sampler = PostSplitUserSampler(
        n_users=None,
        seed=7,
        require_train_history=False,
        max_rows_per_user=2,
    )

    sampled, summary = sampler.sample(rows, train=pd.DataFrame())
    sampled_from_shuffled, _ = sampler.sample(shuffled, train=pd.DataFrame())

    assert sampled.groupby("user_id").size().to_dict() == {
        "u1": 2,
        "u2": 2,
        "u3": 2,
    }
    assert set(sampled["interaction_id"]) == set(
        sampled_from_shuffled["interaction_id"]
    )
    assert summary == {
        "eligible_users": 3,
        "selected_users": 3,
        "rows_before": 14,
        "rows_after": 6,
        "max_rows_per_user": 2,
        "rows_after_user_selection": 14,
        "users_with_rows_capped": 2,
    }


def test_post_split_user_sampler_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="n_users"):
        PostSplitUserSampler(n_users=0)
    with pytest.raises(ValueError, match="max_rows_per_user"):
        PostSplitUserSampler(n_users=1, max_rows_per_user=0)

    sampler = PostSplitUserSampler(n_users=1)
    with pytest.raises(ValueError, match="Missing user column"):
        sampler.sample(
            pd.DataFrame({"item_id": ["i1"]}),
            train=pd.DataFrame({"user_id": ["u1"]}),
        )

    with pytest.raises(ValueError, match="Missing user column in train"):
        sampler.sample(
            pd.DataFrame({"user_id": ["u1"]}),
            train=pd.DataFrame({"other": ["u1"]}),
        )
