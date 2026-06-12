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


def test_post_split_user_sampler_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="n_users"):
        PostSplitUserSampler(n_users=0)

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
