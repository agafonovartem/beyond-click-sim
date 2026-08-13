from __future__ import annotations

import pandas as pd
import pytest

from beyond_click_sim.tasks import PerTargetCandidateSampler, PostSplitUserSampler


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


# ---------------------------------------------------------------------------
# PerTargetCandidateSampler
# ---------------------------------------------------------------------------


def _per_target_inputs():
    """Two frame rows for u1 (targets i1, i2); 10 extra catalog items."""
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_id": ["i1", "i2"],
        }
    )
    items = pd.DataFrame(
        {"item_id": ["i1", "i2"] + [f"n{j}" for j in range(1, 11)]}
    )
    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_id": ["i1", "i2"],
        }
    )
    return frame, items, interactions


def test_per_target_sampler_builds_one_group_per_frame_row() -> None:
    frame, items, interactions = _per_target_inputs()
    sampler = PerTargetCandidateSampler(negative_ratio=3, seed=0)

    result = sampler.sample(frame, interactions=interactions, items=items)

    assert result["candidate_group"].nunique() == 2
    assert set(result["candidate_group"]) == {
        "candidate:user:u1:item:i1",
        "candidate:user:u1:item:i2",
    }
    for _, group in result.groupby("candidate_group"):
        assert len(group) == 4  # 1 positive + 3 negatives
        positives = group.loc[group["target"] == 1]
        assert len(positives) == 1
        assert not positives["sampled"].iloc[0]
        negatives = group.loc[group["target"] == 0]
        assert len(negatives) == 3
        assert negatives["sampled"].all()
    assert sampler.diagnostics_ == {"groups": 2, "bin_fallback_groups": 0}


def test_per_target_sampler_negatives_are_not_observed_items() -> None:
    frame, items, interactions = _per_target_inputs()
    sampler = PerTargetCandidateSampler(negative_ratio=5, seed=0)

    result = sampler.sample(frame, interactions=interactions, items=items)

    negatives = result.loc[result["target"] == 0, "item_id"]
    assert not negatives.isin({"i1", "i2"}).any()
    # within a group, negatives are unique
    for _, group in result.groupby("candidate_group"):
        assert group["item_id"].is_unique


def test_per_target_sampler_negative_draw_depends_only_on_seed() -> None:
    frame, items, interactions = _per_target_inputs()
    first = PerTargetCandidateSampler(negative_ratio=3, seed=0).sample(
        frame, interactions=interactions, items=items
    )
    second = PerTargetCandidateSampler(negative_ratio=3, seed=0).sample(
        frame, interactions=interactions, items=items
    )
    third = PerTargetCandidateSampler(negative_ratio=3, seed=1).sample(
        frame, interactions=interactions, items=items
    )

    pd.testing.assert_frame_equal(first, second)
    # same targets, (almost surely) different negatives under a new seed
    assert set(first.loc[first["target"] == 1, "item_id"]) == set(
        third.loc[third["target"] == 1, "item_id"]
    )
    assert set(first.loc[first["target"] == 0, "item_id"]) != set(
        third.loc[third["target"] == 0, "item_id"]
    )


def test_per_target_sampler_empty_frame_returns_empty() -> None:
    frame = pd.DataFrame(columns=["user_id", "item_id"])
    items = pd.DataFrame({"item_id": ["i1", "i2"]})
    interactions = pd.DataFrame(columns=["user_id", "item_id"])
    sampler = PerTargetCandidateSampler(negative_ratio=1, seed=0)

    result = sampler.sample(frame, interactions=interactions, items=items)

    assert result.empty
    assert sampler.diagnostics_ == {"groups": 0, "bin_fallback_groups": 0}


def test_per_target_sampler_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="strategy"):
        PerTargetCandidateSampler(negative_ratio=1, strategy="popularity")
    with pytest.raises(ValueError, match="n_popularity_bins"):
        PerTargetCandidateSampler(negative_ratio=1, n_popularity_bins=0)

    frame, items, interactions = _per_target_inputs()
    sampler = PerTargetCandidateSampler(negative_ratio=0)
    with pytest.raises(ValueError, match="negative_ratio"):
        sampler.sample(frame, interactions=interactions, items=items)

    matched = PerTargetCandidateSampler(
        negative_ratio=1, strategy="popularity_matched"
    )
    with pytest.raises(ValueError, match="train_interactions"):
        matched.sample(frame, interactions=interactions, items=items)


def _popularity_matched_inputs():
    """Catalog with a clear popularity split: 4 hot items, 4 cold items."""
    hot = [f"hot{j}" for j in range(4)]
    cold = [f"cold{j}" for j in range(4)]
    items = pd.DataFrame({"item_id": hot + cold})
    # warm train: hot items consumed 100 times each, cold items once each;
    # equal group sizes so the q=2 median separates the two groups
    train = pd.DataFrame(
        {
            "user_id": [f"w{j}" for j in range(100) for _ in hot] + ["w0"] * 4,
            "item_id": hot * 100 + cold,
        }
    )
    # u1's frame: one hot target, one cold target; observed = the targets
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_id": ["hot0", "cold0"],
        }
    )
    interactions = frame.copy()
    return frame, items, interactions, train


def test_popularity_matched_negatives_come_from_target_bin() -> None:
    frame, items, interactions, train = _popularity_matched_inputs()
    sampler = PerTargetCandidateSampler(
        negative_ratio=2,
        strategy="popularity_matched",
        n_popularity_bins=2,
        seed=0,
    )

    result = sampler.sample(
        frame,
        interactions=interactions,
        items=items,
        train_interactions=train,
    )

    for _, group in result.groupby("candidate_group"):
        target_item = group.loc[group["target"] == 1, "item_id"].iloc[0]
        negatives = group.loc[group["target"] == 0, "item_id"].tolist()
        if target_item.startswith("hot"):
            assert all(item.startswith("hot") for item in negatives)
        else:
            assert all(item.startswith("cold") for item in negatives)
    # bins are big enough here: no fallback needed
    assert sampler.diagnostics_["bin_fallback_groups"] == 0


def test_popularity_matched_falls_back_to_adjacent_bins_and_counts_it() -> None:
    # star sits alone in the top bin; m=2 negatives force adjacent-bin fallback
    items = pd.DataFrame({"item_id": ["star", "a", "b", "c"]})
    train = pd.DataFrame(
        {
            "user_id": (
                [f"w{j}" for j in range(100)]
                + [f"w{j}" for j in range(50)]
                + [f"w{j}" for j in range(50)]
                + ["w0"]
            ),
            "item_id": ["star"] * 100 + ["a"] * 50 + ["b"] * 50 + ["c"],
        }
    )
    frame = pd.DataFrame({"user_id": ["u1"], "item_id": ["star"]})
    interactions = frame.copy()
    sampler = PerTargetCandidateSampler(
        negative_ratio=2,
        strategy="popularity_matched",
        n_popularity_bins=3,
        seed=0,
    )

    result = sampler.sample(
        frame,
        interactions=interactions,
        items=items,
        train_interactions=train,
    )

    negatives = result.loc[result["target"] == 0, "item_id"].tolist()
    assert len(negatives) == 2
    assert set(negatives) <= {"a", "b", "c"}
    assert sampler.diagnostics_["bin_fallback_groups"] == 1


def test_popularity_matched_is_deterministic_per_seed() -> None:
    frame, items, interactions, train = _popularity_matched_inputs()
    kwargs = dict(
        negative_ratio=2, strategy="popularity_matched", n_popularity_bins=2
    )
    first = PerTargetCandidateSampler(seed=0, **kwargs).sample(
        frame, interactions=interactions, items=items, train_interactions=train
    )
    second = PerTargetCandidateSampler(seed=0, **kwargs).sample(
        frame, interactions=interactions, items=items, train_interactions=train
    )

    pd.testing.assert_frame_equal(first, second)
