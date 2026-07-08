from __future__ import annotations

import pandas as pd
import pytest

from beyond_click_sim.tasks import PostSplitUserSampler, TemporalColdStartCandidateSampler


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
# TemporalColdStartCandidateSampler
# ---------------------------------------------------------------------------

def _temporal_sampler_inputs():
    """User u1 with 6 positives in temporal order; 10 extra items as negatives."""
    positives = pd.DataFrame({
        "user_id": ["u1"] * 6,
        "item_id": ["i1", "i2", "i3", "i4", "i5", "i6"],
    })
    items = pd.DataFrame({
        "item_id": [f"i{j}" for j in range(1, 7)] + [f"n{j}" for j in range(1, 11)],
    })
    # All observed interactions for u1 (used to exclude from negatives).
    interactions = pd.DataFrame({
        "user_id": ["u1"] * 6,
        "item_id": ["i1", "i2", "i3", "i4", "i5", "i6"],
    })
    return positives, items, interactions


def test_temporal_sampler_first_window_uses_first_positives() -> None:
    positives, items, interactions = _temporal_sampler_inputs()
    # negative_ratio=1, total_items=2 → max_positive_items=1 → one positive per group
    sampler = TemporalColdStartCandidateSampler(
        negative_ratio=1,
        total_items=2,
        max_candidate_groups_per_user=2,
        group_offset=0,
        seed=0,
    )
    result = sampler.sample(positives, interactions=interactions, items=items)

    assert result["candidate_group"].nunique() == 2
    positive_items = set(result.loc[result["target"] == 1, "item_id"])
    assert positive_items == {"i1", "i2"}


def test_temporal_sampler_offset_skips_to_later_window() -> None:
    positives, items, interactions = _temporal_sampler_inputs()
    sampler = TemporalColdStartCandidateSampler(
        negative_ratio=1,
        total_items=2,
        max_candidate_groups_per_user=2,
        group_offset=2,
        seed=0,
    )
    result = sampler.sample(positives, interactions=interactions, items=items)

    assert result["candidate_group"].nunique() == 2
    positive_items = set(result.loc[result["target"] == 1, "item_id"])
    assert positive_items == {"i3", "i4"}


def test_temporal_sampler_windows_are_non_overlapping() -> None:
    positives, items, interactions = _temporal_sampler_inputs()
    sampler0 = TemporalColdStartCandidateSampler(
        negative_ratio=1, total_items=2, max_candidate_groups_per_user=2,
        group_offset=0, seed=0,
    )
    sampler1 = TemporalColdStartCandidateSampler(
        negative_ratio=1, total_items=2, max_candidate_groups_per_user=2,
        group_offset=2, seed=1,
    )
    result0 = sampler0.sample(positives, interactions=interactions, items=items)
    result1 = sampler1.sample(positives, interactions=interactions, items=items)

    positives0 = set(result0.loc[result0["target"] == 1, "item_id"])
    positives1 = set(result1.loc[result1["target"] == 1, "item_id"])
    assert positives0.isdisjoint(positives1)


def test_temporal_sampler_group_ids_use_global_chunk_index() -> None:
    positives, items, interactions = _temporal_sampler_inputs()
    sampler = TemporalColdStartCandidateSampler(
        negative_ratio=1, total_items=2, max_candidate_groups_per_user=2,
        group_offset=2, seed=0,
    )
    result = sampler.sample(positives, interactions=interactions, items=items)

    group_ids = set(result["candidate_group"].unique())
    # Local positions 0,1 at offset 2 → global chunks 2,3
    assert "candidate:user:u1:chunk:2" in group_ids
    assert "candidate:user:u1:chunk:3" in group_ids
    assert "candidate:user:u1:chunk:0" not in group_ids


def test_temporal_sampler_negatives_are_not_observed_items() -> None:
    positives, items, interactions = _temporal_sampler_inputs()
    sampler = TemporalColdStartCandidateSampler(
        negative_ratio=1, total_items=2, max_candidate_groups_per_user=2,
        group_offset=0, seed=0,
    )
    result = sampler.sample(positives, interactions=interactions, items=items)

    observed = {"i1", "i2", "i3", "i4", "i5", "i6"}
    negatives = result.loc[result["target"] == 0, "item_id"]
    assert not negatives.isin(observed).any()


def test_temporal_sampler_max_eval_users_caps_user_count() -> None:
    positives = pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2"],
        "item_id": ["i1", "i2", "i3", "i4"],
    })
    items = pd.DataFrame({
        "item_id": ["i1", "i2", "i3", "i4", "n1", "n2", "n3", "n4"],
    })
    interactions = positives.copy()

    sampler = TemporalColdStartCandidateSampler(
        negative_ratio=1, total_items=2, max_eval_users=1,
        max_candidate_groups_per_user=2, group_offset=0, seed=0,
    )
    result = sampler.sample(positives, interactions=interactions, items=items)

    assert result["user_id"].nunique() == 1


def test_temporal_sampler_offset_beyond_chunks_returns_empty() -> None:
    positives, items, interactions = _temporal_sampler_inputs()
    # u1 has 6 items → 6 chunks of size 1. Offset 10 → no chunks available.
    sampler = TemporalColdStartCandidateSampler(
        negative_ratio=1, total_items=2, max_candidate_groups_per_user=2,
        group_offset=10, seed=0,
    )
    result = sampler.sample(positives, interactions=interactions, items=items)

    assert result.empty


def test_temporal_sampler_empty_positives_returns_empty_frame() -> None:
    positives = pd.DataFrame(columns=["user_id", "item_id"])
    items = pd.DataFrame({"item_id": ["i1", "i2"]})
    interactions = pd.DataFrame(columns=["user_id", "item_id"])

    sampler = TemporalColdStartCandidateSampler(negative_ratio=1, total_items=2, seed=0)
    result = sampler.sample(positives, interactions=interactions, items=items)

    assert result.empty
