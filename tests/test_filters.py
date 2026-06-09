from __future__ import annotations

import pandas as pd

from beyond_click_sim.tasks import (
    MinUserInteractionsFilter,
    SampleUsersFilter,
    SequentialDatasetFilter,
)


def test_sample_users_filter_is_seeded_and_order_independent() -> None:
    users, items, interactions = _toy_frames()
    dataset_filter = SampleUsersFilter(n_users=3, seed=0)

    first_users, first_items, first_interactions = dataset_filter.filter(
        users=users,
        items=items,
        interactions=interactions,
    )
    second_users, _, second_interactions = dataset_filter.filter(
        users=users.iloc[::-1].reset_index(drop=True),
        items=items,
        interactions=interactions.iloc[::-1].reset_index(drop=True),
    )

    first_selected = set(first_users["user_id"])
    second_selected = set(second_users["user_id"])
    assert first_selected == second_selected
    assert len(first_selected) == 3
    assert set(first_interactions["user_id"]) == first_selected
    assert set(second_interactions["user_id"]) == second_selected
    assert first_items.equals(items)


def test_sample_users_filter_keeps_all_when_limit_is_larger_than_population() -> None:
    users, items, interactions = _toy_frames()

    filtered_users, filtered_items, filtered_interactions = SampleUsersFilter(
        n_users=10,
        seed=0,
    ).filter(users=users, items=items, interactions=interactions)

    assert filtered_users["user_id"].tolist() == users["user_id"].tolist()
    assert filtered_items.equals(items)
    assert filtered_interactions["interaction_id"].tolist() == interactions[
        "interaction_id"
    ].tolist()


def test_sequential_dataset_filter_applies_filters_in_order() -> None:
    users, items, interactions = _toy_frames()
    dataset_filter = SequentialDatasetFilter(
        [
            MinUserInteractionsFilter(min_interactions=3),
            SampleUsersFilter(n_users=2, seed=0),
        ]
    )

    filtered_users, _, filtered_interactions = dataset_filter.filter(
        users=users,
        items=items,
        interactions=interactions,
    )

    selected_users = set(filtered_users["user_id"])
    assert len(selected_users) == 2
    assert selected_users <= {"u3", "u4", "u5"}
    assert set(filtered_interactions["user_id"]) == selected_users


def _toy_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users = pd.DataFrame({"user_id": [f"u{idx}" for idx in range(1, 6)]})
    items = pd.DataFrame({"item_id": [f"i{idx}" for idx in range(1, 4)]})
    interactions = pd.DataFrame(
        {
            "interaction_id": [f"r{idx}" for idx in range(1, 16)],
            "user_id": [
                "u1",
                "u2",
                "u2",
                "u3",
                "u3",
                "u3",
                "u4",
                "u4",
                "u4",
                "u4",
                "u5",
                "u5",
                "u5",
                "u5",
                "u5",
            ],
            "item_id": ["i1"] * 15,
        }
    )
    return users, items, interactions
