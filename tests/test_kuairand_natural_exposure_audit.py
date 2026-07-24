from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.ood.kuairand_policy_shift.natural_exposure_audit import (
    build_pair_novel_target_pool,
    common_exposure_supported_items,
    feasibility_grid,
    label_blind_cohort,
    select_label_blind_candidates,
    source_item_degrees,
)


def test_natural_cohort_uses_counts_not_click_labels() -> None:
    source = _rows(window="source")
    targets = _rows(window="target")

    first = label_blind_cohort(
        source,
        targets,
        history_size=2,
        candidate_size=2,
    )
    flipped = targets.copy()
    flipped["target_click"] = 1 - flipped["target_click"]
    second = label_blind_cohort(
        source,
        flipped,
        history_size=2,
        candidate_size=2,
    )

    assert first == ["u1", "u2"]
    assert second == first


def test_candidate_membership_is_label_blind_and_fixed_size() -> None:
    targets = _rows(window="target")
    selected = select_label_blind_candidates(
        targets,
        users=["u1", "u2"],
        candidate_size=2,
        seed=0,
        split="test",
    )
    flipped = targets.copy()
    flipped["target_click"] = 1 - flipped["target_click"]
    selected_flipped = select_label_blind_candidates(
        flipped,
        users=["u1", "u2"],
        candidate_size=2,
        seed=0,
        split="test",
    )

    for policy in ("standard", "random"):
        keys = ["user_id", "item_id", "interaction_id"]
        assert selected[policy][keys].equals(selected_flipped[policy][keys])
        assert selected[policy].groupby("user_id").size().eq(2).all()


def test_pair_novel_pool_removes_previously_seen_edges() -> None:
    source = _rows(window="source")
    targets = _rows(window="target")
    rows = pd.concat([source, targets], ignore_index=True)
    other_tab_prior = source.iloc[[0]].copy()
    other_tab_prior["user_id"] = "u2"
    other_tab_prior["item_id"] = "u2-random-target-b"
    other_tab_prior["tab"] = 0
    all_rows = pd.concat([rows, other_tab_prior], ignore_index=True)
    allowed_items = set(rows["item_id"])
    pool, audit = build_pair_novel_target_pool(
        rows.loc[rows["tab"].eq(1)],
        all_interactions=all_rows,
        split_start=20220504,
        split_end=20220508,
        allowed_items=allowed_items,
    )

    assert not (
        (pool["user_id"] == "u1") & (pool["item_id"] == "shared")
    ).any()
    assert not (
        (pool["user_id"] == "u2")
        & (pool["item_id"] == "u2-random-target-b")
    ).any()
    assert audit["pair_novelty_violations"] == 0


def test_exposure_support_does_not_require_positive_clicks() -> None:
    source = _rows(window="source")
    degrees = source_item_degrees(source)
    supported = common_exposure_supported_items(
        degrees,
        min_exposure_users_per_policy=1,
    )

    # "never-clicked" is exposed under both policies but has no positives.
    assert "never-clicked" in supported


def test_feasibility_grid_reports_h_by_m_counts() -> None:
    grid = feasibility_grid(
        _rows(window="source"),
        _rows(window="target"),
        history_sizes=(2, 3),
        candidate_sizes=(2, 3),
    )

    assert grid == [
        {"history_size": 2, "candidate_size": 2, "users": 2},
        {"history_size": 2, "candidate_size": 3, "users": 2},
        {"history_size": 3, "candidate_size": 2, "users": 2},
        {"history_size": 3, "candidate_size": 3, "users": 2},
    ]


def _rows(*, window: str) -> pd.DataFrame:
    date = 20220425 if window == "source" else 20220505
    rows = []
    for user_index, user_id in enumerate(("u1", "u2")):
        for policy_index, policy in enumerate(("standard", "random")):
            item_ids = (
                ["shared", "never-clicked", f"{user_id}-{policy}-source"]
                if window == "source"
                else [
                    "shared",
                    "never-clicked",
                    f"{user_id}-{policy}-target-a",
                    f"{user_id}-{policy}-target-b",
                ]
            )
            for position, item_id in enumerate(item_ids):
                rows.append(
                    {
                        "interaction_id": (
                            f"{window}-{user_id}-{policy}-{position}"
                        ),
                        "user_id": user_id,
                        "item_id": item_id,
                        "exposure_policy": policy,
                        "tab": 1,
                        "date": date,
                        "timestamp_ms": (
                            date * 10_000
                            + user_index * 100
                            + policy_index * 10
                            + position
                        ),
                        "target_click": int(
                            item_id != "never-clicked" and position % 2 == 0
                        ),
                    }
                )
    return pd.DataFrame(rows)
