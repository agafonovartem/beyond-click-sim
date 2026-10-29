from __future__ import annotations

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from beyond_click_sim.scorers.constant import select_user_history_rows
from beyond_click_sim.tasks import RandomFractionSplitter, TaskBuilder


def test_random_fraction_splitter_can_stratify_global_rows() -> None:
    interactions = pd.DataFrame(
        {
            "interaction_id": [f"r{idx}" for idx in range(24)],
            "target": [0, 1] * 12,
        }
    )

    split = RandomFractionSplitter(
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
        seed=0,
        group_column=None,
        stratify_column="target",
        shuffle=True,
    ).split(interactions)

    assert len(split.train) == 12
    assert len(split.val) == 6
    assert len(split.test) == 6
    assert split.train["target"].value_counts().to_dict() == {0: 6, 1: 6}
    assert split.val["target"].value_counts().to_dict() == {0: 3, 1: 3}
    assert split.test["target"].value_counts().to_dict() == {0: 3, 1: 3}


@pytest.mark.parametrize("shuffle", [False, True])
def test_random_fraction_splitter_grouped_uses_sklearn_counts(shuffle: bool) -> None:
    interactions = pd.DataFrame(
        {
            "interaction_id": [f"u1-{idx}" for idx in range(15)]
            + [f"u2-{idx}" for idx in range(10)],
            "user_id": ["u1"] * 15 + ["u2"] * 10,
            "timestamp": list(range(25)),
        }
    )

    split = RandomFractionSplitter(
        train_fraction=0.7,
        val_fraction=0.0,
        test_fraction=0.3,
        seed=0,
        group_column="user_id",
        stratify_column=None,
        shuffle=shuffle,
    ).split(interactions)

    assert split.test["user_id"].value_counts().to_dict() == {"u1": 5, "u2": 3}
    assert split.train["user_id"].value_counts().to_dict() == {"u1": 10, "u2": 7}


def _temporal_interactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "user_id": user_id,
                "item_id": f"{user_id}-i{time}",
                "timestamp": time + offset,
            }
            for time in range(10, 0, -1)
            for user_id, offset in (("u2", 100), ("u1", 0))
        ],
        index=range(100, 120),
    )


def test_default_split_preserves_each_users_chronology_and_latest_history() -> None:
    interactions = _temporal_interactions()
    original = interactions.copy(deep=True)
    splitter = RandomFractionSplitter(
        train_fraction=0.7, val_fraction=0.1, test_fraction=0.2
    )

    split = splitter.split(interactions)

    # Preserve sklearn's existing two-stage rounding: ten rows yield 6/2/2.
    for user_id, offset in (("u1", 0), ("u2", 100)):
        for frame, times in (
            (split.train, range(1, 7)),
            (split.val, [7, 8]),
            (split.test, [9, 10]),
        ):
            assert frame.loc[frame.user_id == user_id, "timestamp"].tolist() == [
                time + offset for time in times
            ]
        combined = pd.concat([split.train, split.val, split.test])
        assert set(combined.loc[combined.user_id == user_id, "item_id"]) == {
            f"{user_id}-i{time}" for time in range(1, 11)
        }

    assert sum(map(len, (split.train, split.val, split.test))) == len(interactions)
    for frame in (split.train, split.val, split.test):
        assert frame.index.tolist() == list(range(len(frame)))
    histories = select_user_history_rows(split.train, max_history_items=3)
    assert histories["u1"].timestamp.tolist() == [4, 5, 6]
    assert histories["u2"].timestamp.tolist() == [104, 105, 106]
    pd.testing.assert_frame_equal(interactions, original)

    manifest = TaskBuilder._component_manifest(splitter)
    assert manifest["shuffle"] is False
    assert manifest["timestamp_column"] == "timestamp"


def test_temporal_split_is_independent_of_seed_and_distinct_timestamp_input_order() -> None:
    interactions = _temporal_interactions()
    expected = RandomFractionSplitter(0.7, 0.1, 0.2).split(interactions)

    for seed in (0, 7, 42):
        actual = RandomFractionSplitter(0.7, 0.1, 0.2, seed=seed).split(
            interactions.sample(frac=1, random_state=seed)
        )
        for name in ("train", "val", "test"):
            pd.testing.assert_frame_equal(getattr(actual, name), getattr(expected, name))


def test_temporal_split_preserves_input_order_for_timestamp_ties() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": ["u1"] * 10,
            "item_id": ["d", "a", "c", "b", "j", "f", "i", "e", "h", "g"],
            "timestamp": [2] * 6 + [1] * 4,
        }
    )

    split = RandomFractionSplitter().split(interactions)

    assert split.train.item_id.tolist() == ["i", "e", "h", "g", "d", "a", "c"]
    assert split.test.item_id.tolist() == ["b", "j", "f"]
    assert split.train.timestamp.max() == split.test.timestamp.min() == 2
    assert split.val.empty


@pytest.mark.parametrize("stratify_column", ["target", "absent_column"])
def test_temporal_split_ignores_stratification(stratify_column: str) -> None:
    interactions = _temporal_interactions().assign(target=[0] * 10 + [1] * 10)
    expected = RandomFractionSplitter(0.7, 0.1, 0.2).split(interactions)

    actual = RandomFractionSplitter(
        0.7, 0.1, 0.2, stratify_column=stratify_column
    ).split(interactions)

    for name in ("train", "val", "test"):
        pd.testing.assert_frame_equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("datetime_timestamps", [False, True])
def test_temporal_split_supports_global_order_and_custom_timestamp_column(
    datetime_timestamps: bool,
) -> None:
    times = [10, 2, 7, 8, 4, 1, 6, 3, 9, 5]
    interactions = pd.DataFrame(
        {
            "user_id": ["u2", "u1"] * 5,
            "item_id": [f"i{time}" for time in times],
            "event_time": (
                pd.to_datetime(times, unit="s", utc=True)
                if datetime_timestamps
                else times
            ),
        }
    )

    split = RandomFractionSplitter(
        0.7, 0.1, 0.2, group_column=None, timestamp_column="event_time"
    ).split(interactions)

    assert split.train.item_id.tolist() == [f"i{time}" for time in range(1, 7)]
    assert split.val.item_id.tolist() == ["i7", "i8"]
    assert split.test.item_id.tolist() == ["i9", "i10"]


@pytest.mark.parametrize("group_column", [None, "user_id"])
@pytest.mark.parametrize("stratify_column", [None, "target"])
def test_random_mode_matches_previous_two_stage_sklearn_split(
    group_column: str | None,
    stratify_column: str | None,
) -> None:
    interactions = pd.DataFrame(
        {
            "interaction_id": [f"r{idx}" for idx in range(48)],
            "user_id": ["u2"] * 24 + ["u1"] * 24,
            "target": [0, 1] * 24,
        }
    )
    actual = RandomFractionSplitter(
        0.5,
        0.25,
        0.25,
        seed=7,
        group_column=group_column,
        stratify_column=stratify_column,
        shuffle=True,
    ).split(interactions)
    groups = (
        [interactions]
        if group_column is None
        else [group for _, group in interactions.groupby(group_column, sort=True)]
    )
    expected: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    for position, group in enumerate(groups):
        random_state = 7 + position
        train_val, test = train_test_split(
            group,
            test_size=0.25,
            random_state=random_state,
            stratify=None if stratify_column is None else group[stratify_column],
        )
        train, val = train_test_split(
            train_val,
            test_size=0.25 / 0.75,
            random_state=random_state,
            stratify=None if stratify_column is None else train_val[stratify_column],
        )
        for name, frame in (("train", train), ("val", val), ("test", test)):
            expected[name].append(frame)

    for name, parts in expected.items():
        pd.testing.assert_frame_equal(
            getattr(actual, name), pd.concat(parts, ignore_index=True)
        )


@pytest.mark.parametrize(
    ("timestamp_values", "message"),
    [
        (None, "requires timestamp column"),
        ([1, None, 3, 4], "requires non-null timestamps"),
        (["1", "2", "3", "4"], "requires numeric or datetime timestamps"),
    ],
)
def test_temporal_split_rejects_invalid_timestamps(
    timestamp_values: list[object] | None,
    message: str,
) -> None:
    interactions = pd.DataFrame({"user_id": ["u1"] * 4})
    if timestamp_values is not None:
        interactions["timestamp"] = timestamp_values

    with pytest.raises(ValueError, match=message):
        RandomFractionSplitter().split(interactions)


@pytest.mark.parametrize("shuffle", [False, True])
def test_fraction_splitter_preserves_empty_input_behavior(shuffle: bool) -> None:
    interactions = pd.DataFrame(columns=["user_id", "item_id"])

    split = RandomFractionSplitter(shuffle=shuffle).split(interactions)

    for frame in (split.train, split.val, split.test):
        pd.testing.assert_frame_equal(frame, interactions)
        assert frame is not interactions


def test_temporal_split_sorts_even_when_all_rows_are_training() -> None:
    interactions = _temporal_interactions()

    split = RandomFractionSplitter(1.0, 0.0, 0.0).split(interactions)

    assert split.val.empty and split.test.empty
    assert split.train.groupby("user_id").timestamp.apply(
        lambda values: values.is_monotonic_increasing
    ).all()
    assert len(split.train) == len(interactions)


@pytest.mark.parametrize("fractions", [(-0.1, 0.1, 1.0), (0.7, 0.1, 0.3)])
def test_fraction_splitter_rejects_invalid_fractions(fractions: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="Split fractions"):
        RandomFractionSplitter(*fractions).split(_temporal_interactions())


def test_temporal_split_preserves_small_group_error_context() -> None:
    with pytest.raises(ValueError, match="Cannot split group user_id='u1'"):
        RandomFractionSplitter().split(pd.DataFrame({"user_id": ["u1"], "timestamp": [1]}))
