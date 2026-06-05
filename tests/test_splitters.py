from __future__ import annotations

import pandas as pd

from beyond_click_sim.tasks import RandomFractionSplitter


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
    ).split(interactions)

    assert len(split.train) == 12
    assert len(split.val) == 6
    assert len(split.test) == 6
    assert split.train["target"].value_counts().to_dict() == {0: 6, 1: 6}
    assert split.val["target"].value_counts().to_dict() == {0: 3, 1: 3}
    assert split.test["target"].value_counts().to_dict() == {0: 3, 1: 3}


def test_random_fraction_splitter_grouped_uses_sklearn_counts() -> None:
    interactions = pd.DataFrame(
        {
            "interaction_id": [f"u1-{idx}" for idx in range(15)]
            + [f"u2-{idx}" for idx in range(10)],
            "user_id": ["u1"] * 15 + ["u2"] * 10,
        }
    )

    split = RandomFractionSplitter(
        train_fraction=0.7,
        val_fraction=0.0,
        test_fraction=0.3,
        seed=0,
        group_column="user_id",
        stratify_column=None,
    ).split(interactions)

    assert split.test["user_id"].value_counts().to_dict() == {"u1": 5, "u2": 3}
    assert split.train["user_id"].value_counts().to_dict() == {"u1": 10, "u2": 7}
