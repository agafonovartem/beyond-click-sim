from __future__ import annotations

import pandas as pd
import pytest

from beyond_click_sim.tasks import SplitFrames, Task, TaskSchema, split_xy


def test_split_frames_names_train_val_test_frames() -> None:
    train = pd.DataFrame({"row": ["train"]})
    val = pd.DataFrame({"row": ["val"]})
    test = pd.DataFrame({"row": ["test"]})

    split = SplitFrames(train=train, val=val, test=test)

    assert split.train.equals(train)
    assert split.val.equals(val)
    assert split.test.equals(test)


def test_task_stores_split_dataframes_and_schema() -> None:
    train = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_id": ["i1"],
            "target": [1],
        }
    )
    val = pd.DataFrame(
        {
            "candidate_set_id": ["u1:0"],
            "user_id": ["u1"],
            "item_id": ["i2"],
            "target": [0],
        }
    )
    test = val.copy()
    schema = TaskSchema(
        target_column="target",
        feature_columns=("user_id", "item_id"),
        candidate_group_column="candidate_set_id",
        history_context_columns=("rating",),
    )

    task = Task(
        name="toy-interaction-ranking",
        train=train,
        val=val,
        test=test,
        schema=schema,
        manifest={"seed": 0},
    )

    assert task.schema.target_column == "target"
    assert task.schema.feature_columns == ("user_id", "item_id")
    assert task.schema.candidate_group_column == "candidate_set_id"
    assert task.schema.history_context_columns == ("rating",)
    assert task.manifest["seed"] == 0
    assert task.train.equals(train)
    assert task.val.equals(val)
    assert task.test.equals(test)


def test_split_xy_separates_target_from_inputs() -> None:
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "item_id": ["i1", "i2"],
            "feature": [10, 20],
            "target": [1, 0],
        },
        index=["a", "b"],
    )

    X, y = split_xy(frame, target_column="target")

    assert list(X.columns) == ["user_id", "item_id", "feature"]
    assert list(X.index) == ["a", "b"]
    assert y.name == "target"
    assert y.tolist() == [1, 0]

    X.loc["a", "feature"] = 999
    y.loc["a"] = 999
    assert frame.loc["a", "feature"] == 10
    assert frame.loc["a", "target"] == 1


def test_split_xy_requires_target_column() -> None:
    frame = pd.DataFrame({"user_id": ["u1"], "item_id": ["i1"]})

    with pytest.raises(ValueError, match="Missing target column"):
        split_xy(frame, target_column="target")
