from __future__ import annotations

import pandas as pd

from beyond_click_sim.tasks import SplitFrames, Task, TaskSchema


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
    assert task.manifest["seed"] == 0
    assert task.train.equals(train)
    assert task.val.equals(val)
    assert task.test.equals(test)
