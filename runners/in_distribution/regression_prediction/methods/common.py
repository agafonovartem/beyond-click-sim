from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from beyond_click_sim.evaluation import (
    regression_metrics,
    user_grouped_regression_metrics,
)
from beyond_click_sim.tasks import Task, split_xy


def task_xy(task: Task) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    target_column = task.schema.target_column
    return {
        "train": split_xy(task.train, target_column=target_column),
        "val": split_xy(task.val, target_column=target_column),
        "test": split_xy(task.test, target_column=target_column),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def current_git_commit(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def score_frame(
    *,
    split: str,
    X: pd.DataFrame,
    y: pd.Series,
    scores: pd.Series,
) -> pd.DataFrame:
    frame = X.copy()
    frame.insert(0, "split", split)
    frame["target"] = y
    frame["score"] = scores
    return frame


def regression_metrics_for_split(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    scores: pd.Series,
) -> dict[str, dict[str, float | int]]:
    return {
        "micro": regression_metrics(y, scores),
        "macro_by_user_mean": user_grouped_regression_metrics(
            y,
            scores,
            X["user_id"],
        ),
    }
