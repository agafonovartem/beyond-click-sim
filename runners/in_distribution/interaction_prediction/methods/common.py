from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

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


def prediction_frame(
    *,
    split: str,
    X: pd.DataFrame,
    y: pd.Series,
    scores: pd.Series,
    predictions: pd.Series,
) -> pd.DataFrame:
    frame = X.copy()
    frame.insert(0, "split", split)
    frame["target"] = y
    frame["score"] = scores
    frame["prediction"] = predictions
    return frame


def candidate_group_summary(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    candidate_group_column: str,
    sampled_column: str | None,
) -> dict[str, int | float]:
    frame = pd.DataFrame(
        {
            "group": X[candidate_group_column].to_numpy(),
            "target": y.astype(int).to_numpy(),
        }
    )
    if sampled_column is not None and sampled_column in X.columns:
        frame["sampled"] = X[sampled_column].astype(bool).to_numpy().astype(int)
    else:
        frame["sampled"] = 0

    grouped = frame.groupby("group", sort=False).agg(
        rows=("target", "size"),
        positives=("target", "sum"),
        sampled=("sampled", "sum"),
    )
    if grouped.empty:
        return {
            "groups": 0,
            "rows_min": 0,
            "rows_mean": 0.0,
            "rows_max": 0,
            "positives_min": 0,
            "positives_mean": 0.0,
            "positives_max": 0,
            "sampled_min": 0,
            "sampled_mean": 0.0,
            "sampled_max": 0,
        }
    return {
        "groups": int(len(grouped)),
        "rows_min": int(grouped["rows"].min()),
        "rows_mean": float(grouped["rows"].mean()),
        "rows_max": int(grouped["rows"].max()),
        "positives_min": int(grouped["positives"].min()),
        "positives_mean": float(grouped["positives"].mean()),
        "positives_max": int(grouped["positives"].max()),
        "sampled_min": int(grouped["sampled"].min()),
        "sampled_mean": float(grouped["sampled"].mean()),
        "sampled_max": int(grouped["sampled"].max()),
    }


def limit_candidate_groups(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    candidate_group_column: str,
    max_candidate_groups: int | None,
) -> tuple[pd.DataFrame, pd.Series]:
    if max_candidate_groups is None:
        return X, y
    groups = X[candidate_group_column].drop_duplicates().head(max_candidate_groups)
    mask = X[candidate_group_column].isin(groups)
    return X.loc[mask].copy(), y.loc[mask].copy()
