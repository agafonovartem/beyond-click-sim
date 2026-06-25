"""Shared utilities for policy ranking agreement method runners."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from beyond_click_sim.tasks import Task, split_xy


def task_xy(task: Task) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Return train and test (X, y) pairs. Val is omitted (always empty for Q3)."""
    target_column = task.schema.target_column
    return {
        "train": split_xy(task.train, target_column=target_column),
        "test": split_xy(task.test, target_column=target_column),
    }


def compute_policy_utilities(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scores: pd.Series,
    *,
    policy_column: str = "policy",
) -> tuple[dict[str, float], dict[str, float]]:
    """Aggregate simulated and real utilities per policy.

    Simulated utility = mean simulated score per policy.
    Real utility = mean target value per policy (mean hit rate on held-out data).

    Returns
    -------
    (simulated_utilities, real_utilities): one float scalar per policy name.
    """
    frame = pd.DataFrame(
        {
            "policy": X_test[policy_column].to_numpy(),
            "score": scores.to_numpy(),
            "target": y_test.to_numpy(),
        }
    )
    sim_util = frame.groupby("policy")["score"].mean().to_dict()
    real_util = frame.groupby("policy")["target"].mean().to_dict()
    return (
        {str(k): float(v) for k, v in sim_util.items()},
        {str(k): float(v) for k, v in real_util.items()},
    )


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
