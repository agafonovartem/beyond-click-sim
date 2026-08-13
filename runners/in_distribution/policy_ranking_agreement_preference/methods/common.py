"""Shared utilities for policy ranking agreement method runners."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from beyond_click_sim.tasks import Task, split_xy

POLICY_METRICS_FILENAME = "policy_metrics.json"


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
    user_column: str = "user_id",
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    """Aggregate simulated and real utilities per policy over common-coverage users.

    Simulated utility = mean simulated score per policy.
    Real utility = mean target value per policy (mean positive-preference rate
    on held-out data).

    Utilities are restricted to users that have at least one recommendation row
    under *every* policy, so all policies are compared over the same user
    population. Preference-trained CF policies produce no rows for users with
    zero positive-preference train rows; without this restriction each policy's
    mean would be computed over a slightly different user set.

    Returns
    -------
    (simulated_utilities, real_utilities, coverage):
        one float scalar per policy name, plus per-policy coverage stats
        (`n_users`/`n_rows` before restriction, `common_users` after).
    """
    frame = pd.DataFrame(
        {
            "user": X_test[user_column].to_numpy(),
            "policy": X_test[policy_column].to_numpy(),
            "score": scores.to_numpy(),
            "target": y_test.to_numpy(),
        }
    )
    users_by_policy = frame.groupby("policy")["user"].agg(set)
    if users_by_policy.empty:
        raise ValueError("compute_policy_utilities got an empty frame")
    common_users = set.intersection(*users_by_policy.tolist())
    coverage: dict[str, Any] = {
        "per_policy": {
            str(policy): {
                "n_users": int(len(users)),
                "n_rows": int((frame["policy"] == policy).sum()),
            }
            for policy, users in users_by_policy.items()
        },
        "common_users": int(len(common_users)),
        "common_rows": int(frame["user"].isin(common_users).sum()),
    }
    restricted = frame[frame["user"].isin(common_users)]
    sim_util = restricted.groupby("policy")["score"].mean().to_dict()
    real_util = restricted.groupby("policy")["target"].mean().to_dict()
    return (
        {str(k): float(v) for k, v in sim_util.items()},
        {str(k): float(v) for k, v in real_util.items()},
        coverage,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_policy_metrics(task: Task, output_dir: Path) -> None:
    """Write per-policy recommendation metrics from task manifest."""
    payload = {
        "task": task.name,
        "policy_recommendation_metrics": task.manifest.get(
            "policy_recommendation_metrics"
        ),
    }
    write_json(output_dir / POLICY_METRICS_FILENAME, payload)


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
