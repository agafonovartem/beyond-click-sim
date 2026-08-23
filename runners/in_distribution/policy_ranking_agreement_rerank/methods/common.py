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
    min_coverage: float = 0.90,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    """Aggregate simulated and real utilities per policy.

    Two differences from the free-retrieval families this was forked from.

    **The reference does not move.** Real utility is the mean target over *every*
    slate row of a policy, whether or not the scorer produced a score for it.
    Upstream, `y_test` is pre-filtered to successfully-parsed rows, so a method
    with parse failures is compared against a slightly different reference than
    one without — which makes two methods' tau incommensurable. Pass the full
    frame here and let the simulated side skip NaN instead.

    **Coverage is enforced per policy.** A run can be 95% covered overall while
    one policy sits at 60%; that policy's displacement changes the ordering,
    which is the entire output. Raises below `min_coverage` for any policy.

    Utilities are still restricted to users covered by every policy. With a
    re-ranking protocol whose `rank_candidates` always returns a permutation
    that restriction is a no-op, and the assertion below keeps it that way.
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

    dropped = {
        str(policy): sorted(users - common_users)
        for policy, users in users_by_policy.items()
        if users - common_users
    }
    if dropped:
        raise ValueError(
            "Re-ranking slates must cover the same users under every policy, but "
            f"the common-user intersection dropped some: { {k: len(v) for k, v in dropped.items()} }. "
            "rank_candidates must return a full permutation for every user."
        )

    restricted = frame[frame["user"].isin(common_users)]
    scored = restricted[restricted["score"].notna()]

    coverage_by_policy = {
        str(policy): {
            "scored_rows": int((scored["policy"] == policy).sum()),
            "requested_rows": int((restricted["policy"] == policy).sum()),
        }
        for policy in users_by_policy.index
    }
    for policy, counts in coverage_by_policy.items():
        requested = counts["requested_rows"]
        counts["coverage"] = (
            float(counts["scored_rows"] / requested) if requested else 0.0
        )
        if counts["coverage"] < min_coverage:
            raise ValueError(
                f"Policy {policy!r} scored only {counts['coverage']:.1%} of its "
                f"{requested} slate rows, below the {min_coverage:.0%} floor. A "
                "policy scored on a biased subset of its own slates lands in the "
                "wrong place in the ordering, which is the whole measurement."
            )

    coverage: dict[str, Any] = {
        "per_policy": {
            str(policy): {
                "n_users": int(len(users)),
                "n_rows": int((frame["policy"] == policy).sum()),
                **coverage_by_policy[str(policy)],
            }
            for policy, users in users_by_policy.items()
        },
        "common_users": int(len(common_users)),
        "common_rows": int(frame["user"].isin(common_users).sum()),
        "min_coverage": float(min_coverage),
        "overall_coverage": (
            float(len(scored) / len(restricted)) if len(restricted) else 0.0
        ),
    }

    # Simulated skips unscored rows; real deliberately does not.
    sim_util = scored.groupby("policy")["score"].mean().to_dict()
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
