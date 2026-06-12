from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(
    y_true: pd.Series,
    scores: pd.Series,
) -> dict[str, float | int]:
    """Compute basic numeric prediction metrics."""

    _validate_regression_inputs(y_true, scores)
    true = y_true.astype(float)
    pred = scores.astype(float)
    mse = float(mean_squared_error(true, pred))
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mse)),
        "n": int(len(true)),
    }


def user_grouped_regression_metrics(
    y_true: pd.Series,
    scores: pd.Series,
    users: pd.Series,
) -> dict[str, float | int]:
    """Compute regression metrics per user, then average users equally."""

    _validate_regression_inputs(y_true, scores)
    _require_same_length(y_true, users, left_name="y_true", right_name="users")
    if users.isna().any():
        raise ValueError("users contains NA values")

    frame = pd.DataFrame(
        {
            "user": users.to_numpy(),
            "target": y_true.astype(float).to_numpy(),
            "score": scores.astype(float).to_numpy(),
        }
    )
    per_user = frame.groupby("user", sort=False).apply(
        lambda group: pd.Series(
            {
                "mae": mean_absolute_error(group["target"], group["score"]),
                "rmse": np.sqrt(mean_squared_error(group["target"], group["score"])),
                "n": len(group),
            }
        ),
        include_groups=False,
    )
    return {
        "mae": float(per_user["mae"].mean()),
        "rmse": float(per_user["rmse"].mean()),
        "n_users": int(len(per_user)),
        "n": int(len(frame)),
    }


def _validate_regression_inputs(y_true: pd.Series, scores: pd.Series) -> None:
    _require_same_length(y_true, scores, left_name="y_true", right_name="scores")
    if len(y_true) == 0:
        raise ValueError("Cannot compute regression metrics on empty inputs")
    if y_true.isna().any():
        raise ValueError("y_true contains NA values")
    if scores.isna().any():
        raise ValueError("scores contains NaN values")


def _require_same_length(
    left: pd.Series,
    right: pd.Series,
    *,
    left_name: str,
    right_name: str,
) -> None:
    if len(left) != len(right):
        raise ValueError(f"{left_name} and {right_name} must have the same length")
