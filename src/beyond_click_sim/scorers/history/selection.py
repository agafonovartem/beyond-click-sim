from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from beyond_click_sim.scorers.constant import select_user_history_positions


@dataclass(frozen=True)
class UserHistory:
    """Selected train-history rows for one user."""

    user_id: Any
    rows: pd.DataFrame
    item_ids: tuple[Any, ...]


def select_history_by_user(
    X: pd.DataFrame,
    *,
    user_column: str = "user_id",
    item_column: str | None = "item_id",
    max_history_items: int | None = 30,
) -> dict[Any, UserHistory]:
    """Select per-user history rows in the same train-order window as LLM scorers."""

    required_columns = [user_column]
    if item_column is not None:
        required_columns.append(item_column)
    missing = [column for column in required_columns if column not in X.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    histories: dict[Any, UserHistory] = {}
    for user_id, selected_positions in select_user_history_positions(
        X,
        user_column=user_column,
        max_history_items=max_history_items,
    ).items():
        rows = X.iloc[selected_positions].copy()
        item_ids = (
            tuple(rows[item_column].tolist())
            if item_column is not None
            else ()
        )
        histories[user_id] = UserHistory(
            user_id=user_id,
            rows=rows,
            item_ids=item_ids,
        )
    return histories
