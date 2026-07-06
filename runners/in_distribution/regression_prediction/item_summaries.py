from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from runners.in_distribution.regression_prediction.task_builders import repo_root


ITEM_SUMMARY_COLUMN = "item_summary"
ITEM_SUMMARY_COLUMN_LABEL = "summary"


def agent4rec_ml1m_movies_augmentation_path() -> Path:
    return (
        repo_root().parent
        / "Agent4Rec"
        / "datasets"
        / "ml-1m"
        / "raw_data"
        / "movies_augmentation.csv"
    )


def maybe_add_item_summary_prompt_columns(
    dataset_name: str,
    prompt_columns: dict[str, tuple[str, ...]],
    *,
    use_item_summaries: bool,
) -> dict[str, tuple[str, ...]]:
    if not use_item_summaries:
        return prompt_columns
    _require_ml1m(dataset_name)
    return {
        "history_description_columns": (
            *prompt_columns["history_description_columns"],
            ITEM_SUMMARY_COLUMN,
        ),
        "candidate_description_columns": (
            *prompt_columns["candidate_description_columns"],
            ITEM_SUMMARY_COLUMN,
        ),
    }


def add_ml1m_item_summaries(
    *,
    dataset_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    use_item_summaries: bool,
    source_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not use_item_summaries:
        return X_train, X_test, {"uses_item_summaries": False}
    _require_ml1m(dataset_name)

    source = (
        agent4rec_ml1m_movies_augmentation_path()
        if source_path is None
        else source_path.expanduser().resolve()
    )
    summaries = load_ml1m_item_summaries(source)
    train_with_summary = _merge_item_summaries(X_train, summaries)
    test_with_summary = _merge_item_summaries(X_test, summaries)
    metadata = {
        "uses_item_summaries": True,
        "source_path": str(source),
        "summary_column": ITEM_SUMMARY_COLUMN,
        "train_rows": int(len(train_with_summary)),
        "test_rows": int(len(test_with_summary)),
        "train_missing_summaries": int(
            train_with_summary[ITEM_SUMMARY_COLUMN].isna().sum()
        ),
        "test_missing_summaries": int(
            test_with_summary[ITEM_SUMMARY_COLUMN].isna().sum()
        ),
    }
    return train_with_summary, test_with_summary, metadata


def load_ml1m_item_summaries(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            "Agent4Rec MovieLens summary file is missing: "
            f"{path}. Expected raw_data/movies_augmentation.csv."
        )
    rows = pd.read_csv(
        path,
        usecols=["movie_id", "summary"],
        dtype={"movie_id": "string", "summary": "string"},
    )
    rows = rows.rename(
        columns={
            "movie_id": "_item_summary_item_id",
            "summary": ITEM_SUMMARY_COLUMN,
        }
    )
    rows[ITEM_SUMMARY_COLUMN] = rows[ITEM_SUMMARY_COLUMN].str.strip()
    rows = rows.dropna(subset=["_item_summary_item_id", ITEM_SUMMARY_COLUMN])
    rows = rows[rows[ITEM_SUMMARY_COLUMN] != ""]
    return rows.drop_duplicates(subset=["_item_summary_item_id"], keep="first")


def _merge_item_summaries(
    frame: pd.DataFrame,
    summaries: pd.DataFrame,
) -> pd.DataFrame:
    if "item_id" not in frame.columns:
        raise ValueError("Item summaries require an item_id column")
    frame_without_old_summary = frame.drop(
        columns=[ITEM_SUMMARY_COLUMN],
        errors="ignore",
    ).copy()
    frame_without_old_summary["_item_summary_item_id"] = frame_without_old_summary[
        "item_id"
    ].astype("string")
    merged = frame_without_old_summary.merge(
        summaries,
        on="_item_summary_item_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    return merged.drop(columns=["_item_summary_item_id"])


def _require_ml1m(dataset_name: str) -> None:
    if dataset_name != "ml-1m":
        raise ValueError(f"Item summaries are configured only for ml-1m, got {dataset_name!r}")
