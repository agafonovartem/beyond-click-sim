from __future__ import annotations

from pathlib import Path

import pandas as pd

from runners.in_distribution.regression_prediction.item_summaries import (
    ITEM_SUMMARY_COLUMN,
    add_ml1m_item_summaries,
    load_ml1m_item_summaries,
    maybe_add_item_summary_prompt_columns,
    resolve_item_summary_visibility,
)


def test_load_ml1m_item_summaries_uses_raw_movie_id_as_item_id(tmp_path: Path) -> None:
    source = tmp_path / "movies_augmentation.csv"
    source.write_text(
        "movie_id,title,genres,rating,summary\n"
        "1,Toy Story (1995),Animation,9.5,Toys plan a rescue.\n"
        "2,Jumanji (1995),Adventure,8.5,A magical board game.\n",
        encoding="utf-8",
    )

    summaries = load_ml1m_item_summaries(source)

    assert summaries["_item_summary_item_id"].tolist() == ["1", "2"]
    assert summaries[ITEM_SUMMARY_COLUMN].tolist() == [
        "Toys plan a rescue.",
        "A magical board game.",
    ]


def test_add_ml1m_item_summaries_merges_train_and_test(tmp_path: Path) -> None:
    source = tmp_path / "movies_augmentation.csv"
    source.write_text(
        "movie_id,title,genres,rating,summary\n"
        "1,Toy Story (1995),Animation,9.5,Toys plan a rescue.\n"
        "2,Jumanji (1995),Adventure,8.5,A magical board game.\n",
        encoding="utf-8",
    )
    X_train = pd.DataFrame({"item_id": ["1", "missing"], "value": [1, 2]})
    X_test = pd.DataFrame({"item_id": ["2"], "value": [3]})

    train, test, metadata = add_ml1m_item_summaries(
        dataset_name="ml-1m",
        X_train=X_train,
        X_test=X_test,
        use_item_summaries=True,
        source_path=source,
    )

    assert train[ITEM_SUMMARY_COLUMN].iloc[0] == "Toys plan a rescue."
    assert pd.isna(train[ITEM_SUMMARY_COLUMN].iloc[1])
    assert test[ITEM_SUMMARY_COLUMN].tolist() == ["A magical board game."]
    assert metadata["history_item_summaries"] is True
    assert metadata["candidate_item_summaries"] is True
    assert metadata["train_missing_summaries"] == 1
    assert metadata["test_missing_summaries"] == 0


def test_item_summary_prompt_columns_can_be_split() -> None:
    columns = {
        "history_description_columns": ("item_title", "rating"),
        "candidate_description_columns": ("item_title",),
    }

    history_only = maybe_add_item_summary_prompt_columns(
        "ml-1m",
        columns,
        history_item_summaries=True,
        candidate_item_summaries=False,
    )
    candidate_only = maybe_add_item_summary_prompt_columns(
        "ml-1m",
        columns,
        history_item_summaries=False,
        candidate_item_summaries=True,
    )

    assert history_only["history_description_columns"] == (
        "item_title",
        "rating",
        ITEM_SUMMARY_COLUMN,
    )
    assert history_only["candidate_description_columns"] == ("item_title",)
    assert candidate_only["history_description_columns"] == ("item_title", "rating")
    assert candidate_only["candidate_description_columns"] == (
        "item_title",
        ITEM_SUMMARY_COLUMN,
    )


def test_resolve_item_summary_visibility_keeps_legacy_bool_as_both() -> None:
    assert resolve_item_summary_visibility(use_item_summaries=True) == {
        "history": True,
        "candidate": True,
        "any": True,
    }
    assert resolve_item_summary_visibility(
        history_item_summaries=True,
        candidate_item_summaries=False,
    ) == {
        "history": True,
        "candidate": False,
        "any": True,
    }
