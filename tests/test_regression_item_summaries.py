from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

from beyond_click_sim.tasks import Task, TaskSchema

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.in_distribution.item_summaries import (
    ITEM_SUMMARY_COLUMN,
    canonical_agent4rec_summary_usage,
    maybe_add_item_summary_prompt_columns,
    resolve_agent4rec_summary_usage,
    resolve_item_summary_visibility,
    task_item_summary_metadata,
)


def test_canonical_agent4rec_uses_candidate_summaries_when_enriched() -> None:
    assert canonical_agent4rec_summary_usage(_summary_task()) == "candidate"


def test_canonical_agent4rec_uses_none_without_enrichment() -> None:
    task = _summary_task()
    task.manifest["item_enrichment"] = {"movie_summaries": {"enabled": False}}

    assert canonical_agent4rec_summary_usage(task) == "none"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("none", {"history": False, "candidate": False, "any": False}),
        ("history", {"history": True, "candidate": False, "any": True}),
        ("candidate", {"history": False, "candidate": True, "any": True}),
        ("both", {"history": True, "candidate": True, "any": True}),
    ],
)
def test_resolve_item_summary_visibility(value: str, expected: dict[str, bool]) -> None:
    assert resolve_item_summary_visibility(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("none", {"profile": False, "candidate": False, "any": False}),
        ("profile", {"profile": True, "candidate": False, "any": True}),
        ("candidate", {"profile": False, "candidate": True, "any": True}),
        ("both", {"profile": True, "candidate": True, "any": True}),
    ],
)
def test_resolve_agent4rec_summary_usage(value: str, expected: dict[str, bool]) -> None:
    assert resolve_agent4rec_summary_usage(value) == expected


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


def test_task_item_summary_metadata_records_visibility_and_provenance() -> None:
    task = _summary_task()

    metadata = task_item_summary_metadata(
        task,
        profile=True,
        candidate=True,
    )

    assert metadata == {
        "uses_item_summaries": True,
        "summary_column": "item_summary",
        "history_item_summaries": False,
        "profile_item_summaries": True,
        "candidate_item_summaries": True,
        "canonical_enrichment": {
            "movie_summaries": {
                "enabled": True,
                "canonical_column": "summary",
                "task_column": "item_summary",
                "source_sha256": "fixture-sha256",
            }
        },
    }


def test_task_item_summary_metadata_requires_canonical_task_column() -> None:
    task = _summary_task()
    for frame in (task.train, task.val, task.test):
        frame.drop(columns=[ITEM_SUMMARY_COLUMN], inplace=True)

    with pytest.raises(ValueError, match="Rebuild canonical MovieLens data"):
        task_item_summary_metadata(task, candidate=True)


def test_task_item_summary_metadata_rejects_missing_values() -> None:
    task = _summary_task()
    task.test.loc[0, ITEM_SUMMARY_COLUMN] = pd.NA

    with pytest.raises(ValueError, match="missing values"):
        task_item_summary_metadata(task, history=True)


def test_task_item_summary_metadata_requires_canonical_provenance() -> None:
    task = _summary_task()
    task.manifest.clear()

    with pytest.raises(ValueError, match="canonical movie-summary provenance"):
        task_item_summary_metadata(task, candidate=True)


def _summary_task() -> Task:
    frame = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_id": ["i1"],
            ITEM_SUMMARY_COLUMN: ["A movie summary."],
            "target": [5],
        }
    )
    return Task(
        name="summary-task",
        train=frame.copy(),
        val=frame.copy(),
        test=frame.copy(),
        schema=TaskSchema(
            target_column="target",
            feature_columns=(ITEM_SUMMARY_COLUMN,),
        ),
        manifest={
            "item_enrichment": {
                "movie_summaries": {
                    "enabled": True,
                    "canonical_column": "summary",
                    "task_column": "item_summary",
                    "source_sha256": "fixture-sha256",
                }
            }
        },
    )
