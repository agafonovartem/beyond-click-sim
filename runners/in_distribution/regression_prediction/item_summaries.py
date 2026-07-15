from __future__ import annotations

from typing import Literal

from beyond_click_sim.tasks import Task


ITEM_SUMMARY_COLUMN = "item_summary"
ITEM_SUMMARY_COLUMN_LABEL = "summary"

SummaryVisibility = Literal["none", "history", "candidate", "both"]
Agent4RecSummaryUsage = Literal["none", "profile", "candidate", "both"]


def resolve_item_summary_visibility(
    summary_visibility: SummaryVisibility,
) -> dict[str, bool]:
    if summary_visibility not in {"none", "history", "candidate", "both"}:
        raise ValueError(f"Unsupported summary_visibility: {summary_visibility!r}")
    history = summary_visibility in {"history", "both"}
    candidate = summary_visibility in {"candidate", "both"}
    return {
        "history": history,
        "candidate": candidate,
        "any": history or candidate,
    }


def resolve_agent4rec_summary_usage(
    summary_usage: Agent4RecSummaryUsage,
) -> dict[str, bool]:
    if summary_usage not in {"none", "profile", "candidate", "both"}:
        raise ValueError(f"Unsupported summary_usage: {summary_usage!r}")
    profile = summary_usage in {"profile", "both"}
    candidate = summary_usage in {"candidate", "both"}
    return {
        "profile": profile,
        "candidate": candidate,
        "any": profile or candidate,
    }


def maybe_add_item_summary_prompt_columns(
    dataset_name: str,
    prompt_columns: dict[str, tuple[str, ...]],
    *,
    history_item_summaries: bool,
    candidate_item_summaries: bool,
) -> dict[str, tuple[str, ...]]:
    if not history_item_summaries and not candidate_item_summaries:
        return prompt_columns
    _require_ml1m(dataset_name)
    return {
        "history_description_columns": (
            *prompt_columns["history_description_columns"],
            *((ITEM_SUMMARY_COLUMN,) if history_item_summaries else ()),
        ),
        "candidate_description_columns": (
            *prompt_columns["candidate_description_columns"],
            *((ITEM_SUMMARY_COLUMN,) if candidate_item_summaries else ()),
        ),
    }


def task_item_summary_metadata(
    task: Task,
    *,
    history: bool = False,
    profile: bool = False,
    candidate: bool = False,
) -> dict[str, object]:
    requested = history or profile or candidate
    available = all(
        ITEM_SUMMARY_COLUMN in frame.columns
        for frame in (task.train, task.val, task.test)
    )
    if requested and not available:
        raise ValueError(
            "Item summaries were requested but the task has no item_summary column. "
            "Rebuild canonical MovieLens data with movie summary enrichment enabled."
        )
    if requested:
        missing_by_split = {
            split_name: int(frame[ITEM_SUMMARY_COLUMN].isna().sum())
            for split_name, frame in (
                ("train", task.train),
                ("val", task.val),
                ("test", task.test),
            )
        }
        if any(missing_by_split.values()):
            raise ValueError(
                "Item summaries were requested but task rows contain missing values: "
                f"{missing_by_split}"
            )

    canonical_enrichment = task.manifest.get("item_enrichment")
    if requested:
        movie_summaries = (
            canonical_enrichment.get("movie_summaries")
            if isinstance(canonical_enrichment, dict)
            else None
        )
        if (
            not isinstance(movie_summaries, dict)
            or movie_summaries.get("enabled") is not True
            or movie_summaries.get("task_column") != ITEM_SUMMARY_COLUMN
            or not movie_summaries.get("source_sha256")
        ):
            raise ValueError(
                "Item summaries were requested but the task lacks valid canonical "
                "movie-summary provenance"
            )
    return {
        "uses_item_summaries": requested,
        "summary_column": ITEM_SUMMARY_COLUMN if requested else None,
        "history_item_summaries": history,
        "profile_item_summaries": profile,
        "candidate_item_summaries": candidate,
        "canonical_enrichment": canonical_enrichment,
    }


def _require_ml1m(dataset_name: str) -> None:
    if dataset_name != "ml-1m":
        raise ValueError(
            f"Item summaries are configured only for ml-1m, got {dataset_name!r}"
        )
