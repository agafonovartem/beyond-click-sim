from __future__ import annotations

import pandas as pd


CANONICAL_MOVIE_SUMMARY_COLUMN = "summary"
TASK_MOVIE_SUMMARY_COLUMN = "item_summary"


def item_enrichment_manifest(
    canonical_manifest: dict[str, object],
    items: pd.DataFrame,
) -> dict[str, object] | None:
    """Project canonical item-enrichment provenance into a task manifest."""

    item_enrichment = canonical_manifest.get("item_enrichment")
    if not isinstance(item_enrichment, dict):
        return None
    movie_summaries = item_enrichment.get("movie_summaries")
    if not isinstance(movie_summaries, dict):
        return None
    if not bool(movie_summaries.get("enabled", False)):
        return {"movie_summaries": {"enabled": False}}

    canonical_column = str(movie_summaries.get("column", ""))
    if canonical_column != CANONICAL_MOVIE_SUMMARY_COLUMN:
        raise ValueError(
            "Canonical manifest enables movie summaries with an unsupported "
            f"column: {canonical_column!r}"
        )
    if canonical_column not in items.columns:
        raise ValueError(
            "Canonical manifest enables movie summaries but items.parquet "
            f"does not contain column {canonical_column!r}"
        )
    source = movie_summaries.get("source")
    source_sha256 = source.get("sha256") if isinstance(source, dict) else None
    if not isinstance(source_sha256, str) or not source_sha256:
        raise ValueError(
            "Canonical manifest enables movie summaries but does not record "
            "the source SHA256"
        )
    return {
        "movie_summaries": {
            "enabled": True,
            "canonical_column": canonical_column,
            "task_column": TASK_MOVIE_SUMMARY_COLUMN,
            "source_sha256": source_sha256,
        }
    }
