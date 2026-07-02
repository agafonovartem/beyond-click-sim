from __future__ import annotations

from beyond_click_sim.tasks import (
    PREFIXED_ITEM_RATING_COUNT_COLUMN,
    PREFIXED_ITEM_RATING_MEAN_COLUMN,
    PREFIXED_ITEM_RATING_STATS_COLUMNS,
)


ITEM_RATING_COLUMN_LABELS = {
    "ml-1m": {
        "rating": "user rating",
        PREFIXED_ITEM_RATING_MEAN_COLUMN: "average rating",
        PREFIXED_ITEM_RATING_COUNT_COLUMN: "number of prior reviews",
    },
    "steam": {
        "playtime_forever": "user playtime minutes",
        PREFIXED_ITEM_RATING_MEAN_COLUMN: "average prior playtime minutes",
        PREFIXED_ITEM_RATING_COUNT_COLUMN: "number of prior interactions",
    },
}


def maybe_add_item_rating_prompt_columns(
    dataset_name: str,
    prompt_columns: dict[str, tuple[str, ...]],
    *,
    use_item_stats: bool,
) -> dict[str, tuple[str, ...]]:
    """Append item rating stats to prompt columns when explicitly requested."""

    if not use_item_stats:
        return prompt_columns
    _require_item_rating_prompt_config(dataset_name)
    return {
        "history_description_columns": (
            *prompt_columns["history_description_columns"],
            *PREFIXED_ITEM_RATING_STATS_COLUMNS,
        ),
        "candidate_description_columns": (
            *prompt_columns["candidate_description_columns"],
            *PREFIXED_ITEM_RATING_STATS_COLUMNS,
        ),
    }


def item_rating_column_labels(
    dataset_name: str,
    *,
    use_item_stats: bool,
) -> dict[str, str]:
    """Return human-readable labels only for item-stats prompt variants."""

    if not use_item_stats:
        return {}
    _require_item_rating_prompt_config(dataset_name)
    return ITEM_RATING_COLUMN_LABELS[dataset_name]


def _require_item_rating_prompt_config(dataset_name: str) -> None:
    if dataset_name not in ITEM_RATING_COLUMN_LABELS:
        raise ValueError(f"No item rating prompt config for dataset: {dataset_name}")
