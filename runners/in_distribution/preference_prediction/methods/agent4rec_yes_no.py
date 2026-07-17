from __future__ import annotations

from pathlib import Path

from beyond_click_sim.scorers import Agent4RecPreferenceYesNoScorer
from beyond_click_sim.tasks import Task
from runners.in_distribution.item_summaries import (
    canonical_agent4rec_summary_usage,
)
from runners.in_distribution.preference_prediction.methods import (
    _grouped_agent4rec_yes_no,
)
from runners.in_distribution.preference_prediction.methods.llm_yes_no import (
    CLIENT_NAME,
    QWEN_EXTRA_BODY,
    QWEN3_8B_MAX_WORKERS,
    QWEN3_8B_MODEL,
    QWEN36_27B_MAX_WORKERS,
    QWEN36_27B_MODEL,
    SMOKE_CANDIDATE_GROUPS,
    TARGET_DESCRIPTIONS,
    _serving_metadata,
    _source_metadata,
)


QWEN3_8B_METHOD_NAME = "agent4rec_preference_yes_no_litellm_qwen3_8b_traits"
QWEN36_27B_METHOD_NAME = (
    "agent4rec_preference_yes_no_litellm_qwen36_27b_traits"
)
QWEN3_8B_TRAITS_TASTE_METHOD_NAME = (
    "agent4rec_preference_yes_no_litellm_qwen3_8b_traits_taste_"
    "gpt4o_mini_candidate_summary"
)

DATASET_CANDIDATE_COLUMNS = {
    "ml-1m": ("item_title", "item_genres"),
    "steam": ("item_title", "item_genres_json", "item_tags_json"),
}
DATASET_COLUMN_LABELS = {
    "ml-1m": {
        "item_title": "movie title",
        "item_genres": "genres",
    },
    "steam": {
        "item_title": "game title",
        "item_genres_json": "genres",
        "item_tags_json": "tags",
    },
}


def run_qwen3_8b_smoke(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{QWEN3_8B_METHOD_NAME}_smoke",
        model=QWEN3_8B_MODEL,
        max_candidate_groups=SMOKE_CANDIDATE_GROUPS,
        max_workers=QWEN3_8B_MAX_WORKERS,
    )


def run_qwen3_8b_full(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{QWEN3_8B_METHOD_NAME}_full",
        model=QWEN3_8B_MODEL,
        max_candidate_groups=None,
        max_workers=QWEN3_8B_MAX_WORKERS,
    )


def run_qwen36_27b_smoke(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{QWEN36_27B_METHOD_NAME}_smoke",
        model=QWEN36_27B_MODEL,
        max_candidate_groups=SMOKE_CANDIDATE_GROUPS,
        max_workers=QWEN36_27B_MAX_WORKERS,
    )


def run_qwen36_27b_full(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{QWEN36_27B_METHOD_NAME}_full",
        model=QWEN36_27B_MODEL,
        max_candidate_groups=None,
        max_workers=QWEN36_27B_MAX_WORKERS,
    )


def run_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary(
        task,
        output_dir,
        max_candidate_groups=SMOKE_CANDIDATE_GROUPS,
        suffix="smoke",
    )


def run_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary(
        task,
        output_dir,
        max_candidate_groups=None,
        suffix="full",
    )


def _run_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary(
    task: Task,
    output_dir: Path,
    *,
    max_candidate_groups: int | None,
    suffix: str,
) -> dict[str, object]:
    dataset_name = str(task.manifest["dataset"])
    try:
        target_description = TARGET_DESCRIPTIONS[dataset_name]
        candidate_description_columns = DATASET_CANDIDATE_COLUMNS[dataset_name]
        column_labels = DATASET_COLUMN_LABELS[dataset_name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported Agent4Rec preference dataset: {dataset_name!r}"
        ) from error

    return _grouped_agent4rec_yes_no.run_method(
        task,
        output_dir,
        method_name=f"{QWEN3_8B_TRAITS_TASTE_METHOD_NAME}_{suffix}",
        client_name=CLIENT_NAME,
        model=QWEN3_8B_MODEL,
        max_candidate_groups=max_candidate_groups,
        max_workers=QWEN3_8B_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("traits", "taste"),
        taste_client_name=_grouped_agent4rec_yes_no.OPENAI_CLIENT,
        taste_model=_grouped_agent4rec_yes_no.GPT4O_MINI_TASTE_MODEL,
        taste_temperature=_grouped_agent4rec_yes_no.TASTE_TEMPERATURE,
        taste_max_tokens=_grouped_agent4rec_yes_no.TASTE_MAX_TOKENS,
        scorer_class=Agent4RecPreferenceYesNoScorer,
        scorer_kwargs={"target_description": target_description},
        candidate_description_columns=candidate_description_columns,
        column_labels=column_labels,
        parser_contract="agent4rec_labeled_id_entity_preference_reason",
        serving_metadata=_serving_metadata(),
        source_metadata=_source_metadata(),
        summary_usage="candidate",
    )


def run_method(
    task: Task,
    output_dir: Path,
    *,
    method_name: str,
    model: str,
    max_candidate_groups: int | None,
    max_workers: int,
) -> dict[str, object]:
    dataset_name = str(task.manifest["dataset"])
    try:
        target_description = TARGET_DESCRIPTIONS[dataset_name]
        candidate_description_columns = DATASET_CANDIDATE_COLUMNS[dataset_name]
        column_labels = DATASET_COLUMN_LABELS[dataset_name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported Agent4Rec preference dataset: {dataset_name!r}"
        ) from error

    return _grouped_agent4rec_yes_no.run_method(
        task,
        output_dir,
        method_name=method_name,
        client_name=CLIENT_NAME,
        model=model,
        max_candidate_groups=max_candidate_groups,
        max_workers=max_workers,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("traits",),
        scorer_class=Agent4RecPreferenceYesNoScorer,
        scorer_kwargs={"target_description": target_description},
        candidate_description_columns=candidate_description_columns,
        column_labels=column_labels,
        parser_contract="agent4rec_labeled_id_entity_preference_reason",
        serving_metadata=_serving_metadata(),
        source_metadata=_source_metadata(),
        summary_usage=canonical_agent4rec_summary_usage(task),
    )
