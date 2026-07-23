from __future__ import annotations

from pathlib import Path

from beyond_click_sim.scorers import LLMPreferenceYesNoScorer
from beyond_click_sim.tasks import Task
from runners.in_distribution.preference_prediction.methods._grouped_llm_yes_no import (
    run_method as run_grouped_yes_no_method,
)
from runners.in_distribution.preference_prediction.methods.llm_listwise_ranking import (
    QWEN3_8B_MAX_WORKERS as LISTWISE_QWEN3_8B_MAX_WORKERS,
    QWEN3_8B_MODEL,
    QWEN36_27B_MAX_WORKERS as LISTWISE_QWEN36_27B_MAX_WORKERS,
    QWEN36_27B_MODEL,
    run_method as run_listwise_method,
)
from runners.in_distribution.preference_prediction.methods.llm_yes_no import (
    CLIENT_NAME,
    QWEN3_8B_MAX_WORKERS,
    QWEN36_27B_MAX_WORKERS,
    QWEN_EXTRA_BODY,
    TARGET_DESCRIPTIONS,
    _serving_metadata,
    _source_metadata,
)


SMOKE_CANDIDATE_GROUPS = 5
PROMPT_FAMILY = "openp5_style"
OLLAMA_CLIENT = "ollama_local"
OLLAMA_MODEL = "qwen3:30b-a3b-instruct-2507-q4_K_M"
OLLAMA_MAX_WORKERS = 1
SERVER_SMOKE_CANDIDATE_GROUPS = 25


def run_yes_no_ollama_qwen3_30b_a3b_smoke5(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    """Smoke-test the neutral OpenP5-style preference yes/no prompt."""

    dataset_name = str(task.manifest["dataset"])
    try:
        target_description = TARGET_DESCRIPTIONS[dataset_name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported preference-prediction dataset: {dataset_name!r}"
        ) from error

    return run_grouped_yes_no_method(
        task,
        output_dir,
        method_name=(
            "llm_preference_yes_no_openp5_style_ollama_"
            "qwen3_30b_a3b_smoke5"
        ),
        client_name=OLLAMA_CLIENT,
        model=OLLAMA_MODEL,
        max_candidate_groups=SMOKE_CANDIDATE_GROUPS,
        max_workers=OLLAMA_MAX_WORKERS,
        scorer_class=LLMPreferenceYesNoScorer,
        scorer_kwargs={"target_description": target_description},
        prompt_family=PROMPT_FAMILY,
    )


def run_listwise_ollama_qwen3_30b_a3b_smoke5(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    """Smoke-test the neutral OpenP5-style preference listwise prompt."""

    return run_listwise_method(
        task,
        output_dir,
        method_name=(
            "llm_preference_listwise_ranking_openp5_style_"
            "ollama_qwen3_30b_a3b_smoke5"
        ),
        client_name=OLLAMA_CLIENT,
        model=OLLAMA_MODEL,
        max_candidate_groups=SMOKE_CANDIDATE_GROUPS,
        max_workers=OLLAMA_MAX_WORKERS,
        prompt_family=PROMPT_FAMILY,
    )


def run_yes_no_litellm_qwen3_8b_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_yes_no_litellm_qwen(
        task,
        output_dir,
        model_slug="qwen3_8b",
        model=QWEN3_8B_MODEL,
        max_candidate_groups=SERVER_SMOKE_CANDIDATE_GROUPS,
        max_workers=QWEN3_8B_MAX_WORKERS,
        suffix="smoke",
    )


def run_yes_no_litellm_qwen3_8b_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_yes_no_litellm_qwen(
        task,
        output_dir,
        model_slug="qwen3_8b",
        model=QWEN3_8B_MODEL,
        max_candidate_groups=None,
        max_workers=QWEN3_8B_MAX_WORKERS,
        suffix="full",
    )


def run_yes_no_litellm_qwen36_27b_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_yes_no_litellm_qwen(
        task,
        output_dir,
        model_slug="qwen36_27b",
        model=QWEN36_27B_MODEL,
        max_candidate_groups=SERVER_SMOKE_CANDIDATE_GROUPS,
        max_workers=QWEN36_27B_MAX_WORKERS,
        suffix="smoke",
    )


def run_yes_no_litellm_qwen36_27b_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_yes_no_litellm_qwen(
        task,
        output_dir,
        model_slug="qwen36_27b",
        model=QWEN36_27B_MODEL,
        max_candidate_groups=None,
        max_workers=QWEN36_27B_MAX_WORKERS,
        suffix="full",
    )


def run_listwise_litellm_qwen3_8b_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_listwise_litellm_qwen(
        task,
        output_dir,
        model_slug="qwen3_8b",
        model=QWEN3_8B_MODEL,
        max_candidate_groups=SERVER_SMOKE_CANDIDATE_GROUPS,
        max_workers=LISTWISE_QWEN3_8B_MAX_WORKERS,
        suffix="smoke",
    )


def run_listwise_litellm_qwen3_8b_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_listwise_litellm_qwen(
        task,
        output_dir,
        model_slug="qwen3_8b",
        model=QWEN3_8B_MODEL,
        max_candidate_groups=None,
        max_workers=LISTWISE_QWEN3_8B_MAX_WORKERS,
        suffix="full",
    )


def run_listwise_litellm_qwen36_27b_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_listwise_litellm_qwen(
        task,
        output_dir,
        model_slug="qwen36_27b",
        model=QWEN36_27B_MODEL,
        max_candidate_groups=SERVER_SMOKE_CANDIDATE_GROUPS,
        max_workers=LISTWISE_QWEN36_27B_MAX_WORKERS,
        suffix="smoke",
    )


def run_listwise_litellm_qwen36_27b_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_listwise_litellm_qwen(
        task,
        output_dir,
        model_slug="qwen36_27b",
        model=QWEN36_27B_MODEL,
        max_candidate_groups=None,
        max_workers=LISTWISE_QWEN36_27B_MAX_WORKERS,
        suffix="full",
    )


def _target_description(task: Task) -> str:
    dataset_name = str(task.manifest["dataset"])
    try:
        return TARGET_DESCRIPTIONS[dataset_name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported preference-prediction dataset: {dataset_name!r}"
        ) from error


def _run_yes_no_litellm_qwen(
    task: Task,
    output_dir: Path,
    *,
    model_slug: str,
    model: str,
    max_candidate_groups: int | None,
    max_workers: int,
    suffix: str,
) -> dict[str, object]:
    return run_grouped_yes_no_method(
        task,
        output_dir,
        method_name=(
            f"llm_preference_yes_no_openp5_style_litellm_"
            f"{model_slug}_{suffix}"
        ),
        client_name=CLIENT_NAME,
        model=model,
        max_candidate_groups=max_candidate_groups,
        max_workers=max_workers,
        scorer_class=LLMPreferenceYesNoScorer,
        scorer_kwargs={"target_description": _target_description(task)},
        extra_body=QWEN_EXTRA_BODY,
        serving_metadata=_serving_metadata(),
        source_metadata=_source_metadata(),
        prompt_family=PROMPT_FAMILY,
    )


def _run_listwise_litellm_qwen(
    task: Task,
    output_dir: Path,
    *,
    model_slug: str,
    model: str,
    max_candidate_groups: int | None,
    max_workers: int,
    suffix: str,
) -> dict[str, object]:
    return run_listwise_method(
        task,
        output_dir,
        method_name=(
            f"llm_preference_listwise_ranking_openp5_style_litellm_"
            f"{model_slug}_{suffix}"
        ),
        client_name=CLIENT_NAME,
        model=model,
        max_candidate_groups=max_candidate_groups,
        max_workers=max_workers,
        use_item_stats=False,
        extra_body=QWEN_EXTRA_BODY,
        serving_metadata=_serving_metadata(),
        source_metadata=_source_metadata(),
        prompt_family=PROMPT_FAMILY,
    )
