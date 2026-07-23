from __future__ import annotations

from pathlib import Path

from beyond_click_sim.tasks import Task
from runners.in_distribution.interaction_prediction.methods.llm_listwise_ranking import (
    QWEN3_8B_MAX_WORKERS as LISTWISE_QWEN3_8B_MAX_WORKERS,
    QWEN36_27B_MAX_WORKERS as LISTWISE_QWEN36_27B_MAX_WORKERS,
    run_method as run_listwise_method,
)
from runners.in_distribution.interaction_prediction.methods.llm_yes_no import (
    LITELLM_CLIENT_NAME,
    LITELLM_QWEN3_8B_MAX_WORKERS,
    LITELLM_QWEN3_8B_MODEL,
    LITELLM_QWEN36_27B_MAX_WORKERS,
    LITELLM_QWEN36_27B_MODEL,
    QWEN_EXTRA_BODY,
    _serving_metadata,
    _source_metadata,
    run_method as run_yes_no_method,
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
    """Smoke-test the neutral OpenP5-style interaction yes/no prompt."""

    return run_yes_no_method(
        task,
        output_dir,
        method_name="llm_yes_no_openp5_style_ollama_qwen3_30b_a3b_smoke5",
        client_name=OLLAMA_CLIENT,
        model=OLLAMA_MODEL,
        max_candidate_groups=SMOKE_CANDIDATE_GROUPS,
        max_workers=OLLAMA_MAX_WORKERS,
        prompt_family=PROMPT_FAMILY,
    )


def run_listwise_ollama_qwen3_30b_a3b_smoke5(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    """Smoke-test the neutral OpenP5-style interaction listwise prompt."""

    return run_listwise_method(
        task,
        output_dir,
        method_name=(
            "llm_listwise_ranking_openp5_style_ollama_"
            "qwen3_30b_a3b_smoke5"
        ),
        client_name=OLLAMA_CLIENT,
        model=OLLAMA_MODEL,
        max_candidate_groups=SMOKE_CANDIDATE_GROUPS,
        max_workers=OLLAMA_MAX_WORKERS,
        prompt_family=PROMPT_FAMILY,
    )


def run_yes_no_litellm_qwen3_8b_with_item_stats_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_yes_no_litellm_qwen(
        task,
        output_dir,
        model_slug="qwen3_8b",
        model=LITELLM_QWEN3_8B_MODEL,
        max_candidate_groups=SERVER_SMOKE_CANDIDATE_GROUPS,
        max_workers=LITELLM_QWEN3_8B_MAX_WORKERS,
        suffix="smoke",
    )


def run_yes_no_litellm_qwen3_8b_with_item_stats_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_yes_no_litellm_qwen(
        task,
        output_dir,
        model_slug="qwen3_8b",
        model=LITELLM_QWEN3_8B_MODEL,
        max_candidate_groups=None,
        max_workers=LITELLM_QWEN3_8B_MAX_WORKERS,
        suffix="full",
    )


def run_yes_no_litellm_qwen36_27b_with_item_stats_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_yes_no_litellm_qwen(
        task,
        output_dir,
        model_slug="qwen36_27b",
        model=LITELLM_QWEN36_27B_MODEL,
        max_candidate_groups=SERVER_SMOKE_CANDIDATE_GROUPS,
        max_workers=LITELLM_QWEN36_27B_MAX_WORKERS,
        suffix="smoke",
    )


def run_yes_no_litellm_qwen36_27b_with_item_stats_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_yes_no_litellm_qwen(
        task,
        output_dir,
        model_slug="qwen36_27b",
        model=LITELLM_QWEN36_27B_MODEL,
        max_candidate_groups=None,
        max_workers=LITELLM_QWEN36_27B_MAX_WORKERS,
        suffix="full",
    )


def run_listwise_litellm_qwen3_8b_with_item_stats_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_listwise_litellm_qwen(
        task,
        output_dir,
        model_slug="qwen3_8b",
        model=LITELLM_QWEN3_8B_MODEL,
        max_candidate_groups=SERVER_SMOKE_CANDIDATE_GROUPS,
        max_workers=LISTWISE_QWEN3_8B_MAX_WORKERS,
        suffix="smoke",
    )


def run_listwise_litellm_qwen3_8b_with_item_stats_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_listwise_litellm_qwen(
        task,
        output_dir,
        model_slug="qwen3_8b",
        model=LITELLM_QWEN3_8B_MODEL,
        max_candidate_groups=None,
        max_workers=LISTWISE_QWEN3_8B_MAX_WORKERS,
        suffix="full",
    )


def run_listwise_litellm_qwen36_27b_with_item_stats_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_listwise_litellm_qwen(
        task,
        output_dir,
        model_slug="qwen36_27b",
        model=LITELLM_QWEN36_27B_MODEL,
        max_candidate_groups=SERVER_SMOKE_CANDIDATE_GROUPS,
        max_workers=LISTWISE_QWEN36_27B_MAX_WORKERS,
        suffix="smoke",
    )


def run_listwise_litellm_qwen36_27b_with_item_stats_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return _run_listwise_litellm_qwen(
        task,
        output_dir,
        model_slug="qwen36_27b",
        model=LITELLM_QWEN36_27B_MODEL,
        max_candidate_groups=None,
        max_workers=LISTWISE_QWEN36_27B_MAX_WORKERS,
        suffix="full",
    )


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
    return run_yes_no_method(
        task,
        output_dir,
        method_name=(
            f"llm_yes_no_openp5_style_litellm_{model_slug}_"
            f"with_item_stats_{suffix}"
        ),
        client_name=LITELLM_CLIENT_NAME,
        model=model,
        max_candidate_groups=max_candidate_groups,
        max_workers=max_workers,
        use_item_stats=True,
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
            f"llm_listwise_ranking_openp5_style_litellm_{model_slug}_"
            f"with_item_stats_{suffix}"
        ),
        client_name=LITELLM_CLIENT_NAME,
        model=model,
        max_candidate_groups=max_candidate_groups,
        max_workers=max_workers,
        use_item_stats=True,
        extra_body=QWEN_EXTRA_BODY,
        serving_metadata=_serving_metadata(),
        source_metadata=_source_metadata(),
        prompt_family=PROMPT_FAMILY,
    )
