from __future__ import annotations

from pathlib import Path

import pandas as pd

from beyond_click_sim.llm_clients import make_llm_client
from beyond_click_sim.scorers import (
    LLMInteractionListwiseRankingScorer,
    LLMPreferenceListwiseRankingScorer,
)
from beyond_click_sim.tasks import split_xy
from beyond_click_sim.tasks.cold_start import ColdStartTask
from runners.in_distribution.cold_start.methods._listwise import (
    evaluate_listwise_scorer,
)
from runners.in_distribution.cold_start.methods.llm_yes_no import (
    DATASET_PROMPT_COLUMNS,
    QWEN_EXTRA_BODY,
    VLLM_MAX_WORKERS,
    VLLM_QWEN36_27B_CLIENT,
    VLLM_QWEN36_27B_MODEL,
    VLLM_QWEN36_35B_A3B_CLIENT,
    VLLM_QWEN36_35B_A3B_MODEL,
)
from runners.in_distribution.interaction_prediction.methods.common import (
    limit_candidate_groups,
    task_xy,
)
from runners.in_distribution.interaction_prediction.task_builders import repo_root
from runners.in_distribution.item_summaries import (
    ITEM_SUMMARY_COLUMN,
    ITEM_SUMMARY_COLUMN_LABEL,
    SummaryVisibility,
    maybe_add_item_summary_prompt_columns,
    resolve_item_summary_visibility,
    task_item_summary_metadata,
)
from runners.in_distribution.llm_error_budget import (
    DEFAULT_MAX_ERROR_RATE,
    DEFAULT_MIN_GROUPS_BEFORE_CHECK,
)
from runners.in_distribution.llm_item_stats import (
    item_rating_column_labels,
    maybe_add_item_rating_prompt_columns,
)


VLLM_QWEN36_27B_METHOD_NAME = "llm_preference_listwise_ranking_vllm_qwen36_27b"
VLLM_QWEN36_35B_A3B_METHOD_NAME = "llm_preference_listwise_ranking_vllm_qwen36_35b_a3b"
VLLM_QWEN3_8B_METHOD_NAME = "llm_preference_listwise_ranking_vllm_qwen3_8b"
VLLM_QWEN3_8B_CLIENT = "vllm_local"
VLLM_QWEN3_8B_MODEL = "Qwen/Qwen3-8B"
VLLM_LLAMA33_70B_METHOD_NAME = "llm_preference_listwise_ranking_vllm_llama33_70b"
VLLM_LLAMA33_70B_CLIENT = "vllm_local"
VLLM_LLAMA33_70B_MODEL = "llama-3.3-70b-instruct"
TARGET_DESCRIPTIONS = {
    "ml-1m": "The user would rate the candidate movie at least 4 out of 5.",
}
PREFERENCE_TARGET_SOURCE_COLUMN = "target_like_ge4"
INTERACTION_TARGET_SOURCE_COLUMN = "target_interact"
PREFERENCE_PROTOCOL = "direct_listwise_preference_ranking"
INTERACTION_PROTOCOL = "direct_listwise_interaction_ranking"
MAX_HISTORY_ITEMS = 20
TEMPERATURE = 0.0
MAX_TOKENS = 1024
MAX_LLM_ATTEMPTS = 5


def run_qwen36_27b_smoke(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_smoke",
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=25,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
    )


def run_qwen36_27b_full(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_full",
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=None,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
    )


def run_qwen36_27b_sumboth_smoke(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_sumboth_smoke",
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=25,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        **_sumboth_kwargs(),
    )


def run_qwen36_27b_sumboth_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_sumboth_full",
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=None,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        **_sumboth_kwargs(),
    )


def run_qwen36_27b_item_stats_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_full_variant(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_item_stats_full",
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        use_item_stats=True,
    )


def run_qwen36_27b_sumhistory_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_full_variant(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_sumhistory_full",
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        summary_visibility="history",
    )


def run_qwen36_27b_sumcandidate_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_full_variant(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_sumcandidate_full",
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        summary_visibility="candidate",
    )


def run_qwen36_27b_item_stats_sumhistory_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_full_variant(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_item_stats_sumhistory_full",
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        use_item_stats=True,
        summary_visibility="history",
    )


def run_qwen36_27b_item_stats_sumcandidate_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_full_variant(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_item_stats_sumcandidate_full",
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        use_item_stats=True,
        summary_visibility="candidate",
    )


def run_qwen36_27b_item_stats_sumboth_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_full_variant(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_item_stats_sumboth_full",
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        use_item_stats=True,
        summary_visibility="both",
    )


def run_qwen36_35b_a3b_smoke(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_smoke",
        client_name=VLLM_QWEN36_35B_A3B_CLIENT,
        model=VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=25,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
    )


def run_qwen36_35b_a3b_sumboth_smoke(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_sumboth_smoke",
        client_name=VLLM_QWEN36_35B_A3B_CLIENT,
        model=VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=25,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        **_sumboth_kwargs(),
    )


def run_qwen36_35b_a3b_full(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_full",
        client_name=VLLM_QWEN36_35B_A3B_CLIENT,
        model=VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=None,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
    )


def run_qwen36_35b_a3b_sumboth_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_sumboth_full",
        client_name=VLLM_QWEN36_35B_A3B_CLIENT,
        model=VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=None,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        **_sumboth_kwargs(),
    )


def run_qwen36_35b_a3b_item_stats_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_full_variant(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_item_stats_full",
        client_name=VLLM_QWEN36_35B_A3B_CLIENT,
        model=VLLM_QWEN36_35B_A3B_MODEL,
        use_item_stats=True,
    )


def run_qwen36_35b_a3b_sumhistory_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_full_variant(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_sumhistory_full",
        client_name=VLLM_QWEN36_35B_A3B_CLIENT,
        model=VLLM_QWEN36_35B_A3B_MODEL,
        summary_visibility="history",
    )


def run_qwen36_35b_a3b_sumcandidate_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_full_variant(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_sumcandidate_full",
        client_name=VLLM_QWEN36_35B_A3B_CLIENT,
        model=VLLM_QWEN36_35B_A3B_MODEL,
        summary_visibility="candidate",
    )


def run_qwen36_35b_a3b_item_stats_sumhistory_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_full_variant(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_item_stats_sumhistory_full",
        client_name=VLLM_QWEN36_35B_A3B_CLIENT,
        model=VLLM_QWEN36_35B_A3B_MODEL,
        use_item_stats=True,
        summary_visibility="history",
    )


def run_qwen36_35b_a3b_item_stats_sumcandidate_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_full_variant(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_item_stats_sumcandidate_full",
        client_name=VLLM_QWEN36_35B_A3B_CLIENT,
        model=VLLM_QWEN36_35B_A3B_MODEL,
        use_item_stats=True,
        summary_visibility="candidate",
    )


def run_qwen36_35b_a3b_item_stats_sumboth_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_full_variant(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_item_stats_sumboth_full",
        client_name=VLLM_QWEN36_35B_A3B_CLIENT,
        model=VLLM_QWEN36_35B_A3B_MODEL,
        use_item_stats=True,
        summary_visibility="both",
    )


def run_qwen3_8b_full(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return _run_backbone_ablation_full(
        task,
        output_dir,
        method_base=VLLM_QWEN3_8B_METHOD_NAME,
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        extra_body=QWEN_EXTRA_BODY,
    )


def run_qwen3_8b_item_stats_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_backbone_ablation_full(
        task,
        output_dir,
        method_base=VLLM_QWEN3_8B_METHOD_NAME,
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        extra_body=QWEN_EXTRA_BODY,
        variant="item_stats",
        use_item_stats=True,
    )


def run_qwen3_8b_sumhistory_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_backbone_ablation_full(
        task,
        output_dir,
        method_base=VLLM_QWEN3_8B_METHOD_NAME,
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        extra_body=QWEN_EXTRA_BODY,
        variant="sumhistory",
        summary_visibility="history",
    )


def run_qwen3_8b_sumcandidate_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_backbone_ablation_full(
        task,
        output_dir,
        method_base=VLLM_QWEN3_8B_METHOD_NAME,
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        extra_body=QWEN_EXTRA_BODY,
        variant="sumcandidate",
        summary_visibility="candidate",
    )


def run_qwen3_8b_sumboth_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_backbone_ablation_full(
        task,
        output_dir,
        method_base=VLLM_QWEN3_8B_METHOD_NAME,
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        extra_body=QWEN_EXTRA_BODY,
        variant="sumboth",
        summary_visibility="both",
    )


def run_qwen3_8b_item_stats_sumhistory_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_backbone_ablation_full(
        task,
        output_dir,
        method_base=VLLM_QWEN3_8B_METHOD_NAME,
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        extra_body=QWEN_EXTRA_BODY,
        variant="item_stats_sumhistory",
        use_item_stats=True,
        summary_visibility="history",
    )


def run_qwen3_8b_item_stats_sumcandidate_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_backbone_ablation_full(
        task,
        output_dir,
        method_base=VLLM_QWEN3_8B_METHOD_NAME,
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        extra_body=QWEN_EXTRA_BODY,
        variant="item_stats_sumcandidate",
        use_item_stats=True,
        summary_visibility="candidate",
    )


def run_qwen3_8b_item_stats_sumboth_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_backbone_ablation_full(
        task,
        output_dir,
        method_base=VLLM_QWEN3_8B_METHOD_NAME,
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        extra_body=QWEN_EXTRA_BODY,
        variant="item_stats_sumboth",
        use_item_stats=True,
        summary_visibility="both",
    )


def run_llama33_70b_full(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return _run_backbone_ablation_full(
        task,
        output_dir,
        method_base=VLLM_LLAMA33_70B_METHOD_NAME,
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        extra_body=None,
    )


def run_llama33_70b_item_stats_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_backbone_ablation_full(
        task,
        output_dir,
        method_base=VLLM_LLAMA33_70B_METHOD_NAME,
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        extra_body=None,
        variant="item_stats",
        use_item_stats=True,
    )


def run_llama33_70b_sumhistory_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_backbone_ablation_full(
        task,
        output_dir,
        method_base=VLLM_LLAMA33_70B_METHOD_NAME,
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        extra_body=None,
        variant="sumhistory",
        summary_visibility="history",
    )


def run_llama33_70b_sumcandidate_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_backbone_ablation_full(
        task,
        output_dir,
        method_base=VLLM_LLAMA33_70B_METHOD_NAME,
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        extra_body=None,
        variant="sumcandidate",
        summary_visibility="candidate",
    )


def run_llama33_70b_sumboth_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_backbone_ablation_full(
        task,
        output_dir,
        method_base=VLLM_LLAMA33_70B_METHOD_NAME,
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        extra_body=None,
        variant="sumboth",
        summary_visibility="both",
    )


def run_llama33_70b_item_stats_sumhistory_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_backbone_ablation_full(
        task,
        output_dir,
        method_base=VLLM_LLAMA33_70B_METHOD_NAME,
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        extra_body=None,
        variant="item_stats_sumhistory",
        use_item_stats=True,
        summary_visibility="history",
    )


def run_llama33_70b_item_stats_sumcandidate_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_backbone_ablation_full(
        task,
        output_dir,
        method_base=VLLM_LLAMA33_70B_METHOD_NAME,
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        extra_body=None,
        variant="item_stats_sumcandidate",
        use_item_stats=True,
        summary_visibility="candidate",
    )


def run_llama33_70b_item_stats_sumboth_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_backbone_ablation_full(
        task,
        output_dir,
        method_base=VLLM_LLAMA33_70B_METHOD_NAME,
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        extra_body=None,
        variant="item_stats_sumboth",
        use_item_stats=True,
        summary_visibility="both",
    )


def _run_backbone_ablation_full(
    task: ColdStartTask,
    output_dir: Path,
    *,
    method_base: str,
    client_name: str,
    model: str,
    extra_body: dict | None,
    variant: str | None = None,
    use_item_stats: bool = False,
    summary_visibility: SummaryVisibility = "none",
) -> dict[str, object]:
    method_name = (
        f"{method_base}_{variant}_full"
        if variant is not None
        else f"{method_base}_full"
    )
    return _run_full_variant(
        task,
        output_dir,
        method_name=method_name,
        client_name=client_name,
        model=model,
        extra_body=extra_body,
        use_item_stats=use_item_stats,
        summary_visibility=summary_visibility,
    )


def _run_full_variant(
    task: ColdStartTask,
    output_dir: Path,
    *,
    method_name: str,
    client_name: str,
    model: str,
    extra_body: dict | None = QWEN_EXTRA_BODY,
    use_item_stats: bool = False,
    summary_visibility: SummaryVisibility = "none",
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=method_name,
        client_name=client_name,
        model=model,
        max_candidate_groups=None,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=extra_body,
        use_item_stats=use_item_stats,
        summary_visibility=summary_visibility,
    )


def _sumboth_kwargs() -> dict[str, object]:
    return {"summary_visibility": "both"}


def run_method(
    task: ColdStartTask,
    output_dir: Path,
    *,
    method_name: str,
    client_name: str,
    model: str,
    max_candidate_groups: int | None,
    max_history_items: int = MAX_HISTORY_ITEMS,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    max_llm_attempts: int = MAX_LLM_ATTEMPTS,
    max_workers: int = 1,
    extra_body: dict | None = None,
    use_item_stats: bool = False,
    summary_visibility: SummaryVisibility = "none",
    max_error_rate: float = DEFAULT_MAX_ERROR_RATE,
    min_groups_before_check: int = DEFAULT_MIN_GROUPS_BEFORE_CHECK,
    ranking_target: str = "preference",
) -> dict[str, object]:
    """Run direct listwise ranking on cold-start val/test groups."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = str(task.manifest["dataset"])
    target_source_column = str(task.manifest.get("target_source_column"))
    if ranking_target == "preference":
        expected_source = PREFERENCE_TARGET_SOURCE_COLUMN
        protocol = PREFERENCE_PROTOCOL
    elif ranking_target == "interaction":
        expected_source = INTERACTION_TARGET_SOURCE_COLUMN
        protocol = INTERACTION_PROTOCOL
    else:
        raise ValueError(f"Unsupported ranking_target: {ranking_target!r}")
    if target_source_column != expected_source:
        raise ValueError(
            "Task target mismatch for listwise runner: "
            f"ranking_target={ranking_target!r} requires "
            f"target_source_column={expected_source!r}, got "
            f"{target_source_column!r}."
        )
    try:
        base_prompt_columns = DATASET_PROMPT_COLUMNS[dataset_name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported cold-start listwise dataset: {dataset_name!r}"
        ) from error
    target_description = (
        TARGET_DESCRIPTIONS[dataset_name]
        if ranking_target == "preference"
        else None
    )
    prompt_columns = maybe_add_item_rating_prompt_columns(
        dataset_name,
        base_prompt_columns,
        use_item_stats=use_item_stats,
    )
    resolved_summary_visibility = resolve_item_summary_visibility(summary_visibility)
    prompt_columns = maybe_add_item_summary_prompt_columns(
        dataset_name,
        prompt_columns,
        history_item_summaries=resolved_summary_visibility["history"],
        candidate_item_summaries=resolved_summary_visibility["candidate"],
    )
    column_labels = item_rating_column_labels(
        dataset_name,
        use_item_stats=use_item_stats,
    )
    if resolved_summary_visibility["any"]:
        column_labels = {
            **column_labels,
            ITEM_SUMMARY_COLUMN: ITEM_SUMMARY_COLUMN_LABEL,
        }
    candidate_group_column = task.schema.candidate_group_column
    if candidate_group_column is None:
        raise ValueError("LLM listwise ranking method requires candidate_group_column")

    # Cold-start fit contract: use each cold user's k-item online history only.
    X_history, y_history = split_xy(
        task.online_session_history,
        target_column=task.schema.target_column,
    )
    xy = task_xy(task)
    X_val, y_val = limit_candidate_groups(
        *xy["val"],
        candidate_group_column=candidate_group_column,
        max_candidate_groups=max_candidate_groups,
    )
    X_test, y_test = limit_candidate_groups(
        *xy["test"],
        candidate_group_column=candidate_group_column,
        max_candidate_groups=max_candidate_groups,
    )
    history_user_ids = pd.concat(
        [X_val["user_id"], X_test["user_id"]],
        ignore_index=True,
    ).drop_duplicates()
    item_summary_metadata = task_item_summary_metadata(
        task,
        history=resolved_summary_visibility["history"],
        candidate=resolved_summary_visibility["candidate"],
        extra_frames={"online_session_history": task.online_session_history},
    )

    scorer_kwargs = {
        "client": make_llm_client(client_name),
        "model": model,
        "history_description_columns": prompt_columns["history_description_columns"],
        "candidate_description_columns": prompt_columns["candidate_description_columns"],
        "column_labels": column_labels,
        "json_list_columns": (),
        "max_history_items": max_history_items,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": extra_body,
    }
    if ranking_target == "preference":
        scorer = LLMPreferenceListwiseRankingScorer(
            target_description=target_description,
            **scorer_kwargs,
        )
    else:
        scorer = LLMInteractionListwiseRankingScorer(**scorer_kwargs)
    scorer = scorer.fit(X_history, y_history, history_user_ids=history_user_ids.tolist())

    scorer_manifest = {
        "class": scorer.__class__.__name__,
        "fit_on": "online_session_history",
        "k": task.k,
        "client_name": client_name,
        "model": model,
        "max_history_items": max_history_items,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_columns": prompt_columns,
        "column_labels": column_labels,
        "json_list_columns": [],
        "extra_body": extra_body,
        "uses_item_stats": use_item_stats,
        "summary_visibility": summary_visibility,
        "item_summaries": item_summary_metadata,
        "ranking_target": ranking_target,
    }
    if target_description is not None:
        scorer_manifest["target_description"] = target_description

    return evaluate_listwise_scorer(
        task=task,
        output_dir=output_dir,
        method_name=method_name,
        scorer=scorer,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        candidate_group_column=candidate_group_column,
        max_candidate_groups=max_candidate_groups,
        max_llm_attempts=max_llm_attempts,
        max_workers=max_workers,
        max_error_rate=max_error_rate,
        min_groups_before_check=min_groups_before_check,
        scorer_manifest=scorer_manifest,
        repo_root=repo_root(),
        protocol=protocol,
    )
