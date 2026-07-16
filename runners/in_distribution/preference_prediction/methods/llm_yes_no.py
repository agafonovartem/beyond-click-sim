from __future__ import annotations

import os
from pathlib import Path

from beyond_click_sim.scorers import LLMPreferenceYesNoScorer
from beyond_click_sim.tasks import Task
from runners.in_distribution.item_summaries import SummaryVisibility
from runners.in_distribution.preference_prediction.methods._grouped_llm_yes_no import (
    run_method as run_grouped_yes_no_method,
)


CLIENT_NAME = "litellm_local"
QWEN3_8B_METHOD_NAME = "llm_preference_yes_no_litellm_qwen3_8b"
QWEN3_8B_MODEL = "Qwen/Qwen3-8B"
QWEN3_8B_MAX_WORKERS = 128
QWEN36_27B_METHOD_NAME = "llm_preference_yes_no_litellm_qwen36_27b"
QWEN36_27B_MODEL = "Qwen/Qwen3.6-27B"
QWEN36_27B_MAX_WORKERS = 64
SMOKE_CANDIDATE_GROUPS = 25
QWEN_EXTRA_BODY: dict[str, object] = {
    "chat_template_kwargs": {"enable_thinking": False}
}

TARGET_DESCRIPTIONS = {
    "ml-1m": "The user would rate the candidate movie at least 4 out of 5.",
    "steam": "The user would play the candidate game for at least 120 minutes in total.",
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


def run_qwen3_8b_summary_full(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{QWEN3_8B_METHOD_NAME}_summary_full",
        model=QWEN3_8B_MODEL,
        max_candidate_groups=None,
        max_workers=QWEN3_8B_MAX_WORKERS,
        summary_visibility="both",
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


def run_qwen36_27b_summary_full(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{QWEN36_27B_METHOD_NAME}_summary_full",
        model=QWEN36_27B_MODEL,
        max_candidate_groups=None,
        max_workers=QWEN36_27B_MAX_WORKERS,
        summary_visibility="both",
    )


def run_method(
    task: Task,
    output_dir: Path,
    *,
    method_name: str,
    model: str = QWEN3_8B_MODEL,
    max_candidate_groups: int | None,
    max_workers: int = QWEN3_8B_MAX_WORKERS,
    summary_visibility: SummaryVisibility = "none",
) -> dict[str, object]:
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
        method_name=method_name,
        client_name=CLIENT_NAME,
        model=model,
        max_candidate_groups=max_candidate_groups,
        max_workers=max_workers,
        summary_visibility=summary_visibility,
        extra_body=QWEN_EXTRA_BODY,
        scorer_class=LLMPreferenceYesNoScorer,
        scorer_kwargs={"target_description": target_description},
        serving_metadata=_serving_metadata(),
        source_metadata=_source_metadata(),
    )


def _serving_metadata() -> dict[str, object]:
    return {
        "backend": "litellm_proxy_over_vllm",
        "litellm_base_url": os.environ.get(
            "BEYOND_CLICK_SIM_LITELLM_LOCAL_BASE_URL",
            "http://127.0.0.1:8080/v1",
        ),
        "litellm_version": os.environ.get(
            "BEYOND_CLICK_SIM_LITELLM_VERSION",
            "unknown",
        ),
        "routing_strategy": os.environ.get(
            "BEYOND_CLICK_SIM_LITELLM_ROUTING_STRATEGY",
            "simple-shuffle",
        ),
        "vllm_version": os.environ.get(
            "BEYOND_CLICK_SIM_VLLM_VERSION",
            "unknown",
        ),
        "vllm_replicas": int(
            os.environ.get("BEYOND_CLICK_SIM_VLLM_REPLICAS", "4")
        ),
        "vllm_ports": [8000, 8001, 8002, 8003],
        "tensor_parallel_size_per_replica": 1,
        "max_model_len": int(
            os.environ.get("BEYOND_CLICK_SIM_VLLM_MAX_MODEL_LEN", "4096")
        ),
        "gpu_memory_utilization": float(
            os.environ.get(
                "BEYOND_CLICK_SIM_VLLM_GPU_MEMORY_UTILIZATION",
                "0.90",
            )
        ),
        "model_revision": os.environ.get(
            "BEYOND_CLICK_SIM_MODEL_REVISION",
            "unknown",
        ),
        "thinking_enabled": False,
    }


def _source_metadata() -> dict[str, object]:
    return {
        "base_git_commit": os.environ.get(
            "BEYOND_CLICK_SIM_SOURCE_BASE_GIT_COMMIT"
        ),
        "snapshot_sha256": os.environ.get(
            "BEYOND_CLICK_SIM_SOURCE_SNAPSHOT_SHA256"
        ),
        "diff_sha256": os.environ.get(
            "BEYOND_CLICK_SIM_SOURCE_DIFF_SHA256"
        ),
    }
