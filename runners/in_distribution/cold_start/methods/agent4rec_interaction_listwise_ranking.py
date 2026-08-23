from __future__ import annotations

from pathlib import Path

from beyond_click_sim.tasks.cold_start import ColdStartTask
from runners.in_distribution.cold_start.methods import agent4rec_listwise_ranking as preference_listwise
from runners.in_distribution.item_summaries import canonical_agent4rec_summary_usage


VLLM_QWEN36_27B_TRAITS_METHOD_NAME = "agent4rec_listwise_ranking_vllm_qwen36_27b_traits"
VLLM_QWEN36_27B_TRAITS_TASTE_METHOD_NAME = (
    "agent4rec_listwise_ranking_vllm_qwen36_27b_"
    "traits_taste_gpt4o_mini_candidate_summary"
)
VLLM_QWEN36_27B_TRAITS_TASTE_SUMBOTH_METHOD_NAME = (
    "agent4rec_listwise_ranking_vllm_qwen36_27b_"
    "traits_taste_gpt4o_mini_sumboth"
)
VLLM_QWEN36_35B_A3B_TRAITS_METHOD_NAME = (
    "agent4rec_listwise_ranking_vllm_qwen36_35b_a3b_traits"
)
VLLM_QWEN36_35B_A3B_TRAITS_TASTE_METHOD_NAME = (
    "agent4rec_listwise_ranking_vllm_qwen36_35b_a3b_"
    "traits_taste_gpt4o_mini_candidate_summary"
)
VLLM_QWEN36_35B_A3B_TRAITS_TASTE_SUMBOTH_METHOD_NAME = (
    "agent4rec_listwise_ranking_vllm_qwen36_35b_a3b_"
    "traits_taste_gpt4o_mini_sumboth"
)
VLLM_QWEN3_8B_METHOD_BASE = "agent4rec_listwise_ranking_vllm_qwen3_8b"
VLLM_LLAMA33_70B_METHOD_BASE = "agent4rec_listwise_ranking_vllm_llama33_70b"


def _run_traits(
    task: ColdStartTask,
    output_dir: Path,
    *,
    method_name: str,
    client_name: str,
    model: str,
    max_candidate_groups: int | None,
    extra_body: dict | None = None,
) -> dict[str, object]:
    resolved_extra_body = (
        preference_listwise.QWEN_EXTRA_BODY if extra_body is None else extra_body
    )
    return preference_listwise.run_method(
        task,
        output_dir,
        method_name=method_name,
        client_name=client_name,
        model=model,
        max_candidate_groups=max_candidate_groups,
        max_workers=preference_listwise.VLLM_MAX_WORKERS,
        extra_body=resolved_extra_body,
        profile_components=("traits",),
        summary_usage=canonical_agent4rec_summary_usage(task),
        ranking_target="interaction",
    )


def _run_traits_taste_candidate_summary(
    task: ColdStartTask,
    output_dir: Path,
    *,
    method_name: str,
    client_name: str,
    model: str,
    max_candidate_groups: int | None,
    extra_body: dict | None = None,
) -> dict[str, object]:
    resolved_extra_body = (
        preference_listwise.QWEN_EXTRA_BODY if extra_body is None else extra_body
    )
    return preference_listwise.run_method(
        task,
        output_dir,
        method_name=method_name,
        client_name=client_name,
        model=model,
        max_candidate_groups=max_candidate_groups,
        max_workers=preference_listwise.VLLM_MAX_WORKERS,
        extra_body=resolved_extra_body,
        profile_components=("traits", "taste"),
        taste_client_name=preference_listwise.OPENAI_CLIENT,
        taste_model=preference_listwise.GPT4O_MINI_TASTE_MODEL,
        summary_usage="candidate",
        ranking_target="interaction",
    )


def _run_traits_taste_sumboth(
    task: ColdStartTask,
    output_dir: Path,
    *,
    method_name: str,
    client_name: str,
    model: str,
    max_candidate_groups: int | None,
    extra_body: dict | None = None,
) -> dict[str, object]:
    resolved_extra_body = (
        preference_listwise.QWEN_EXTRA_BODY if extra_body is None else extra_body
    )
    return preference_listwise.run_method(
        task,
        output_dir,
        method_name=method_name,
        client_name=client_name,
        model=model,
        max_candidate_groups=max_candidate_groups,
        max_workers=preference_listwise.VLLM_MAX_WORKERS,
        extra_body=resolved_extra_body,
        profile_components=("traits", "taste"),
        taste_client_name=preference_listwise.OPENAI_CLIENT,
        taste_model=preference_listwise.GPT4O_MINI_TASTE_MODEL,
        summary_usage="both",
        ranking_target="interaction",
    )


def run_qwen36_27b_traits_smoke(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_TRAITS_METHOD_NAME}_smoke",
        client_name=preference_listwise.VLLM_QWEN36_27B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=25,
    )


def run_qwen36_27b_traits_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_TRAITS_METHOD_NAME}_full",
        client_name=preference_listwise.VLLM_QWEN36_27B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=None,
    )


def run_qwen36_27b_traits_taste_gpt4o_mini_candidate_summary_smoke(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_candidate_summary(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_TRAITS_TASTE_METHOD_NAME}_smoke",
        client_name=preference_listwise.VLLM_QWEN36_27B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=25,
    )


def run_qwen36_27b_traits_taste_gpt4o_mini_candidate_summary_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_candidate_summary(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_TRAITS_TASTE_METHOD_NAME}_full",
        client_name=preference_listwise.VLLM_QWEN36_27B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=None,
    )


def run_qwen36_27b_traits_taste_gpt4o_mini_sumboth_smoke(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_sumboth(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_TRAITS_TASTE_SUMBOTH_METHOD_NAME}_smoke",
        client_name=preference_listwise.VLLM_QWEN36_27B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=25,
    )


def run_qwen36_27b_traits_taste_gpt4o_mini_sumboth_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_sumboth(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_TRAITS_TASTE_SUMBOTH_METHOD_NAME}_full",
        client_name=preference_listwise.VLLM_QWEN36_27B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=None,
    )


def run_qwen36_35b_a3b_traits_smoke(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_TRAITS_METHOD_NAME}_smoke",
        client_name=preference_listwise.VLLM_QWEN36_35B_A3B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=25,
    )


def run_qwen36_35b_a3b_traits_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_TRAITS_METHOD_NAME}_full",
        client_name=preference_listwise.VLLM_QWEN36_35B_A3B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=None,
    )


def run_qwen36_35b_a3b_traits_taste_gpt4o_mini_candidate_summary_smoke(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_candidate_summary(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_TRAITS_TASTE_METHOD_NAME}_smoke",
        client_name=preference_listwise.VLLM_QWEN36_35B_A3B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=25,
    )


def run_qwen36_35b_a3b_traits_taste_gpt4o_mini_candidate_summary_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_candidate_summary(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_TRAITS_TASTE_METHOD_NAME}_full",
        client_name=preference_listwise.VLLM_QWEN36_35B_A3B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=None,
    )


def run_qwen36_35b_a3b_traits_taste_gpt4o_mini_sumboth_smoke(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_sumboth(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_TRAITS_TASTE_SUMBOTH_METHOD_NAME}_smoke",
        client_name=preference_listwise.VLLM_QWEN36_35B_A3B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=25,
    )


def run_qwen36_35b_a3b_traits_taste_gpt4o_mini_sumboth_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_sumboth(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_TRAITS_TASTE_SUMBOTH_METHOD_NAME}_full",
        client_name=preference_listwise.VLLM_QWEN36_35B_A3B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=None,
    )


def run_qwen3_8b_traits_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN3_8B_METHOD_BASE}_traits_full",
        client_name=preference_listwise.VLLM_QWEN3_8B_CLIENT,
        model=preference_listwise.VLLM_QWEN3_8B_MODEL,
        max_candidate_groups=None,
    )


def run_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_candidate_summary(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN3_8B_METHOD_BASE}_traits_taste_gpt4o_mini_candidate_summary_full",
        client_name=preference_listwise.VLLM_QWEN3_8B_CLIENT,
        model=preference_listwise.VLLM_QWEN3_8B_MODEL,
        max_candidate_groups=None,
    )


def run_qwen3_8b_traits_taste_gpt4o_mini_sumboth_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_sumboth(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN3_8B_METHOD_BASE}_traits_taste_gpt4o_mini_sumboth_full",
        client_name=preference_listwise.VLLM_QWEN3_8B_CLIENT,
        model=preference_listwise.VLLM_QWEN3_8B_MODEL,
        max_candidate_groups=None,
    )


def run_llama33_70b_traits_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits(
        task,
        output_dir,
        method_name=f"{VLLM_LLAMA33_70B_METHOD_BASE}_traits_full",
        client_name=preference_listwise.VLLM_LLAMA33_70B_CLIENT,
        model=preference_listwise.VLLM_LLAMA33_70B_MODEL,
        max_candidate_groups=None,
        extra_body=None,
    )


def run_llama33_70b_traits_taste_gpt4o_mini_candidate_summary_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_candidate_summary(
        task,
        output_dir,
        method_name=f"{VLLM_LLAMA33_70B_METHOD_BASE}_traits_taste_gpt4o_mini_candidate_summary_full",
        client_name=preference_listwise.VLLM_LLAMA33_70B_CLIENT,
        model=preference_listwise.VLLM_LLAMA33_70B_MODEL,
        max_candidate_groups=None,
        extra_body=None,
    )


def run_llama33_70b_traits_taste_gpt4o_mini_sumboth_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_traits_taste_sumboth(
        task,
        output_dir,
        method_name=f"{VLLM_LLAMA33_70B_METHOD_BASE}_traits_taste_gpt4o_mini_sumboth_full",
        client_name=preference_listwise.VLLM_LLAMA33_70B_CLIENT,
        model=preference_listwise.VLLM_LLAMA33_70B_MODEL,
        max_candidate_groups=None,
        extra_body=None,
    )
