from __future__ import annotations

from pathlib import Path

from beyond_click_sim.tasks.cold_start import ColdStartTask
from runners.in_distribution.cold_start.methods import llm_listwise_ranking as preference_listwise
from runners.in_distribution.item_summaries import SummaryVisibility


VLLM_QWEN36_27B_METHOD_NAME = "llm_listwise_ranking_vllm_qwen36_27b"
VLLM_QWEN36_35B_A3B_METHOD_NAME = "llm_listwise_ranking_vllm_qwen36_35b_a3b"
VLLM_QWEN3_8B_METHOD_NAME = "llm_listwise_ranking_vllm_qwen3_8b"
VLLM_LLAMA33_70B_METHOD_NAME = "llm_listwise_ranking_vllm_llama33_70b"


def run_qwen36_27b_smoke(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return preference_listwise.run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_smoke",
        client_name=preference_listwise.VLLM_QWEN36_27B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=25,
        max_workers=preference_listwise.VLLM_MAX_WORKERS,
        extra_body=preference_listwise.QWEN_EXTRA_BODY,
        ranking_target="interaction",
    )


def run_qwen36_27b_full(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return preference_listwise.run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_full",
        client_name=preference_listwise.VLLM_QWEN36_27B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=None,
        max_workers=preference_listwise.VLLM_MAX_WORKERS,
        extra_body=preference_listwise.QWEN_EXTRA_BODY,
        ranking_target="interaction",
    )


def run_qwen36_27b_sumboth_smoke(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return preference_listwise.run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_sumboth_smoke",
        client_name=preference_listwise.VLLM_QWEN36_27B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=25,
        max_workers=preference_listwise.VLLM_MAX_WORKERS,
        extra_body=preference_listwise.QWEN_EXTRA_BODY,
        summary_visibility="both",
        ranking_target="interaction",
    )


def run_qwen36_27b_sumboth_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return preference_listwise.run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_sumboth_full",
        client_name=preference_listwise.VLLM_QWEN36_27B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_27B_MODEL,
        max_candidate_groups=None,
        max_workers=preference_listwise.VLLM_MAX_WORKERS,
        extra_body=preference_listwise.QWEN_EXTRA_BODY,
        summary_visibility="both",
        ranking_target="interaction",
    )


def run_qwen36_35b_a3b_smoke(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return preference_listwise.run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_smoke",
        client_name=preference_listwise.VLLM_QWEN36_35B_A3B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=25,
        max_workers=preference_listwise.VLLM_MAX_WORKERS,
        extra_body=preference_listwise.QWEN_EXTRA_BODY,
        ranking_target="interaction",
    )


def run_qwen36_35b_a3b_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return preference_listwise.run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_full",
        client_name=preference_listwise.VLLM_QWEN36_35B_A3B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=None,
        max_workers=preference_listwise.VLLM_MAX_WORKERS,
        extra_body=preference_listwise.QWEN_EXTRA_BODY,
        ranking_target="interaction",
    )


def run_qwen36_35b_a3b_sumboth_smoke(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return preference_listwise.run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_sumboth_smoke",
        client_name=preference_listwise.VLLM_QWEN36_35B_A3B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=25,
        max_workers=preference_listwise.VLLM_MAX_WORKERS,
        extra_body=preference_listwise.QWEN_EXTRA_BODY,
        summary_visibility="both",
        ranking_target="interaction",
    )


def run_qwen36_35b_a3b_sumboth_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return preference_listwise.run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_sumboth_full",
        client_name=preference_listwise.VLLM_QWEN36_35B_A3B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_35B_A3B_MODEL,
        max_candidate_groups=None,
        max_workers=preference_listwise.VLLM_MAX_WORKERS,
        extra_body=preference_listwise.QWEN_EXTRA_BODY,
        summary_visibility="both",
        ranking_target="interaction",
    )


def _run_full_variant(
    task: ColdStartTask,
    output_dir: Path,
    *,
    method_name: str,
    client_name: str,
    model: str,
    extra_body: dict | None = None,
    use_item_stats: bool = False,
    summary_visibility: SummaryVisibility = "none",
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
        max_candidate_groups=None,
        max_workers=preference_listwise.VLLM_MAX_WORKERS,
        extra_body=resolved_extra_body,
        use_item_stats=use_item_stats,
        summary_visibility=summary_visibility,
        ranking_target="interaction",
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


def run_qwen36_27b_item_stats_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_full_variant(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_item_stats_full",
        client_name=preference_listwise.VLLM_QWEN36_27B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_27B_MODEL,
        use_item_stats=True,
    )


def run_qwen36_27b_sumhistory_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_full_variant(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_sumhistory_full",
        client_name=preference_listwise.VLLM_QWEN36_27B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_27B_MODEL,
        summary_visibility="history",
    )


def run_qwen36_27b_sumcandidate_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_full_variant(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_sumcandidate_full",
        client_name=preference_listwise.VLLM_QWEN36_27B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_27B_MODEL,
        summary_visibility="candidate",
    )


def run_qwen36_27b_item_stats_sumhistory_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_full_variant(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_27B_METHOD_NAME}_item_stats_sumhistory_full",
        client_name=preference_listwise.VLLM_QWEN36_27B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_27B_MODEL,
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
        client_name=preference_listwise.VLLM_QWEN36_27B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_27B_MODEL,
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
        client_name=preference_listwise.VLLM_QWEN36_27B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_27B_MODEL,
        use_item_stats=True,
        summary_visibility="both",
    )


def run_qwen36_35b_a3b_item_stats_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_full_variant(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_item_stats_full",
        client_name=preference_listwise.VLLM_QWEN36_35B_A3B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_35B_A3B_MODEL,
        use_item_stats=True,
    )


def run_qwen36_35b_a3b_sumhistory_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_full_variant(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_sumhistory_full",
        client_name=preference_listwise.VLLM_QWEN36_35B_A3B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_35B_A3B_MODEL,
        summary_visibility="history",
    )


def run_qwen36_35b_a3b_sumcandidate_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_full_variant(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_sumcandidate_full",
        client_name=preference_listwise.VLLM_QWEN36_35B_A3B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_35B_A3B_MODEL,
        summary_visibility="candidate",
    )


def run_qwen36_35b_a3b_item_stats_sumhistory_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_full_variant(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN36_35B_A3B_METHOD_NAME}_item_stats_sumhistory_full",
        client_name=preference_listwise.VLLM_QWEN36_35B_A3B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_35B_A3B_MODEL,
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
        client_name=preference_listwise.VLLM_QWEN36_35B_A3B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_35B_A3B_MODEL,
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
        client_name=preference_listwise.VLLM_QWEN36_35B_A3B_CLIENT,
        model=preference_listwise.VLLM_QWEN36_35B_A3B_MODEL,
        use_item_stats=True,
        summary_visibility="both",
    )


def run_qwen3_8b_full(task: ColdStartTask, output_dir: Path) -> dict[str, object]:
    return _run_backbone_ablation_full(
        task,
        output_dir,
        method_base=VLLM_QWEN3_8B_METHOD_NAME,
        client_name=preference_listwise.VLLM_QWEN3_8B_CLIENT,
        model=preference_listwise.VLLM_QWEN3_8B_MODEL,
        extra_body=preference_listwise.QWEN_EXTRA_BODY,
    )


def run_qwen3_8b_item_stats_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_backbone_ablation_full(
        task,
        output_dir,
        method_base=VLLM_QWEN3_8B_METHOD_NAME,
        client_name=preference_listwise.VLLM_QWEN3_8B_CLIENT,
        model=preference_listwise.VLLM_QWEN3_8B_MODEL,
        extra_body=preference_listwise.QWEN_EXTRA_BODY,
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
        client_name=preference_listwise.VLLM_QWEN3_8B_CLIENT,
        model=preference_listwise.VLLM_QWEN3_8B_MODEL,
        extra_body=preference_listwise.QWEN_EXTRA_BODY,
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
        client_name=preference_listwise.VLLM_QWEN3_8B_CLIENT,
        model=preference_listwise.VLLM_QWEN3_8B_MODEL,
        extra_body=preference_listwise.QWEN_EXTRA_BODY,
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
        client_name=preference_listwise.VLLM_QWEN3_8B_CLIENT,
        model=preference_listwise.VLLM_QWEN3_8B_MODEL,
        extra_body=preference_listwise.QWEN_EXTRA_BODY,
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
        client_name=preference_listwise.VLLM_QWEN3_8B_CLIENT,
        model=preference_listwise.VLLM_QWEN3_8B_MODEL,
        extra_body=preference_listwise.QWEN_EXTRA_BODY,
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
        client_name=preference_listwise.VLLM_QWEN3_8B_CLIENT,
        model=preference_listwise.VLLM_QWEN3_8B_MODEL,
        extra_body=preference_listwise.QWEN_EXTRA_BODY,
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
        client_name=preference_listwise.VLLM_QWEN3_8B_CLIENT,
        model=preference_listwise.VLLM_QWEN3_8B_MODEL,
        extra_body=preference_listwise.QWEN_EXTRA_BODY,
        variant="item_stats_sumboth",
        use_item_stats=True,
        summary_visibility="both",
    )


def run_llama33_70b_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_backbone_ablation_full(
        task,
        output_dir,
        method_base=VLLM_LLAMA33_70B_METHOD_NAME,
        client_name=preference_listwise.VLLM_LLAMA33_70B_CLIENT,
        model=preference_listwise.VLLM_LLAMA33_70B_MODEL,
        extra_body=None,
    )


def run_llama33_70b_item_stats_full(
    task: ColdStartTask, output_dir: Path
) -> dict[str, object]:
    return _run_backbone_ablation_full(
        task,
        output_dir,
        method_base=VLLM_LLAMA33_70B_METHOD_NAME,
        client_name=preference_listwise.VLLM_LLAMA33_70B_CLIENT,
        model=preference_listwise.VLLM_LLAMA33_70B_MODEL,
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
        client_name=preference_listwise.VLLM_LLAMA33_70B_CLIENT,
        model=preference_listwise.VLLM_LLAMA33_70B_MODEL,
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
        client_name=preference_listwise.VLLM_LLAMA33_70B_CLIENT,
        model=preference_listwise.VLLM_LLAMA33_70B_MODEL,
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
        client_name=preference_listwise.VLLM_LLAMA33_70B_CLIENT,
        model=preference_listwise.VLLM_LLAMA33_70B_MODEL,
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
        client_name=preference_listwise.VLLM_LLAMA33_70B_CLIENT,
        model=preference_listwise.VLLM_LLAMA33_70B_MODEL,
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
        client_name=preference_listwise.VLLM_LLAMA33_70B_CLIENT,
        model=preference_listwise.VLLM_LLAMA33_70B_MODEL,
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
        client_name=preference_listwise.VLLM_LLAMA33_70B_CLIENT,
        model=preference_listwise.VLLM_LLAMA33_70B_MODEL,
        extra_body=None,
        variant="item_stats_sumboth",
        use_item_stats=True,
        summary_visibility="both",
    )
