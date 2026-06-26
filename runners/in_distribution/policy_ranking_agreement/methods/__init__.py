from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from beyond_click_sim.tasks import Task

from runners.in_distribution.policy_ranking_agreement.methods.llm_yes_no import (
    run_llama31_8b_full,
    run_llama31_8b_smoke,
    run_llama32_full,
    run_llama32_smoke,
    run_llama33_70b_full,
    run_llama33_70b_smoke,
    run_qwen3_8b_full,
    run_qwen3_8b_smoke,
    run_qwen36_27b_full,
    run_qwen36_27b_smoke,
    run_qwen36_35b_a3b_full,
    run_qwen36_35b_a3b_smoke,
)
from runners.in_distribution.policy_ranking_agreement.methods.popularity import (
    run as run_popularity,
)


MethodRunner = Callable[[Task, Path], dict[str, object]]

METHOD_RUNNERS: dict[str, MethodRunner] = {
    "popularity_scorer": run_popularity,
    "llm_yes_no_ollama_llama31_8b_smoke": run_llama31_8b_smoke,
    "llm_yes_no_ollama_llama31_8b_full": run_llama31_8b_full,
    "llm_yes_no_ollama_llama32_smoke": run_llama32_smoke,
    "llm_yes_no_ollama_llama32_full": run_llama32_full,
    "llm_yes_no_vllm_llama33_70b_smoke": run_llama33_70b_smoke,
    "llm_yes_no_vllm_llama33_70b_full": run_llama33_70b_full,
    "llm_yes_no_vllm_qwen3_8b_smoke": run_qwen3_8b_smoke,
    "llm_yes_no_vllm_qwen3_8b_full": run_qwen3_8b_full,
    "llm_yes_no_vllm_qwen36_27b_smoke": run_qwen36_27b_smoke,
    "llm_yes_no_vllm_qwen36_27b_full": run_qwen36_27b_full,
    "llm_yes_no_vllm_qwen36_35b_a3b_smoke": run_qwen36_35b_a3b_smoke,
    "llm_yes_no_vllm_qwen36_35b_a3b_full": run_qwen36_35b_a3b_full,
}
