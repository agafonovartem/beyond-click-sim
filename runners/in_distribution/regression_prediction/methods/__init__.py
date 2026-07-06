from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from beyond_click_sim.tasks import Task

from runners.in_distribution.regression_prediction.methods.agent4rec_regressor import (
    run_qwen3_8b_taste_gpt4o_mini_full as run_agent4rec_qwen3_8b_taste_gpt4o_mini_full,
    run_qwen3_8b_taste_gpt4o_mini_summary_full as run_agent4rec_qwen3_8b_taste_gpt4o_mini_summary_full,
    run_qwen3_8b_taste_gpt4o_mini_smoke as run_agent4rec_qwen3_8b_taste_gpt4o_mini_smoke,
    run_qwen3_8b_traits_full as run_agent4rec_qwen3_8b_traits_full,
    run_qwen3_8b_traits_summary_full as run_agent4rec_qwen3_8b_traits_summary_full,
    run_qwen3_8b_traits_smoke as run_agent4rec_qwen3_8b_traits_smoke,
    run_qwen3_8b_traits_taste_gpt4o_mini_full as run_agent4rec_qwen3_8b_traits_taste_gpt4o_mini_full,
    run_qwen3_8b_traits_taste_gpt4o_mini_summary_full as run_agent4rec_qwen3_8b_traits_taste_gpt4o_mini_summary_full,
    run_qwen3_8b_traits_taste_gpt4o_mini_smoke as run_agent4rec_qwen3_8b_traits_taste_gpt4o_mini_smoke,
    run_qwen36_27b_traits_taste_gpt4o_mini_full as run_agent4rec_qwen36_27b_traits_taste_gpt4o_mini_full,
    run_qwen36_27b_traits_taste_gpt4o_mini_smoke as run_agent4rec_qwen36_27b_traits_taste_gpt4o_mini_smoke,
)
from runners.in_distribution.regression_prediction.methods.item import (
    run_item_mean,
    run_item_mode,
)
from runners.in_distribution.regression_prediction.methods.llm_regressor import (
    run_gpt54_mini_full,
    run_gpt54_mini_smoke,
    run_gpt54_mini_with_item_stats_full,
    run_gpt54_mini_with_item_stats_smoke,
    run_gpt55_full,
    run_gpt55_smoke,
    run_gpt55_with_item_stats_full,
    run_gpt55_with_item_stats_smoke,
    run_llama31_8b_full,
    run_llama31_8b_smoke,
    run_llama31_8b_with_item_stats_full,
    run_llama31_8b_with_item_stats_smoke,
    run_llama33_70b_full,
    run_llama33_70b_smoke,
    run_llama33_70b_with_item_stats_full,
    run_llama33_70b_with_item_stats_smoke,
    run_qwen3_8b_with_item_stats_summary_full,
    run_qwen3_8b_with_item_stats_full,
    run_qwen3_8b_with_item_stats_smoke,
)
from runners.in_distribution.regression_prediction.methods.mean import run as run_mean
from runners.in_distribution.regression_prediction.methods.mode import run as run_mode
from runners.in_distribution.regression_prediction.methods.user import (
    run_user_mean,
    run_user_mode,
)


MethodRunner = Callable[[Task, Path], dict[str, object]]

METHOD_RUNNERS: dict[str, MethodRunner] = {
    "mean_regressor": run_mean,
    "mode_regressor": run_mode,
    "item_mean_regressor": run_item_mean,
    "item_mode_regressor": run_item_mode,
    "user_mean_regressor": run_user_mean,
    "user_mode_regressor": run_user_mode,
    "agent4rec_regressor_vllm_qwen36_27b_traits_taste_gpt4o_mini_smoke": (
        run_agent4rec_qwen36_27b_traits_taste_gpt4o_mini_smoke
    ),
    "agent4rec_regressor_vllm_qwen36_27b_traits_taste_gpt4o_mini_full": (
        run_agent4rec_qwen36_27b_traits_taste_gpt4o_mini_full
    ),
    "agent4rec_regressor_vllm_qwen3_8b_traits_smoke": (
        run_agent4rec_qwen3_8b_traits_smoke
    ),
    "agent4rec_regressor_vllm_qwen3_8b_traits_full": (
        run_agent4rec_qwen3_8b_traits_full
    ),
    "agent4rec_regressor_vllm_qwen3_8b_traits_summary_full": (
        run_agent4rec_qwen3_8b_traits_summary_full
    ),
    "agent4rec_regressor_vllm_qwen3_8b_taste_gpt4o_mini_smoke": (
        run_agent4rec_qwen3_8b_taste_gpt4o_mini_smoke
    ),
    "agent4rec_regressor_vllm_qwen3_8b_taste_gpt4o_mini_full": (
        run_agent4rec_qwen3_8b_taste_gpt4o_mini_full
    ),
    "agent4rec_regressor_vllm_qwen3_8b_taste_gpt4o_mini_summary_full": (
        run_agent4rec_qwen3_8b_taste_gpt4o_mini_summary_full
    ),
    "agent4rec_regressor_vllm_qwen3_8b_traits_taste_gpt4o_mini_smoke": (
        run_agent4rec_qwen3_8b_traits_taste_gpt4o_mini_smoke
    ),
    "agent4rec_regressor_vllm_qwen3_8b_traits_taste_gpt4o_mini_full": (
        run_agent4rec_qwen3_8b_traits_taste_gpt4o_mini_full
    ),
    "agent4rec_regressor_vllm_qwen3_8b_traits_taste_gpt4o_mini_summary_full": (
        run_agent4rec_qwen3_8b_traits_taste_gpt4o_mini_summary_full
    ),
    "llm_regressor_ollama_llama31_8b_smoke": run_llama31_8b_smoke,
    "llm_regressor_ollama_llama31_8b_full": run_llama31_8b_full,
    "llm_regressor_ollama_llama31_8b_with_item_stats_smoke": (
        run_llama31_8b_with_item_stats_smoke
    ),
    "llm_regressor_ollama_llama31_8b_with_item_stats_full": (
        run_llama31_8b_with_item_stats_full
    ),
    "llm_regressor_vllm_llama33_70b_smoke": run_llama33_70b_smoke,
    "llm_regressor_vllm_llama33_70b_full": run_llama33_70b_full,
    "llm_regressor_vllm_llama33_70b_with_item_stats_smoke": (
        run_llama33_70b_with_item_stats_smoke
    ),
    "llm_regressor_vllm_llama33_70b_with_item_stats_full": (
        run_llama33_70b_with_item_stats_full
    ),
    "llm_regressor_vllm_qwen3_8b_with_item_stats_smoke": (
        run_qwen3_8b_with_item_stats_smoke
    ),
    "llm_regressor_vllm_qwen3_8b_with_item_stats_full": (
        run_qwen3_8b_with_item_stats_full
    ),
    "llm_regressor_vllm_qwen3_8b_with_item_stats_summary_full": (
        run_qwen3_8b_with_item_stats_summary_full
    ),
    "llm_regressor_openai_vk_gpt54_mini_smoke": run_gpt54_mini_smoke,
    "llm_regressor_openai_vk_gpt54_mini_full": run_gpt54_mini_full,
    "llm_regressor_openai_vk_gpt54_mini_with_item_stats_smoke": (
        run_gpt54_mini_with_item_stats_smoke
    ),
    "llm_regressor_openai_vk_gpt54_mini_with_item_stats_full": (
        run_gpt54_mini_with_item_stats_full
    ),
    "llm_regressor_openai_vk_gpt55_smoke": run_gpt55_smoke,
    "llm_regressor_openai_vk_gpt55_full": run_gpt55_full,
    "llm_regressor_openai_vk_gpt55_with_item_stats_smoke": (
        run_gpt55_with_item_stats_smoke
    ),
    "llm_regressor_openai_vk_gpt55_with_item_stats_full": (
        run_gpt55_with_item_stats_full
    ),
}
