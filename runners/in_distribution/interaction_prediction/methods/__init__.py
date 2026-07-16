from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from beyond_click_sim.tasks import Task

from runners.in_distribution.interaction_prediction.methods.agent4rec_yes_no import (
    run_llama31_8b_full as run_agent4rec_llama31_8b_full,
    run_llama31_8b_smoke as run_agent4rec_llama31_8b_smoke,
    run_llama33_70b_full as run_agent4rec_llama33_70b_full,
    run_llama33_70b_smoke as run_agent4rec_llama33_70b_smoke,
    run_qwen36_27b_traits_taste_gpt4o_mini_full as run_agent4rec_qwen36_27b_traits_taste_gpt4o_mini_full,
    run_qwen36_27b_traits_taste_gpt4o_mini_smoke as run_agent4rec_qwen36_27b_traits_taste_gpt4o_mini_smoke,
    run_qwen36_27b_port8001_full as run_agent4rec_qwen36_27b_port8001_full,
    run_qwen36_27b_port8001_smoke as run_agent4rec_qwen36_27b_port8001_smoke,
    run_qwen36_27b_port8002_full as run_agent4rec_qwen36_27b_port8002_full,
    run_qwen36_27b_port8002_smoke as run_agent4rec_qwen36_27b_port8002_smoke,
)
from runners.in_distribution.interaction_prediction.methods.agent4rec_listwise_ranking import (
    run_qwen36_27b_traits_full as run_agent4rec_listwise_qwen36_27b_traits_full,
    run_qwen36_27b_traits_smoke as run_agent4rec_listwise_qwen36_27b_traits_smoke,
)
from runners.in_distribution.interaction_prediction.methods.llm_yes_no import (
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
    run_qwen36_27b_full,
    run_qwen36_27b_smoke,
    run_qwen36_27b_with_item_stats_full,
    run_qwen36_27b_with_item_stats_smoke,
)
from runners.in_distribution.interaction_prediction.methods.llm_listwise_ranking import (
    run_qwen36_27b_with_item_stats_full as run_listwise_qwen36_27b_with_item_stats_full,
    run_qwen36_27b_with_item_stats_smoke as run_listwise_qwen36_27b_with_item_stats_smoke,
)
from runners.in_distribution.interaction_prediction.methods.popularity import (
    run as run_popularity,
    run_ranking as run_popularity_ranking,
)


MethodRunner = Callable[[Task, Path], dict[str, object]]

METHOD_RUNNERS: dict[str, MethodRunner] = {
    "popularity_f1_threshold": run_popularity,
    "popularity_ranking": run_popularity_ranking,
    "agent4rec_yes_no_ollama_llama31_8b_smoke": run_agent4rec_llama31_8b_smoke,
    "agent4rec_yes_no_ollama_llama31_8b_full": run_agent4rec_llama31_8b_full,
    "agent4rec_yes_no_vllm_llama33_70b_smoke": run_agent4rec_llama33_70b_smoke,
    "agent4rec_yes_no_vllm_llama33_70b_full": run_agent4rec_llama33_70b_full,
    "agent4rec_yes_no_vllm_qwen36_27b_traits_taste_gpt4o_mini_smoke": (
        run_agent4rec_qwen36_27b_traits_taste_gpt4o_mini_smoke
    ),
    "agent4rec_yes_no_vllm_qwen36_27b_traits_taste_gpt4o_mini_full": (
        run_agent4rec_qwen36_27b_traits_taste_gpt4o_mini_full
    ),
    "agent4rec_yes_no_vllm_qwen36_27b_port8001_smoke": (
        run_agent4rec_qwen36_27b_port8001_smoke
    ),
    "agent4rec_yes_no_vllm_qwen36_27b_port8001_full": (
        run_agent4rec_qwen36_27b_port8001_full
    ),
    "agent4rec_yes_no_vllm_qwen36_27b_port8002_smoke": (
        run_agent4rec_qwen36_27b_port8002_smoke
    ),
    "agent4rec_yes_no_vllm_qwen36_27b_port8002_full": (
        run_agent4rec_qwen36_27b_port8002_full
    ),
    "agent4rec_listwise_ranking_vllm_qwen36_27b_traits_smoke": (
        run_agent4rec_listwise_qwen36_27b_traits_smoke
    ),
    "agent4rec_listwise_ranking_vllm_qwen36_27b_traits_full": (
        run_agent4rec_listwise_qwen36_27b_traits_full
    ),
    "llm_yes_no_ollama_llama31_8b_smoke": run_llama31_8b_smoke,
    "llm_yes_no_ollama_llama31_8b_full": run_llama31_8b_full,
    "llm_yes_no_ollama_llama31_8b_with_item_stats_smoke": (
        run_llama31_8b_with_item_stats_smoke
    ),
    "llm_yes_no_ollama_llama31_8b_with_item_stats_full": (
        run_llama31_8b_with_item_stats_full
    ),
    "llm_yes_no_vllm_llama33_70b_smoke": run_llama33_70b_smoke,
    "llm_yes_no_vllm_llama33_70b_full": run_llama33_70b_full,
    "llm_yes_no_vllm_llama33_70b_with_item_stats_smoke": (
        run_llama33_70b_with_item_stats_smoke
    ),
    "llm_yes_no_vllm_llama33_70b_with_item_stats_full": (
        run_llama33_70b_with_item_stats_full
    ),
    "llm_yes_no_vllm_qwen36_27b_smoke": run_qwen36_27b_smoke,
    "llm_yes_no_vllm_qwen36_27b_full": run_qwen36_27b_full,
    "llm_yes_no_vllm_qwen36_27b_with_item_stats_smoke": (
        run_qwen36_27b_with_item_stats_smoke
    ),
    "llm_yes_no_vllm_qwen36_27b_with_item_stats_full": (
        run_qwen36_27b_with_item_stats_full
    ),
    "llm_listwise_ranking_vllm_qwen36_27b_with_item_stats_smoke": (
        run_listwise_qwen36_27b_with_item_stats_smoke
    ),
    "llm_listwise_ranking_vllm_qwen36_27b_with_item_stats_full": (
        run_listwise_qwen36_27b_with_item_stats_full
    ),
    "llm_yes_no_openai_vk_gpt54_mini_smoke": run_gpt54_mini_smoke,
    "llm_yes_no_openai_vk_gpt54_mini_full": run_gpt54_mini_full,
    "llm_yes_no_openai_vk_gpt54_mini_with_item_stats_smoke": (
        run_gpt54_mini_with_item_stats_smoke
    ),
    "llm_yes_no_openai_vk_gpt54_mini_with_item_stats_full": (
        run_gpt54_mini_with_item_stats_full
    ),
    "llm_yes_no_openai_vk_gpt55_smoke": run_gpt55_smoke,
    "llm_yes_no_openai_vk_gpt55_full": run_gpt55_full,
    "llm_yes_no_openai_vk_gpt55_with_item_stats_smoke": (
        run_gpt55_with_item_stats_smoke
    ),
    "llm_yes_no_openai_vk_gpt55_with_item_stats_full": (
        run_gpt55_with_item_stats_full
    ),
}
