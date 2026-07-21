from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from beyond_click_sim.tasks.cold_start import ColdStartTask

from runners.in_distribution.cold_start.methods.agent4rec_yes_no import (
    run_llama31_8b_full as run_agent4rec_llama31_8b_full,
    run_llama31_8b_itemwise_full as run_agent4rec_llama31_8b_itemwise_full,
    run_llama31_8b_itemwise_smoke as run_agent4rec_llama31_8b_itemwise_smoke,
    run_llama31_8b_smoke as run_agent4rec_llama31_8b_smoke,
    run_llama33_70b_full as run_agent4rec_llama33_70b_full,
    run_llama33_70b_itemwise_full as run_agent4rec_llama33_70b_itemwise_full,
    run_llama33_70b_itemwise_smoke as run_agent4rec_llama33_70b_itemwise_smoke,
    run_llama33_70b_smoke as run_agent4rec_llama33_70b_smoke,
    run_qwen36_27b_full as run_agent4rec_qwen36_27b_full,
    run_qwen36_27b_item_stats_full,
    run_qwen36_27b_item_stats_smoke,
    run_qwen36_27b_item_stats_traits_taste_gpt4o_mini_full,
    run_qwen36_27b_item_stats_traits_taste_gpt4o_mini_smoke,
    run_qwen36_27b_itemwise_full as run_agent4rec_qwen36_27b_itemwise_full,
    run_qwen36_27b_itemwise_item_stats_full,
    run_qwen36_27b_itemwise_item_stats_smoke,
    run_qwen36_27b_itemwise_item_stats_traits_taste_gpt4o_mini_full,
    run_qwen36_27b_itemwise_item_stats_traits_taste_gpt4o_mini_smoke,
    run_qwen36_27b_itemwise_smoke as run_agent4rec_qwen36_27b_itemwise_smoke,
    run_qwen36_27b_itemwise_traits_taste_gpt4o_mini_full,
    run_qwen36_27b_itemwise_traits_taste_gpt4o_mini_smoke,
    run_qwen36_27b_smoke as run_agent4rec_qwen36_27b_smoke,
    run_qwen36_27b_traits_taste_gpt4o_mini_full,
    run_qwen36_27b_traits_taste_gpt4o_mini_smoke,
)
from runners.in_distribution.cold_start.methods.llm_yes_no import (
    run_llama31_8b_full,
    run_llama31_8b_itemwise_full,
    run_llama31_8b_itemwise_smoke,
    run_llama31_8b_smoke,
    run_llama33_70b_full,
    run_llama33_70b_itemwise_full,
    run_llama33_70b_itemwise_smoke,
    run_llama33_70b_smoke,
    run_qwen36_27b_full,
    run_qwen36_27b_itemwise_full,
    run_qwen36_27b_itemwise_smoke,
    run_qwen36_27b_smoke,
)
from runners.in_distribution.cold_start.methods.item_knn import (
    run as run_item_knn,
    run_smoke as run_item_knn_smoke,
    run_ranking as run_item_knn_ranking,
    run_ranking_smoke as run_item_knn_ranking_smoke,
)
from runners.in_distribution.cold_start.methods.popularity import (
    run as run_popularity,
    run_ranking as run_popularity_ranking,
)


MethodRunner = Callable[[ColdStartTask, Path], dict[str, object]]

METHOD_RUNNERS: dict[str, MethodRunner] = {
    "popularity_f1_threshold": run_popularity,
    "popularity_ranking": run_popularity_ranking,
    "item_knn_cold_start": run_item_knn,
    "item_knn_cold_start_smoke": run_item_knn_smoke,
    "item_knn_cold_start_ranking": run_item_knn_ranking,
    "item_knn_cold_start_ranking_smoke": run_item_knn_ranking_smoke,
    "llm_yes_no_ollama_llama31_8b_smoke": run_llama31_8b_smoke,
    "llm_yes_no_ollama_llama31_8b_full": run_llama31_8b_full,
    "llm_yes_no_vllm_llama33_70b_smoke": run_llama33_70b_smoke,
    "llm_yes_no_vllm_llama33_70b_full": run_llama33_70b_full,
    "llm_yes_no_vllm_qwen36_27b_smoke": run_qwen36_27b_smoke,
    "llm_yes_no_vllm_qwen36_27b_full": run_qwen36_27b_full,
    # LLM yes/no — itemwise (one LLM call per candidate, no cross-candidate coupling)
    "llm_yes_no_itemwise_ollama_llama31_8b_smoke": run_llama31_8b_itemwise_smoke,
    "llm_yes_no_itemwise_ollama_llama31_8b_full": run_llama31_8b_itemwise_full,
    "llm_yes_no_itemwise_vllm_llama33_70b_smoke": run_llama33_70b_itemwise_smoke,
    "llm_yes_no_itemwise_vllm_llama33_70b_full": run_llama33_70b_itemwise_full,
    "llm_yes_no_itemwise_vllm_qwen36_27b_smoke": run_qwen36_27b_itemwise_smoke,
    "llm_yes_no_itemwise_vllm_qwen36_27b_full": run_qwen36_27b_itemwise_full,
    # Agent4Rec — traits only
    "agent4rec_yes_no_ollama_llama31_8b_smoke": run_agent4rec_llama31_8b_smoke,
    "agent4rec_yes_no_ollama_llama31_8b_full": run_agent4rec_llama31_8b_full,
    "agent4rec_yes_no_vllm_llama33_70b_smoke": run_agent4rec_llama33_70b_smoke,
    "agent4rec_yes_no_vllm_llama33_70b_full": run_agent4rec_llama33_70b_full,
    "agent4rec_yes_no_vllm_qwen36_27b_smoke": run_agent4rec_qwen36_27b_smoke,
    "agent4rec_yes_no_vllm_qwen36_27b_full": run_agent4rec_qwen36_27b_full,
    # Agent4Rec — traits only, itemwise
    "agent4rec_yes_no_itemwise_ollama_llama31_8b_smoke": run_agent4rec_llama31_8b_itemwise_smoke,
    "agent4rec_yes_no_itemwise_ollama_llama31_8b_full": run_agent4rec_llama31_8b_itemwise_full,
    "agent4rec_yes_no_itemwise_vllm_llama33_70b_smoke": run_agent4rec_llama33_70b_itemwise_smoke,
    "agent4rec_yes_no_itemwise_vllm_llama33_70b_full": run_agent4rec_llama33_70b_itemwise_full,
    "agent4rec_yes_no_itemwise_vllm_qwen36_27b_smoke": run_agent4rec_qwen36_27b_itemwise_smoke,
    "agent4rec_yes_no_itemwise_vllm_qwen36_27b_full": run_agent4rec_qwen36_27b_itemwise_full,
    # Agent4Rec — traits + taste (requires OPENAI_API_KEY)
    "agent4rec_yes_no_vllm_qwen36_27b_traits_taste_gpt4o_mini_smoke": run_qwen36_27b_traits_taste_gpt4o_mini_smoke,
    "agent4rec_yes_no_vllm_qwen36_27b_traits_taste_gpt4o_mini_full": run_qwen36_27b_traits_taste_gpt4o_mini_full,
    "agent4rec_yes_no_itemwise_vllm_qwen36_27b_traits_taste_gpt4o_mini_smoke": run_qwen36_27b_itemwise_traits_taste_gpt4o_mini_smoke,
    "agent4rec_yes_no_itemwise_vllm_qwen36_27b_traits_taste_gpt4o_mini_full": run_qwen36_27b_itemwise_traits_taste_gpt4o_mini_full,
    # Agent4Rec — traits only + item_stats (requires *_item_stats_* tasks)
    "agent4rec_yes_no_vllm_qwen36_27b_item_stats_smoke": run_qwen36_27b_item_stats_smoke,
    "agent4rec_yes_no_vllm_qwen36_27b_item_stats_full": run_qwen36_27b_item_stats_full,
    "agent4rec_yes_no_itemwise_vllm_qwen36_27b_item_stats_smoke": run_qwen36_27b_itemwise_item_stats_smoke,
    "agent4rec_yes_no_itemwise_vllm_qwen36_27b_item_stats_full": run_qwen36_27b_itemwise_item_stats_full,
    # Agent4Rec — traits + taste + item_stats
    "agent4rec_yes_no_vllm_qwen36_27b_item_stats_traits_taste_gpt4o_mini_smoke": run_qwen36_27b_item_stats_traits_taste_gpt4o_mini_smoke,
    "agent4rec_yes_no_vllm_qwen36_27b_item_stats_traits_taste_gpt4o_mini_full": run_qwen36_27b_item_stats_traits_taste_gpt4o_mini_full,
    "agent4rec_yes_no_itemwise_vllm_qwen36_27b_item_stats_traits_taste_gpt4o_mini_smoke": run_qwen36_27b_itemwise_item_stats_traits_taste_gpt4o_mini_smoke,
    "agent4rec_yes_no_itemwise_vllm_qwen36_27b_item_stats_traits_taste_gpt4o_mini_full": run_qwen36_27b_itemwise_item_stats_traits_taste_gpt4o_mini_full,
}
