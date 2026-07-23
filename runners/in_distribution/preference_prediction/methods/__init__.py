from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from beyond_click_sim.tasks import Task
from runners.in_distribution.preference_prediction.methods.agent4rec_listwise_ranking import (
    run_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_full as run_agent4rec_listwise_qwen3_8b_traits_taste_candidate_summary_full,
    run_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_smoke as run_agent4rec_listwise_qwen3_8b_traits_taste_candidate_summary_smoke,
    run_qwen3_8b_traits_taste_gpt4o_mini_no_summary_full as run_agent4rec_listwise_qwen3_8b_traits_taste_no_summary_full,
    run_qwen3_8b_traits_taste_gpt4o_mini_no_summary_smoke as run_agent4rec_listwise_qwen3_8b_traits_taste_no_summary_smoke,
    run_qwen36_27b_traits_taste_gpt4o_mini_candidate_summary_full as run_agent4rec_listwise_qwen36_27b_traits_taste_candidate_summary_full,
    run_qwen36_27b_traits_taste_gpt4o_mini_candidate_summary_smoke as run_agent4rec_listwise_qwen36_27b_traits_taste_candidate_summary_smoke,
    run_qwen36_27b_traits_taste_gpt4o_mini_no_summary_full as run_agent4rec_listwise_qwen36_27b_traits_taste_no_summary_full,
    run_qwen36_27b_traits_taste_gpt4o_mini_no_summary_smoke as run_agent4rec_listwise_qwen36_27b_traits_taste_no_summary_smoke,
    run_qwen36_27b_traits_full as run_agent4rec_listwise_qwen36_27b_traits_full,
    run_qwen36_27b_traits_smoke as run_agent4rec_listwise_qwen36_27b_traits_smoke,
)
from runners.in_distribution.preference_prediction.methods.agent4rec_yes_no import (
    run_qwen3_8b_full as run_agent4rec_qwen3_8b_full,
    run_qwen3_8b_smoke as run_agent4rec_qwen3_8b_smoke,
    run_qwen36_27b_full as run_agent4rec_qwen36_27b_full,
    run_qwen36_27b_smoke as run_agent4rec_qwen36_27b_smoke,
    run_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_full as run_agent4rec_qwen3_8b_traits_taste_candidate_summary_full,
    run_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_smoke as run_agent4rec_qwen3_8b_traits_taste_candidate_summary_smoke,
    run_qwen3_8b_traits_taste_gpt4o_mini_no_summary_full as run_agent4rec_qwen3_8b_traits_taste_no_summary_full,
    run_qwen3_8b_traits_taste_gpt4o_mini_no_summary_smoke as run_agent4rec_qwen3_8b_traits_taste_no_summary_smoke,
    run_qwen36_27b_traits_taste_gpt4o_mini_candidate_summary_full as run_agent4rec_qwen36_27b_traits_taste_candidate_summary_full,
    run_qwen36_27b_traits_taste_gpt4o_mini_candidate_summary_smoke as run_agent4rec_qwen36_27b_traits_taste_candidate_summary_smoke,
    run_qwen36_27b_traits_taste_gpt4o_mini_no_summary_full as run_agent4rec_qwen36_27b_traits_taste_no_summary_full,
    run_qwen36_27b_traits_taste_gpt4o_mini_no_summary_smoke as run_agent4rec_qwen36_27b_traits_taste_no_summary_smoke,
)
from runners.in_distribution.interaction_prediction.methods.popularity import (
    run as run_popularity,
    run_ranking as run_popularity_ranking,
)
from runners.in_distribution.preference_prediction.methods.llm_yes_no import (
    run_qwen3_8b_full,
    run_qwen3_8b_summary_full,
    run_qwen3_8b_smoke,
    run_qwen36_27b_full,
    run_qwen36_27b_summary_full,
    run_qwen36_27b_smoke,
)
from runners.in_distribution.preference_prediction.methods.llm_listwise_ranking import (
    run_qwen3_8b_full as run_listwise_qwen3_8b_full,
    run_qwen3_8b_smoke as run_listwise_qwen3_8b_smoke,
    run_qwen36_27b_full as run_listwise_qwen36_27b_full,
    run_qwen36_27b_smoke as run_listwise_qwen36_27b_smoke,
)


MethodRunner = Callable[[Task, Path], dict[str, object]]

METHOD_RUNNERS: dict[str, MethodRunner] = {
    "popularity_f1_threshold": run_popularity,
    "popularity_ranking": run_popularity_ranking,
    "agent4rec_preference_yes_no_litellm_qwen3_8b_traits_smoke": (
        run_agent4rec_qwen3_8b_smoke
    ),
    "agent4rec_preference_yes_no_litellm_qwen3_8b_traits_full": (
        run_agent4rec_qwen3_8b_full
    ),
    "agent4rec_preference_yes_no_litellm_qwen36_27b_traits_smoke": (
        run_agent4rec_qwen36_27b_smoke
    ),
    "agent4rec_preference_yes_no_litellm_qwen36_27b_traits_full": (
        run_agent4rec_qwen36_27b_full
    ),
    "agent4rec_preference_yes_no_litellm_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_smoke": (
        run_agent4rec_qwen3_8b_traits_taste_candidate_summary_smoke
    ),
    "agent4rec_preference_yes_no_litellm_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_full": (
        run_agent4rec_qwen3_8b_traits_taste_candidate_summary_full
    ),
    "agent4rec_preference_yes_no_litellm_qwen36_27b_traits_taste_gpt4o_mini_candidate_summary_smoke": (
        run_agent4rec_qwen36_27b_traits_taste_candidate_summary_smoke
    ),
    "agent4rec_preference_yes_no_litellm_qwen36_27b_traits_taste_gpt4o_mini_candidate_summary_full": (
        run_agent4rec_qwen36_27b_traits_taste_candidate_summary_full
    ),
    "agent4rec_preference_yes_no_litellm_qwen3_8b_traits_taste_gpt4o_mini_no_summary_smoke": (
        run_agent4rec_qwen3_8b_traits_taste_no_summary_smoke
    ),
    "agent4rec_preference_yes_no_litellm_qwen3_8b_traits_taste_gpt4o_mini_no_summary_full": (
        run_agent4rec_qwen3_8b_traits_taste_no_summary_full
    ),
    "agent4rec_preference_yes_no_litellm_qwen36_27b_traits_taste_gpt4o_mini_no_summary_smoke": (
        run_agent4rec_qwen36_27b_traits_taste_no_summary_smoke
    ),
    "agent4rec_preference_yes_no_litellm_qwen36_27b_traits_taste_gpt4o_mini_no_summary_full": (
        run_agent4rec_qwen36_27b_traits_taste_no_summary_full
    ),
    "agent4rec_preference_listwise_ranking_vllm_qwen36_27b_traits_smoke": (
        run_agent4rec_listwise_qwen36_27b_traits_smoke
    ),
    "agent4rec_preference_listwise_ranking_vllm_qwen36_27b_traits_full": (
        run_agent4rec_listwise_qwen36_27b_traits_full
    ),
    "agent4rec_preference_listwise_ranking_litellm_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_smoke": (
        run_agent4rec_listwise_qwen3_8b_traits_taste_candidate_summary_smoke
    ),
    "agent4rec_preference_listwise_ranking_litellm_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_full": (
        run_agent4rec_listwise_qwen3_8b_traits_taste_candidate_summary_full
    ),
    "agent4rec_preference_listwise_ranking_litellm_qwen36_27b_traits_taste_gpt4o_mini_candidate_summary_smoke": (
        run_agent4rec_listwise_qwen36_27b_traits_taste_candidate_summary_smoke
    ),
    "agent4rec_preference_listwise_ranking_litellm_qwen36_27b_traits_taste_gpt4o_mini_candidate_summary_full": (
        run_agent4rec_listwise_qwen36_27b_traits_taste_candidate_summary_full
    ),
    "agent4rec_preference_listwise_ranking_litellm_qwen3_8b_traits_taste_gpt4o_mini_no_summary_smoke": (
        run_agent4rec_listwise_qwen3_8b_traits_taste_no_summary_smoke
    ),
    "agent4rec_preference_listwise_ranking_litellm_qwen3_8b_traits_taste_gpt4o_mini_no_summary_full": (
        run_agent4rec_listwise_qwen3_8b_traits_taste_no_summary_full
    ),
    "agent4rec_preference_listwise_ranking_litellm_qwen36_27b_traits_taste_gpt4o_mini_no_summary_smoke": (
        run_agent4rec_listwise_qwen36_27b_traits_taste_no_summary_smoke
    ),
    "agent4rec_preference_listwise_ranking_litellm_qwen36_27b_traits_taste_gpt4o_mini_no_summary_full": (
        run_agent4rec_listwise_qwen36_27b_traits_taste_no_summary_full
    ),
    "llm_preference_yes_no_litellm_qwen3_8b_smoke": run_qwen3_8b_smoke,
    "llm_preference_yes_no_litellm_qwen3_8b_full": run_qwen3_8b_full,
    "llm_preference_yes_no_litellm_qwen3_8b_summary_full": (
        run_qwen3_8b_summary_full
    ),
    "llm_preference_yes_no_litellm_qwen36_27b_smoke": run_qwen36_27b_smoke,
    "llm_preference_yes_no_litellm_qwen36_27b_full": run_qwen36_27b_full,
    "llm_preference_yes_no_litellm_qwen36_27b_summary_full": (
        run_qwen36_27b_summary_full
    ),
    "llm_preference_listwise_ranking_litellm_qwen3_8b_smoke": (
        run_listwise_qwen3_8b_smoke
    ),
    "llm_preference_listwise_ranking_litellm_qwen3_8b_full": (
        run_listwise_qwen3_8b_full
    ),
    "llm_preference_listwise_ranking_litellm_qwen36_27b_smoke": (
        run_listwise_qwen36_27b_smoke
    ),
    "llm_preference_listwise_ranking_litellm_qwen36_27b_full": (
        run_listwise_qwen36_27b_full
    ),
}
