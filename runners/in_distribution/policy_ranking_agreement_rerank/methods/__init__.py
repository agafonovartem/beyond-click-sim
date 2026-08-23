from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from beyond_click_sim.tasks import Task

from runners.in_distribution.policy_ranking_agreement_rerank.methods.agent4rec_yes_no import (
    run_llama33_70b_full as run_agent4rec_llama33_70b_full,
    run_llama33_70b_smoke as run_agent4rec_llama33_70b_smoke,
    run_llama33_70b_traits_taste_gpt4o_mini_full as run_agent4rec_llama33_70b_traits_taste_gpt4o_mini_full,
    run_llama33_70b_traits_taste_gpt4o_mini_smoke as run_agent4rec_llama33_70b_traits_taste_gpt4o_mini_smoke,
    run_qwen36_27b_full as run_agent4rec_qwen36_27b_full,
    run_qwen36_27b_itemwise_full as run_agent4rec_qwen36_27b_itemwise_full,
    run_qwen36_27b_itemwise_smoke as run_agent4rec_qwen36_27b_itemwise_smoke,
    run_qwen36_27b_smoke as run_agent4rec_qwen36_27b_smoke,
    run_qwen36_27b_traits_taste_gpt4o_mini_full as run_agent4rec_qwen36_27b_traits_taste_gpt4o_mini_full,
    run_qwen36_27b_traits_taste_gpt4o_mini_smoke as run_agent4rec_qwen36_27b_traits_taste_gpt4o_mini_smoke,
    run_qwen36_35b_a3b_full as run_agent4rec_qwen36_35b_a3b_full,
    run_qwen36_35b_a3b_smoke as run_agent4rec_qwen36_35b_a3b_smoke,
    run_qwen36_35b_a3b_traits_taste_gpt4o_mini_full as run_agent4rec_qwen36_35b_a3b_traits_taste_gpt4o_mini_full,
    run_qwen36_35b_a3b_traits_taste_gpt4o_mini_smoke as run_agent4rec_qwen36_35b_a3b_traits_taste_gpt4o_mini_smoke,
    run_qwen3_8b_full as run_agent4rec_qwen3_8b_full,
    run_qwen3_8b_smoke as run_agent4rec_qwen3_8b_smoke,
    run_qwen3_8b_traits_taste_gpt4o_mini_full as run_agent4rec_qwen3_8b_traits_taste_gpt4o_mini_full,
    run_qwen3_8b_traits_taste_gpt4o_mini_smoke as run_agent4rec_qwen3_8b_traits_taste_gpt4o_mini_smoke,
)
from runners.in_distribution.policy_ranking_agreement_rerank.methods.llm_yes_no import (
    run_llama33_70b_full,
    run_llama33_70b_itemwise_full,
    run_llama33_70b_itemwise_smoke,
    run_llama33_70b_smoke,
    run_qwen36_27b_full,
    run_qwen36_27b_itemwise_full,
    run_qwen36_27b_itemwise_smoke,
    run_qwen36_27b_smoke,
    run_qwen36_35b_a3b_full,
    run_qwen36_35b_a3b_itemwise_full,
    run_qwen36_35b_a3b_itemwise_smoke,
    run_qwen36_35b_a3b_smoke,
    run_qwen3_8b_full,
    run_qwen3_8b_itemwise_full,
    run_qwen3_8b_itemwise_smoke,
    run_qwen3_8b_smoke,
)
from runners.in_distribution.policy_ranking_agreement_rerank.methods.popularity import (
    run as run_popularity,
)


MethodRunner = Callable[[Task, Path], dict[str, object]]

# The 17 `_full` entries below mirror the method set ml-1m actually scored in
# `policy_ranking_agreement`, so the two sweeps cover a comparable roster: four
# vLLM backbones across four scorer variants, plus the non-LLM floor. The two
# setups are not directly comparable in their *results* -- different target,
# different candidate construction -- only in which methods were run.
#
# One addition over what ml-1m scored: `llm_yes_no_itemwise_vllm_qwen36_35b_a3b`,
# which the interaction sweep skipped but the preference sweep ran.
#
# `_smoke` twins cap the LLM group count and exist to validate wiring before a
# full sweep spends anything. `agent4rec_yes_no_itemwise_*` is new capability --
# no itemwise Agent4Rec exists elsewhere in the repo -- and is registered for
# qwen36_27b only, outside the parity set.
METHOD_RUNNERS: dict[str, MethodRunner] = {
    # --- non-LLM floor. Run first: if this already recovers the unbiased
    # ordering, no LLM result on this dataset means anything.
    "popularity_scorer": run_popularity,

    # --- history-conditioned yes/no, one call per (user, policy) slate
    "llm_yes_no_vllm_llama33_70b_smoke": run_llama33_70b_smoke,
    "llm_yes_no_vllm_llama33_70b_full": run_llama33_70b_full,
    "llm_yes_no_vllm_qwen36_27b_smoke": run_qwen36_27b_smoke,
    "llm_yes_no_vllm_qwen36_27b_full": run_qwen36_27b_full,
    "llm_yes_no_vllm_qwen36_35b_a3b_smoke": run_qwen36_35b_a3b_smoke,
    "llm_yes_no_vllm_qwen36_35b_a3b_full": run_qwen36_35b_a3b_full,
    "llm_yes_no_vllm_qwen3_8b_smoke": run_qwen3_8b_smoke,
    "llm_yes_no_vllm_qwen3_8b_full": run_qwen3_8b_full,

    # --- history-conditioned yes/no, one call per (user, item), deduped
    "llm_yes_no_itemwise_vllm_llama33_70b_smoke": run_llama33_70b_itemwise_smoke,
    "llm_yes_no_itemwise_vllm_llama33_70b_full": run_llama33_70b_itemwise_full,
    "llm_yes_no_itemwise_vllm_qwen36_27b_smoke": run_qwen36_27b_itemwise_smoke,
    "llm_yes_no_itemwise_vllm_qwen36_27b_full": run_qwen36_27b_itemwise_full,
    "llm_yes_no_itemwise_vllm_qwen36_35b_a3b_smoke": run_qwen36_35b_a3b_itemwise_smoke,
    "llm_yes_no_itemwise_vllm_qwen36_35b_a3b_full": run_qwen36_35b_a3b_itemwise_full,
    "llm_yes_no_itemwise_vllm_qwen3_8b_smoke": run_qwen3_8b_itemwise_smoke,
    "llm_yes_no_itemwise_vllm_qwen3_8b_full": run_qwen3_8b_itemwise_full,

    # --- Agent4Rec persona, traits only
    "agent4rec_yes_no_vllm_llama33_70b_smoke": run_agent4rec_llama33_70b_smoke,
    "agent4rec_yes_no_vllm_llama33_70b_full": run_agent4rec_llama33_70b_full,
    "agent4rec_yes_no_vllm_qwen36_27b_smoke": run_agent4rec_qwen36_27b_smoke,
    "agent4rec_yes_no_vllm_qwen36_27b_full": run_agent4rec_qwen36_27b_full,
    "agent4rec_yes_no_vllm_qwen36_35b_a3b_smoke": run_agent4rec_qwen36_35b_a3b_smoke,
    "agent4rec_yes_no_vllm_qwen36_35b_a3b_full": run_agent4rec_qwen36_35b_a3b_full,
    "agent4rec_yes_no_vllm_qwen3_8b_smoke": run_agent4rec_qwen3_8b_smoke,
    "agent4rec_yes_no_vllm_qwen3_8b_full": run_agent4rec_qwen3_8b_full,

    # --- Agent4Rec persona, traits + gpt-4o-mini-generated taste
    "agent4rec_yes_no_vllm_llama33_70b_traits_taste_gpt4o_mini_smoke": run_agent4rec_llama33_70b_traits_taste_gpt4o_mini_smoke,
    "agent4rec_yes_no_vllm_llama33_70b_traits_taste_gpt4o_mini_full": run_agent4rec_llama33_70b_traits_taste_gpt4o_mini_full,
    "agent4rec_yes_no_vllm_qwen36_27b_traits_taste_gpt4o_mini_smoke": run_agent4rec_qwen36_27b_traits_taste_gpt4o_mini_smoke,
    "agent4rec_yes_no_vllm_qwen36_27b_traits_taste_gpt4o_mini_full": run_agent4rec_qwen36_27b_traits_taste_gpt4o_mini_full,
    "agent4rec_yes_no_vllm_qwen36_35b_a3b_traits_taste_gpt4o_mini_smoke": run_agent4rec_qwen36_35b_a3b_traits_taste_gpt4o_mini_smoke,
    "agent4rec_yes_no_vllm_qwen36_35b_a3b_traits_taste_gpt4o_mini_full": run_agent4rec_qwen36_35b_a3b_traits_taste_gpt4o_mini_full,
    "agent4rec_yes_no_vllm_qwen3_8b_traits_taste_gpt4o_mini_smoke": run_agent4rec_qwen3_8b_traits_taste_gpt4o_mini_smoke,
    "agent4rec_yes_no_vllm_qwen3_8b_traits_taste_gpt4o_mini_full": run_agent4rec_qwen3_8b_traits_taste_gpt4o_mini_full,

    # --- outside the parity set: itemwise Agent4Rec is new capability
    "agent4rec_yes_no_itemwise_vllm_qwen36_27b_smoke": run_agent4rec_qwen36_27b_itemwise_smoke,
    "agent4rec_yes_no_itemwise_vllm_qwen36_27b_full": run_agent4rec_qwen36_27b_itemwise_full,
}
