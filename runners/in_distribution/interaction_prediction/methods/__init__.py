from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from beyond_click_sim.tasks import Task

from runners.in_distribution.interaction_prediction.methods.llm_yes_no import (
    run_llama31_8b_full,
    run_llama31_8b_smoke,
    run_llama33_70b_full,
    run_llama33_70b_smoke,
)
from runners.in_distribution.interaction_prediction.methods.popularity import (
    run as run_popularity,
    run_ranking as run_popularity_ranking,
)


MethodRunner = Callable[[Task, Path], dict[str, object]]

METHOD_RUNNERS: dict[str, MethodRunner] = {
    "popularity_f1_threshold": run_popularity,
    "popularity_ranking": run_popularity_ranking,
    "llm_yes_no_ollama_llama31_8b_smoke": run_llama31_8b_smoke,
    "llm_yes_no_ollama_llama31_8b_full": run_llama31_8b_full,
    "llm_yes_no_vllm_llama33_70b_smoke": run_llama33_70b_smoke,
    "llm_yes_no_vllm_llama33_70b_full": run_llama33_70b_full,
}
