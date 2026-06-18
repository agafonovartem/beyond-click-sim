from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from beyond_click_sim.tasks import Task

from runners.in_distribution.regression_prediction.methods.llm_regressor import (
    run_llama31_8b_full,
    run_llama31_8b_smoke,
    run_llama31_8b_with_item_stats_full,
    run_llama31_8b_with_item_stats_smoke,
    run_llama33_70b_full,
    run_llama33_70b_smoke,
    run_llama33_70b_with_item_stats_full,
    run_llama33_70b_with_item_stats_smoke,
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
    "user_mean_regressor": run_user_mean,
    "user_mode_regressor": run_user_mode,
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
}
