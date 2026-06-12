from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from beyond_click_sim.tasks import Task

from runners.in_distribution.regression_prediction.methods.history import (
    run_history_mean,
    run_history_mode,
)
from runners.in_distribution.regression_prediction.methods.llm_regressor import (
    run_llama31_8b_full,
    run_llama31_8b_smoke,
    run_llama33_70b_full,
    run_llama33_70b_smoke,
)
from runners.in_distribution.regression_prediction.methods.mean import run as run_mean
from runners.in_distribution.regression_prediction.methods.mode import run as run_mode


MethodRunner = Callable[[Task, Path], dict[str, object]]

METHOD_RUNNERS: dict[str, MethodRunner] = {
    "mean_regressor": run_mean,
    "mode_regressor": run_mode,
    "history_mean_regressor": run_history_mean,
    "history_mode_regressor": run_history_mode,
    "llm_regressor_ollama_llama31_8b_smoke": run_llama31_8b_smoke,
    "llm_regressor_ollama_llama31_8b_full": run_llama31_8b_full,
    "llm_regressor_vllm_llama33_70b_smoke": run_llama33_70b_smoke,
    "llm_regressor_vllm_llama33_70b_full": run_llama33_70b_full,
}
