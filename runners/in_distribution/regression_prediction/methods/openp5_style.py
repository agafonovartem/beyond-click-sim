from __future__ import annotations

from pathlib import Path

from beyond_click_sim.tasks import Task
from runners.in_distribution.regression_prediction.methods.llm_regressor import (
    run_method,
)


SMOKE_ROWS = 5
PROMPT_FAMILY = "openp5_style"
OLLAMA_CLIENT = "ollama_local"
OLLAMA_MODEL = "qwen3:30b-a3b-instruct-2507-q4_K_M"
OLLAMA_MAX_WORKERS = 1


def run_ollama_qwen3_30b_a3b_smoke5(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    """Smoke-test the neutral OpenP5-style rating-regression prompt."""

    return run_method(
        task,
        output_dir,
        method_name=(
            "llm_regressor_openp5_style_ollama_qwen3_30b_a3b_smoke5"
        ),
        client_name=OLLAMA_CLIENT,
        model=OLLAMA_MODEL,
        max_rows=SMOKE_ROWS,
        max_workers=OLLAMA_MAX_WORKERS,
        prompt_family=PROMPT_FAMILY,
    )
