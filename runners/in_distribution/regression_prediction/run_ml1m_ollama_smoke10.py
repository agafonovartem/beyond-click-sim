from __future__ import annotations

"""Run a small ML-1M Ollama rating-regression smoke with saved predictions."""

from runners.in_distribution.regression_prediction.run_ml1m_qwen3_8b_full import (
    main as run_ml1m_regression,
)


OLLAMA_SMOKE10_METHODS = (
    "llm_regressor_ollama_llama31_8b_smoke10",
    "llm_regressor_ollama_llama31_8b_with_item_stats_smoke10",
    "llm_regressor_ollama_llama31_8b_with_item_stats_summary_smoke10",
)


def main() -> None:
    run_ml1m_regression(
        default_methods=OLLAMA_SMOKE10_METHODS,
        run_suffix="ml1m_ollama_llama31_8b_smoke10",
    )


if __name__ == "__main__":
    main()
