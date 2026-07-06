from __future__ import annotations

"""Run the reduced ML-1M Qwen3-8B rating-regression summary ablation."""

from runners.in_distribution.regression_prediction.run_ml1m_qwen3_8b_full import (
    main as run_ml1m_qwen3_8b,
)


SUMMARY_METHODS = (
    "llm_regressor_vllm_qwen3_8b_with_item_stats_summary_full",
    "agent4rec_regressor_vllm_qwen3_8b_traits_summary_full",
    "agent4rec_regressor_vllm_qwen3_8b_taste_gpt4o_mini_summary_full",
    "agent4rec_regressor_vllm_qwen3_8b_traits_taste_gpt4o_mini_summary_full",
)


def main() -> None:
    run_ml1m_qwen3_8b(
        default_methods=SUMMARY_METHODS,
        run_suffix="ml1m_qwen3_8b_regression_summary_ablation",
    )


if __name__ == "__main__":
    main()
