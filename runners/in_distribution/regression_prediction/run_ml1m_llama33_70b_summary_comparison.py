from __future__ import annotations

"""Run ML-1M Llama-3.3-70B rating-regression with/without movie summaries."""

from runners.in_distribution.regression_prediction.run_ml1m_qwen3_8b_full import (
    main as run_ml1m_regression,
)


LLAMA33_70B_METHODS = (
    "llm_regressor_vllm_llama33_70b_with_item_stats_full",
    "llm_regressor_vllm_llama33_70b_with_item_stats_summary_full",
    "agent4rec_regressor_vllm_llama33_70b_traits_full",
    "agent4rec_regressor_vllm_llama33_70b_traits_summary_full",
    "agent4rec_regressor_vllm_llama33_70b_taste_gpt4o_mini_full",
    "agent4rec_regressor_vllm_llama33_70b_taste_gpt4o_mini_summary_full",
    "agent4rec_regressor_vllm_llama33_70b_traits_taste_gpt4o_mini_full",
    "agent4rec_regressor_vllm_llama33_70b_traits_taste_gpt4o_mini_summary_full",
)


def main() -> None:
    run_ml1m_regression(
        default_methods=LLAMA33_70B_METHODS,
        run_suffix="ml1m_llama33_70b_regression_summary_comparison",
    )


if __name__ == "__main__":
    main()
