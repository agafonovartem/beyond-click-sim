from __future__ import annotations

import pandas as pd
import pytest

from runners.ood.kuairand_policy_shift.metrics import (
    build_paired_comparisons,
    evaluate_predictions,
)


def test_ranking_excludes_all_zero_and_all_positive_groups() -> None:
    predictions = _prediction_frame(
        {
            "mixed": (1, 0, 0, 1, 0, 0, 0, 0, 0, 0),
            "all-zero": (0,) * 10,
            "all-positive": (1,) * 10,
        }
    )

    cells, per_group = evaluate_predictions(predictions)

    assert len(cells) == 1
    cell = cells[0]
    assert cell["n_groups"] == 3
    assert cell["n_mixed_groups"] == 1
    assert cell["n_all_zero_groups"] == 1
    assert cell["n_all_positive_groups"] == 1
    assert cell["n_successful_mixed_groups"] == 1
    assert set(per_group["label_composition"]) == {
        "mixed",
        "all_zero",
        "all_positive",
    }
    degenerate = per_group.loc[
        ~per_group["label_composition"].eq("mixed")
    ]
    assert degenerate["ndcg@5"].isna().all()
    assert degenerate["hit_rate@1"].isna().all()


def test_reports_ndcg5_ndcg10_hit1_and_hit5() -> None:
    predictions = _prediction_frame(
        {"mixed": (1, 0, 0, 0, 0, 1, 0, 0, 0, 0)}
    )

    cells, per_group = evaluate_predictions(predictions)

    for metric in ("ndcg@5", "ndcg@10", "hit_rate@1", "hit_rate@5"):
        assert metric in cells[0]
        assert metric in per_group
        assert 0.0 <= cells[0][metric] <= 1.0
    assert "hit_rate@10" not in cells[0]


def test_failed_mixed_group_is_zero_only_in_sensitivity_metric() -> None:
    predictions = _prediction_frame(
        {
            "successful": (1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            "failed": (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        }
    )
    predictions.loc[
        predictions["candidate_group"].eq("failed"),
        "score",
    ] = float("nan")

    cells, per_group = evaluate_predictions(predictions)

    cell = cells[0]
    assert cell["n_mixed_groups"] == 2
    assert cell["n_successful_mixed_groups"] == 1
    assert cell["n_failed_mixed_groups"] == 1
    assert cell["mixed_group_score_coverage"] == 0.5
    failed = per_group.loc[
        per_group["candidate_group"].eq("failed")
    ].iloc[0]
    assert pd.isna(failed["ndcg@5"])
    assert failed["failure_as_zero_ndcg@5"] == 0.0


def test_cell_reports_exact_user_denominators_and_seed_count_distributions() -> None:
    rows = []
    group_specs = {
        "u1": {
            0: ("mixed", False),
            1: ("mixed", False),
            2: ("mixed", False),
        },
        "u2": {
            0: ("mixed", False),
            1: ("mixed", True),
            2: ("all_zero", False),
        },
        "u3": {
            0: ("all_zero", False),
            1: ("all_positive", False),
            2: ("all_zero", False),
        },
        "u4": {
            0: ("mixed", True),
            1: ("all_zero", False),
            2: ("all_zero", False),
        },
    }
    labels = {
        "mixed": (1, 0),
        "all_zero": (0, 0),
        "all_positive": (1, 1),
    }
    for user_id, seed_specs in group_specs.items():
        for seed, (composition, failed) in seed_specs.items():
            rows.extend(
                _prediction_group(
                    user_id=user_id,
                    seed=seed,
                    labels=labels[composition],
                    failed=failed,
                )
            )

    cells, _ = evaluate_predictions(pd.DataFrame(rows))

    cell = cells[0]
    assert "n_users" not in cell
    assert cell["n_users_cohort"] == 4
    assert cell["n_users_with_mixed_group"] == 3
    assert cell["n_users_in_metric"] == 2
    assert cell["n_users_in_failure_as_zero_metric"] == 3
    assert cell["users_by_eligible_mixed_seed_count"] == {
        "0": 1,
        "1": 1,
        "2": 1,
        "3": 1,
    }
    assert cell["users_by_successful_mixed_seed_count"] == {
        "0": 2,
        "1": 1,
        "3": 1,
    }
    assert cell["ndcg@5"] == pytest.approx(1.0)
    assert cell["failure_as_zero_ndcg@5"] == pytest.approx(0.5)


def test_paired_comparison_averages_seeds_within_user() -> None:
    rows = []
    for user_id, seed_deltas in {
        "u1": (1.0, 1.0, 1.0),
        "u2": (-1.0, -1.0, -1.0),
    }.items():
        for seed, delta in enumerate(seed_deltas):
            rows.extend(
                [
                    _group_metric_row(
                        user_id=user_id,
                        seed=seed,
                        method=(
                            "llm_openai_vk_gpt55_listwise_english"
                        ),
                        history_control="observed",
                        value=0.5 + delta / 2,
                    ),
                    _group_metric_row(
                        user_id=user_id,
                        seed=seed,
                        method="popularity",
                        history_control="observed",
                        value=0.5 - delta / 2,
                    ),
                ]
            )
    comparisons = build_paired_comparisons(
        pd.DataFrame(rows),
        bootstrap_samples=200,
        bootstrap_seed=0,
    )

    comparison = next(
        row
        for row in comparisons
        if row["comparison"] == "gpt_history_minus_popularity"
        and row["metric"] == "ndcg@5"
    )
    assert comparison["n_users"] == 2
    assert comparison["n_seed_user_pairs"] == 6
    assert comparison["mean_delta"] == pytest.approx(0.0)
    assert (
        comparison["bootstrap_unit"]
        == "user_after_mean_over_candidate_seeds"
    )


def _prediction_frame(
    groups: dict[str, tuple[int, ...]],
) -> pd.DataFrame:
    rows = []
    for group_name, labels in groups.items():
        for position, label in enumerate(labels):
            rows.append(
                {
                    "method": "popularity",
                    "task_seed": 0,
                    "split": "test",
                    "target_pool": "common_warm",
                    "target_policy": "random",
                    "history_control": "source",
                    "user_id": f"user-{group_name}",
                    "candidate_group": group_name,
                    "item_id": f"{group_name}-{position}",
                    "target": label,
                    "score": float(len(labels) - position),
                }
            )
    return pd.DataFrame(rows)


def _prediction_group(
    *,
    user_id: str,
    seed: int,
    labels: tuple[int, ...],
    failed: bool,
) -> list[dict[str, object]]:
    group = f"test-random-{seed}-{user_id}"
    return [
        {
            "method": "popularity",
            "task_seed": seed,
            "split": "test",
            "target_pool": "common_warm",
            "target_policy": "random",
            "history_control": "observed",
            "user_id": user_id,
            "candidate_group": group,
            "item_id": f"{group}-{position}",
            "target": label,
            "score": (
                float("nan")
                if failed
                else float(len(labels) - position)
            ),
        }
        for position, label in enumerate(labels)
    ]


def _group_metric_row(
    *,
    user_id: str,
    seed: int,
    method: str,
    history_control: str,
    value: float,
) -> dict[str, object]:
    return {
        "method": method,
        "task_seed": seed,
        "split": "test",
        "target_pool": "common_warm",
        "target_policy": "random",
        "history_control": history_control,
        "user_id": user_id,
        "candidate_group": f"test-random-{seed}-{user_id}",
        "ranking_eligible": True,
        "failed": False,
        "ndcg@5": value,
        "hit_rate@1": value,
        "ndcg@10": value,
        "hit_rate@5": value,
    }
