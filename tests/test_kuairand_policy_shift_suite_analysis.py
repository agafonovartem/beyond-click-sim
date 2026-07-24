from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.ood.kuairand_policy_shift.suite_analysis import (
    CAPTION_METHOD,
    CATEGORIES_METHOD,
    ITEM_CTR_METHOD,
    PRIMARY_COMPARISON_IDS,
    TEXT_CAPTION_METHOD,
    TEXT_CATEGORIES_METHOD,
    analyze_suite,
    holm_adjust,
    run_suite_analysis,
)


def test_holm_adjust_is_step_down_and_preserves_original_order() -> None:
    adjusted = holm_adjust([0.04, 0.01, 0.03])

    assert adjusted == pytest.approx([0.06, 0.03, 0.06])


def test_suite_analysis_builds_all_requested_seed_averaged_contrasts() -> None:
    caption = _method_frame(
        {
            ("standard", "standard", "observed"): 0.80,
            ("random", "standard", "observed"): 0.70,
            ("none", "standard", "none"): 0.60,
            ("standard", "standard", "permuted"): 0.65,
            ("random", "standard", "permuted"): 0.64,
            ("standard", "random", "observed"): 0.75,
            ("random", "random", "observed"): 0.78,
            ("none", "random", "none"): 0.62,
            ("standard", "random", "permuted"): 0.63,
            ("random", "random", "permuted"): 0.66,
        }
    )
    categories = _method_frame(
        {
            ("standard", "standard", "observed"): 0.76,
            ("random", "standard", "observed"): 0.67,
            ("none", "standard", "none"): 0.58,
            ("standard", "random", "observed"): 0.70,
            ("random", "random", "observed"): 0.74,
            ("none", "random", "none"): 0.59,
        }
    )
    item_ctr = _method_frame(
        {
            ("standard", "standard", "observed"): 0.82,
            ("random", "standard", "observed"): 0.68,
            ("standard", "random", "observed"): 0.71,
            ("random", "random", "observed"): 0.83,
        }
    )

    records = analyze_suite(
        caption=caption,
        categories=categories,
        item_ctr=item_ctr,
        metric="ndcg@3",
        bootstrap_samples=100,
        sign_flip_samples=100,
        seed=7,
    )
    by_id = {record["comparison_id"]: record for record in records}

    assert len(records) == 22
    assert {row["comparison_id"] for row in records if row["primary"]} == set(
        PRIMARY_COMPARISON_IDS
    )
    assert by_id["caption_minus_categories_standard_to_random"][
        "mean_delta"
    ] == pytest.approx(0.05)
    assert by_id["caption_gpt_minus_item_ctr_random_to_standard"][
        "mean_delta"
    ] == pytest.approx(0.02)
    # (0.80 - 0.70) - (0.82 - 0.68) = -0.04: GPT is more robust.
    assert by_id[
        "transfer_gap_difference_gpt_minus_item_ctr_target_standard"
    ]["mean_delta"] == pytest.approx(-0.04)
    assert by_id[
        "caption_gpt_transferred_minus_no_history_target_random"
    ]["mean_delta"] == pytest.approx(0.13)
    assert all(record["n_users"] == 3 for record in records)
    assert all(record["n_seed_user_pairs"] == 6 for record in records)
    for record in records:
        if record["primary"]:
            assert record["holm_adjusted_p"] >= record["raw_p_value"]
            assert record["holm_family"] == "primary_transfer"
        else:
            assert record["holm_adjusted_p"] is None


def test_suite_analysis_rejects_unpaired_candidate_rows() -> None:
    values = {
        ("standard", "standard", "observed"): 0.8,
        ("random", "standard", "observed"): 0.7,
        ("none", "standard", "none"): 0.6,
        ("standard", "standard", "permuted"): 0.65,
        ("random", "standard", "permuted"): 0.64,
        ("standard", "random", "observed"): 0.75,
        ("random", "random", "observed"): 0.78,
        ("none", "random", "none"): 0.62,
        ("standard", "random", "permuted"): 0.63,
        ("random", "random", "permuted"): 0.66,
    }
    caption = _method_frame(values)
    categories = _method_frame(
        {
            key: value
            for key, value in values.items()
            if key[2] in {"observed", "none"}
        }
    )
    item_ctr = _method_frame(
        {key: value for key, value in values.items() if key[2] == "observed"}
    )
    categories = categories[
        ~(
            categories["task_seed"].eq(42)
            & categories["user_id"].eq("u1")
            & categories["history_policy"].eq("standard")
            & categories["target_policy"].eq("standard")
        )
    ]

    with pytest.raises(ValueError, match="Unpaired rows"):
        analyze_suite(
            caption=caption,
            categories=categories,
            item_ctr=item_ctr,
            metric="ndcg@3",
            bootstrap_samples=10,
            sign_flip_samples=10,
        )


def test_suite_analysis_allows_permutation_control_cohort_subset() -> None:
    values = {
        ("standard", "standard", "observed"): 0.8,
        ("random", "standard", "observed"): 0.7,
        ("none", "standard", "none"): 0.6,
        ("standard", "standard", "permuted"): 0.65,
        ("random", "standard", "permuted"): 0.64,
        ("standard", "random", "observed"): 0.75,
        ("random", "random", "observed"): 0.78,
        ("none", "random", "none"): 0.62,
        ("standard", "random", "permuted"): 0.63,
        ("random", "random", "permuted"): 0.66,
    }
    caption = _method_frame(values)
    is_permuted_u1 = caption["history_control"].ne("permuted") | caption[
        "user_id"
    ].eq("u1")
    caption = caption.loc[is_permuted_u1].copy()
    item_ctr = caption.loc[
        caption["history_control"].eq("observed")
    ].copy()

    records = analyze_suite(
        caption=caption,
        categories=None,
        item_ctr=item_ctr,
        metric="ndcg@3",
        bootstrap_samples=10,
        sign_flip_samples=10,
    )
    grounding = [
        record
        for record in records
        if "original_minus_permuted" in record["comparison_id"]
    ]

    assert len(grounding) == 4
    assert all(record["n_users"] == 1 for record in grounding)
    assert all(record["n_seed_user_pairs"] == 2 for record in grounding)


def test_run_suite_analysis_validates_targets_and_writes_all_formats(
    tmp_path: Path,
) -> None:
    caption = _method_frame(
        {
            ("standard", "standard", "observed"): 0.80,
            ("random", "standard", "observed"): 0.70,
            ("none", "standard", "none"): 0.60,
            ("standard", "standard", "permuted"): 0.65,
            ("random", "standard", "permuted"): 0.64,
            ("standard", "random", "observed"): 0.75,
            ("random", "random", "observed"): 0.78,
            ("none", "random", "none"): 0.62,
            ("standard", "random", "permuted"): 0.63,
            ("random", "random", "permuted"): 0.66,
        }
    )
    categories = caption[
        caption["history_control"].isin(["observed", "none"])
    ].copy()
    item_ctr = caption[caption["history_control"].eq("observed")].copy()
    gpt_root = tmp_path / "gpt"
    classical_root = tmp_path / "classical"
    text_root = tmp_path / "text"
    _write_method_artifact(gpt_root / CAPTION_METHOD, CAPTION_METHOD, caption)
    _write_method_artifact(
        gpt_root / CATEGORIES_METHOD,
        CATEGORIES_METHOD,
        categories,
    )
    _write_method_artifact(
        classical_root / ITEM_CTR_METHOD,
        ITEM_CTR_METHOD,
        item_ctr,
    )
    _write_method_artifact(
        text_root / TEXT_CAPTION_METHOD,
        TEXT_CAPTION_METHOD,
        caption[caption["history_control"].eq("observed")].copy(),
    )
    _write_method_artifact(
        text_root / TEXT_CATEGORIES_METHOD,
        TEXT_CATEGORIES_METHOD,
        categories[categories["history_control"].eq("observed")].copy(),
    )
    output_dir = tmp_path / "analysis"

    result = run_suite_analysis(
        gpt_run_dir=gpt_root,
        classical_run_dir=classical_root,
        output_dir=output_dir,
        text_run_dir=text_root,
        metric="ndcg@3",
        bootstrap_samples=10,
        sign_flip_samples=10,
    )

    assert result["input_validation"]["exact_target_rows_match"] is True
    assert result["input_validation"]["canonical_target_rows"] == 36
    semantic = [
        row
        for row in result["comparisons"]
        if row["analysis_family"].startswith("gpt_vs_text_affinity")
        or row["analysis_family"] == "caption_gpt_vs_text_transfer_gap"
    ]
    assert len(result["comparisons"]) == 32
    assert len(semantic) == 10
    assert all(row["primary"] is False for row in semantic)
    assert all(row["holm_adjusted_p"] is None for row in semantic)
    assert result["statistical_protocol"]["primary_family_size"] == 8
    assert result["semantic_baseline_diagnostics"]["included"] is True
    assert "text_caption" in result["inputs"]
    assert "text_categories" in result["inputs"]
    assert (output_dir / "suite_analysis.json").exists()
    assert (output_dir / "suite_comparisons.csv").exists()
    report = (output_dir / "suite_report.md").read_text()
    assert "not a preregistered confirmatory analysis" in report
    assert "Difference in transfer penalties" in report
    assert "Caption GPT versus caption text-affinity" in report


def test_run_suite_analysis_auto_skips_missing_categories_artifact(
    tmp_path: Path,
) -> None:
    caption = _method_frame(
        {
            ("standard", "standard", "observed"): 0.80,
            ("random", "standard", "observed"): 0.70,
            ("none", "standard", "none"): 0.60,
            ("standard", "standard", "permuted"): 0.65,
            ("random", "standard", "permuted"): 0.64,
            ("standard", "random", "observed"): 0.75,
            ("random", "random", "observed"): 0.78,
            ("none", "random", "none"): 0.62,
            ("standard", "random", "permuted"): 0.63,
            ("random", "random", "permuted"): 0.66,
        }
    )
    item_ctr = caption[caption["history_control"].eq("observed")].copy()
    gpt_root = tmp_path / "gpt"
    classical_root = tmp_path / "classical"
    _write_method_artifact(gpt_root / CAPTION_METHOD, CAPTION_METHOD, caption)
    _write_method_artifact(
        classical_root / ITEM_CTR_METHOD,
        ITEM_CTR_METHOD,
        item_ctr,
    )

    result = run_suite_analysis(
        gpt_run_dir=gpt_root,
        classical_run_dir=classical_root,
        output_dir=tmp_path / "analysis",
        metric="ndcg@3",
        bootstrap_samples=10,
        sign_flip_samples=10,
    )

    assert result["caption_ablation"] == {
        "included": False,
        "skip_requested": False,
        "categories_artifact_found": False,
        "omission_reason": "categories_artifact_missing",
    }
    assert len(result["comparisons"]) == 16
    assert result["statistical_protocol"]["primary_family_size"] == 6
    assert "categories_gpt" not in result["inputs"]
    assert not any(
        row["analysis_family"] == "metadata_ablation"
        for row in result["comparisons"]
    )


def test_suite_analysis_pairs_only_natural_metric_eligible_groups() -> None:
    caption = _method_frame(
        {
            ("standard", "standard", "observed"): 0.80,
            ("random", "standard", "observed"): 0.70,
            ("none", "standard", "none"): 0.60,
            ("standard", "standard", "permuted"): 0.65,
            ("random", "standard", "permuted"): 0.64,
            ("standard", "random", "observed"): 0.75,
            ("random", "random", "observed"): 0.78,
            ("none", "random", "none"): 0.62,
            ("standard", "random", "permuted"): 0.63,
            ("random", "random", "permuted"): 0.66,
        }
    ).rename(columns={"ndcg@3": "ndcg@5"})
    caption["metric_eligible"] = ~(
        caption["user_id"].eq("u3") & caption["task_seed"].eq(43)
    )
    caption.loc[~caption["metric_eligible"], "ndcg@5"] = pd.NA
    item_ctr = caption.loc[
        caption["history_control"].eq("observed")
    ].copy()

    records = analyze_suite(
        caption=caption,
        categories=None,
        item_ctr=item_ctr,
        metric="ndcg@5",
        bootstrap_samples=10,
        sign_flip_samples=10,
    )

    assert records
    assert all(record["n_users"] == 3 for record in records)
    assert all(record["n_seed_user_pairs"] == 5 for record in records)
    assert all(
        record["metric_eligible_seeds_per_user_min"] == 1
        for record in records
    )
    assert all(
        record["metric_eligible_seeds_per_user_max"] == 2
        for record in records
    )
    assert all(
        record["users_with_all_metric_eligible_seeds"] == 2
        for record in records
    )


def test_lower_is_better_metrics_are_reported_as_quality_advantages() -> None:
    caption = _method_frame(
        {
            ("standard", "standard", "observed"): 0.40,
            ("random", "standard", "observed"): 0.50,
            ("none", "standard", "none"): 0.60,
            ("standard", "standard", "permuted"): 0.55,
            ("random", "standard", "permuted"): 0.56,
            ("standard", "random", "observed"): 0.45,
            ("random", "random", "observed"): 0.42,
            ("none", "random", "none"): 0.62,
            ("standard", "random", "permuted"): 0.57,
            ("random", "random", "permuted"): 0.54,
        }
    ).rename(columns={"ndcg@3": "log_loss"})
    item_ctr = _method_frame(
        {
            ("standard", "standard", "observed"): 0.48,
            ("random", "standard", "observed"): 0.54,
            ("standard", "random", "observed"): 0.52,
            ("random", "random", "observed"): 0.47,
        }
    ).rename(columns={"ndcg@3": "log_loss"})

    records = analyze_suite(
        caption=caption,
        categories=None,
        item_ctr=item_ctr,
        metric="log_loss",
        bootstrap_samples=10,
        sign_flip_samples=10,
    )
    by_id = {record["comparison_id"]: record for record in records}

    # Raw loss difference is .50 - .54 = -.04, but GPT has lower loss and the
    # reported quality-normalized advantage is therefore positive.
    comparison = by_id[
        "caption_gpt_minus_item_ctr_random_to_standard"
    ]
    assert comparison["mean_delta"] == pytest.approx(0.04)
    assert comparison["metric_higher_is_better"] is False
    assert comparison["quality_direction_multiplier"] == -1.0


def test_exposure_logistic_is_added_outside_primary_family() -> None:
    caption = _method_frame(
        {
            ("standard", "standard", "observed"): 0.40,
            ("random", "standard", "observed"): 0.50,
            ("none", "standard", "none"): 0.60,
            ("standard", "standard", "permuted"): 0.55,
            ("random", "standard", "permuted"): 0.56,
            ("standard", "random", "observed"): 0.45,
            ("random", "random", "observed"): 0.42,
            ("none", "random", "none"): 0.62,
            ("standard", "random", "permuted"): 0.57,
            ("random", "random", "permuted"): 0.54,
        }
    ).rename(columns={"ndcg@3": "log_loss"})
    classical = _method_frame(
        {
            ("standard", "standard", "observed"): 0.48,
            ("random", "standard", "observed"): 0.54,
            ("standard", "random", "observed"): 0.52,
            ("random", "random", "observed"): 0.47,
        }
    ).rename(columns={"ndcg@3": "log_loss"})

    records = analyze_suite(
        caption=caption,
        categories=None,
        item_ctr=classical,
        exposure_logistic=classical,
        metric="log_loss",
        bootstrap_samples=10,
        sign_flip_samples=10,
    )
    exposure_records = [
        record
        for record in records
        if "exposure_logistic" in record["comparison_id"]
    ]

    assert len(records) == 22
    assert len(exposure_records) == 6
    assert all(record["primary"] is False for record in exposure_records)
    assert all(
        record["holm_adjusted_p"] is None for record in exposure_records
    )


def _method_frame(
    cell_values: dict[tuple[str, str, str], float],
) -> pd.DataFrame:
    rows = []
    user_offsets = {"u1": -0.01, "u2": 0.0, "u3": 0.01}
    for task_seed in (42, 43):
        seed_offset = (task_seed - 42) * 0.002
        for user_id, user_offset in user_offsets.items():
            for (
                history_policy,
                target_policy,
                history_control,
            ), value in cell_values.items():
                rows.append(
                    {
                        "task_seed": task_seed,
                        "user_id": user_id,
                        "candidate_group": (
                            f"test_{target_policy}_{user_id}_c3"
                        ),
                        "history_policy": history_policy,
                        "target_policy": target_policy,
                        "history_control": history_control,
                        "ndcg@3": value + seed_offset + user_offset,
                    }
                )
    return pd.DataFrame(rows)


def _write_method_artifact(
    root: Path,
    method: str,
    per_user: pd.DataFrame,
) -> None:
    root.mkdir(parents=True)
    manifest = {
        "status": "complete",
        "method": {"name": method},
        "task_seeds": [42, 43],
        "cohort": {
            "n_users": 3,
            "user_ids_sha256": "same-cohort",
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    per_user.to_parquet(root / "per_user_metrics.parquet", index=False)
    predictions = []
    for row in per_user.itertuples(index=False):
        for position, target in enumerate((1, 0, 0), start=1):
            predictions.append(
                {
                    "task_seed": row.task_seed,
                    "target_policy": row.target_policy,
                    "user_id": row.user_id,
                    "candidate_group": row.candidate_group,
                    "item_id": (
                        f"{row.task_seed}-{row.target_policy}-"
                        f"{row.user_id}-c{position}"
                    ),
                    "target": target,
                }
            )
    pd.DataFrame(predictions).to_parquet(
        root / "predictions.parquet",
        index=False,
    )
