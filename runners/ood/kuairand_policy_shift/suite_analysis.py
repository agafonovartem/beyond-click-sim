from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CAPTION_METHOD = "llm_openai_vk_gpt54_mini_probability_caption_categories"
CATEGORIES_METHOD = "llm_openai_vk_gpt54_mini_probability_categories_only"
ITEM_CTR_METHOD = "item_ctr"
EXPOSURE_LOGISTIC_METHOD = "exposure_logistic_bias"
TEXT_CAPTION_METHOD = "text_affinity_caption_categories"
TEXT_CATEGORIES_METHOD = "text_affinity_categories_only"
PAIR_KEYS = ("task_seed", "user_id", "candidate_group")
CELL_COLUMNS = ("history_policy", "target_policy", "history_control")
DEFAULT_METRIC = "log_loss"
DEFAULT_BOOTSTRAP_SAMPLES = 20_000
DEFAULT_SIGN_FLIP_SAMPLES = 50_000
DEFAULT_SEED = 20260724


@dataclass(frozen=True)
class ContrastTerm:
    source: str
    selector: Mapping[str, str]
    coefficient: float
    label: str


@dataclass(frozen=True)
class ContrastSpec:
    comparison_id: str
    analysis_family: str
    target_policy: str
    estimand: str
    positive_meaning: str
    terms: tuple[ContrastTerm, ...]


@dataclass(frozen=True)
class MethodArtifact:
    name: str
    root: Path
    manifest: dict[str, Any]
    per_user: pd.DataFrame
    predictions: pd.DataFrame


# The primary family is intentionally limited to the central transfer question.
# Matched-domain cells, all-cell metadata ablations, and permutation controls are
# still reported as diagnostics. This family is explicit and reproducible, but
# it was written after the experiment outputs existed and is not a preregistration.
CORE_PRIMARY_COMPARISON_IDS = frozenset(
    {
        "caption_gpt_minus_item_ctr_standard_to_random",
        "caption_gpt_minus_item_ctr_random_to_standard",
        "transfer_gap_difference_gpt_minus_item_ctr_target_standard",
        "transfer_gap_difference_gpt_minus_item_ctr_target_random",
        "caption_gpt_transferred_minus_no_history_target_standard",
        "caption_gpt_transferred_minus_no_history_target_random",
    }
)
CAPTION_ABLATION_PRIMARY_COMPARISON_IDS = frozenset(
    {
        "caption_minus_categories_standard_to_random",
        "caption_minus_categories_random_to_standard",
    }
)
PRIMARY_COMPARISON_IDS = frozenset(
    CORE_PRIMARY_COMPARISON_IDS | CAPTION_ABLATION_PRIMARY_COMPARISON_IDS
)


def analyze_suite(
    *,
    caption: pd.DataFrame,
    categories: pd.DataFrame | None,
    item_ctr: pd.DataFrame,
    exposure_logistic: pd.DataFrame | None = None,
    text_caption: pd.DataFrame | None = None,
    text_categories: pd.DataFrame | None = None,
    metric: str = DEFAULT_METRIC,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    sign_flip_samples: int = DEFAULT_SIGN_FLIP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Compute all code-declared cross-method and within-GPT contrasts.

    Every contrast first aligns the exact ``task_seed × user × candidate_group``
    rows, then averages the three candidate-resample seeds within a user. The
    bootstrap and sign-flip test both operate on those per-user values, so seeds
    are never treated as independent users.
    """

    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if sign_flip_samples <= 0:
        raise ValueError("sign_flip_samples must be positive")
    frames = {
        "caption_gpt": caption,
        "item_ctr": item_ctr,
    }
    if exposure_logistic is not None:
        frames["exposure_logistic"] = exposure_logistic
    if categories is not None:
        frames["categories_gpt"] = categories
    if text_caption is not None:
        frames["text_caption"] = text_caption
    if text_categories is not None:
        if categories is None:
            raise ValueError(
                "text_categories requires categories GPT for matched-metadata "
                "comparisons"
            )
        frames["text_categories"] = text_categories
    for name, frame in frames.items():
        _validate_per_user_frame(frame, name=name, metric=metric)

    include_caption_ablation = categories is not None
    primary_ids = primary_comparison_ids(
        include_caption_ablation=include_caption_ablation
    )
    records: list[dict[str, Any]] = []
    for spec in build_contrast_specs(
        include_caption_ablation=include_caption_ablation,
        include_exposure_logistic=exposure_logistic is not None,
        include_text_caption=text_caption is not None,
        include_text_categories=text_categories is not None,
    ):
        contrast_seed = _derived_seed(seed, spec.comparison_id)
        record = _estimate_contrast(
            frames,
            spec,
            metric=metric,
            higher_is_better=_metric_higher_is_better(metric),
            bootstrap_samples=bootstrap_samples,
            sign_flip_samples=sign_flip_samples,
            seed=contrast_seed,
        )
        record["primary"] = spec.comparison_id in primary_ids
        records.append(record)

    _apply_primary_holm(records, primary_ids=primary_ids)
    return records


def primary_comparison_ids(
    *,
    include_caption_ablation: bool,
) -> frozenset[str]:
    if include_caption_ablation:
        return PRIMARY_COMPARISON_IDS
    return CORE_PRIMARY_COMPARISON_IDS


def build_contrast_specs(
    *,
    include_caption_ablation: bool = True,
    include_exposure_logistic: bool = False,
    include_text_caption: bool = False,
    include_text_categories: bool = False,
) -> list[ContrastSpec]:
    """Return the complete, deterministic analysis specification."""

    specs: list[ContrastSpec] = []
    policies = ("standard", "random")

    # Caption ablation: four observed cells plus two no-history cells.
    if include_caption_ablation:
        for target_policy in policies:
            for history_policy in policies:
                specs.append(
                    _simple_spec(
                        comparison_id=(
                            "caption_minus_categories_"
                            f"{history_policy}_to_{target_policy}"
                        ),
                        family="metadata_ablation",
                        target_policy=target_policy,
                        estimand=(
                            "caption+categories GPT minus categories-only GPT "
                            f"for {history_policy} history"
                        ),
                        positive_meaning="caption+categories GPT is better",
                        left_source="caption_gpt",
                        right_source="categories_gpt",
                        left_selector=_observed(history_policy, target_policy),
                        right_selector=_observed(history_policy, target_policy),
                        left_label="caption+categories GPT",
                        right_label="categories-only GPT",
                    )
                )
            specs.append(
                _simple_spec(
                    comparison_id=(
                        "caption_minus_categories_no_history_target_"
                        f"{target_policy}"
                    ),
                    family="metadata_ablation",
                    target_policy=target_policy,
                    estimand=(
                        "caption+categories GPT minus categories-only GPT "
                        "without history"
                    ),
                    positive_meaning="caption+categories GPT is better",
                    left_source="caption_gpt",
                    right_source="categories_gpt",
                    left_selector=_no_history(target_policy),
                    right_selector=_no_history(target_policy),
                    left_label="caption+categories GPT, no history",
                    right_label="categories-only GPT, no history",
                )
            )

    # Caption GPT against the predeclared ItemCTR comparator in all 2x2 cells.
    for target_policy in policies:
        for history_policy in policies:
            specs.append(
                _simple_spec(
                    comparison_id=(
                        "caption_gpt_minus_item_ctr_"
                        f"{history_policy}_to_{target_policy}"
                    ),
                    family="gpt_vs_item_ctr",
                    target_policy=target_policy,
                    estimand=(
                        "caption GPT minus source-policy ItemCTR "
                        f"for {history_policy} history"
                    ),
                    positive_meaning="caption GPT is better",
                    left_source="caption_gpt",
                    right_source="item_ctr",
                    left_selector=_observed(history_policy, target_policy),
                    right_selector=_observed(history_policy, target_policy),
                    left_label="caption GPT",
                    right_label="ItemCTR",
                )
            )

    # Difference in transfer penalties. A negative value means GPT loses less
    # under source-policy mismatch and is therefore more robust than ItemCTR.
    for target_policy in policies:
        transferred_policy = _other_policy(target_policy)
        specs.append(
            ContrastSpec(
                comparison_id=(
                    "transfer_gap_difference_gpt_minus_item_ctr_target_"
                    f"{target_policy}"
                ),
                analysis_family="difference_in_transfer_gaps",
                target_policy=target_policy,
                estimand=(
                    "(GPT matched - GPT transferred) - "
                    "(ItemCTR matched - ItemCTR transferred)"
                ),
                positive_meaning=(
                    "GPT has a larger transfer penalty; negative means GPT is "
                    "more robust to source-policy mismatch"
                ),
                terms=(
                    ContrastTerm(
                        "caption_gpt",
                        _observed(target_policy, target_policy),
                        1.0,
                        "GPT matched",
                    ),
                    ContrastTerm(
                        "caption_gpt",
                        _observed(transferred_policy, target_policy),
                        -1.0,
                        "GPT transferred",
                    ),
                    ContrastTerm(
                        "item_ctr",
                        _observed(target_policy, target_policy),
                        -1.0,
                        "ItemCTR matched",
                    ),
                    ContrastTerm(
                        "item_ctr",
                        _observed(transferred_policy, target_policy),
                        1.0,
                        "ItemCTR transferred",
                    ),
                ),
            )
        )

    # Exposure-aware logistic user/item effects are a stronger probability
    # baseline than ItemCTR in this task. They are reported as an additional
    # diagnostic and do not alter the already-declared primary Holm family.
    if include_exposure_logistic:
        for target_policy in policies:
            for history_policy in policies:
                specs.append(
                    _simple_spec(
                        comparison_id=(
                            "caption_gpt_minus_exposure_logistic_"
                            f"{history_policy}_to_{target_policy}"
                        ),
                        family="gpt_vs_exposure_logistic",
                        target_policy=target_policy,
                        estimand=(
                            "caption GPT minus exposure-aware logistic "
                            f"baseline for {history_policy} history"
                        ),
                        positive_meaning="caption GPT is better",
                        left_source="caption_gpt",
                        right_source="exposure_logistic",
                        left_selector=_observed(
                            history_policy, target_policy
                        ),
                        right_selector=_observed(
                            history_policy, target_policy
                        ),
                        left_label="caption GPT",
                        right_label="exposure-aware logistic",
                    )
                )
        for target_policy in policies:
            transferred_policy = _other_policy(target_policy)
            specs.append(
                ContrastSpec(
                    comparison_id=(
                        "transfer_gap_difference_gpt_minus_"
                        f"exposure_logistic_target_{target_policy}"
                    ),
                    analysis_family=(
                        "difference_in_transfer_gaps_exposure_logistic"
                    ),
                    target_policy=target_policy,
                    estimand=(
                        "(GPT matched - GPT transferred) - "
                        "(exposure logistic matched - exposure logistic "
                        "transferred)"
                    ),
                    positive_meaning=(
                        "GPT has a larger transfer penalty; negative means "
                        "GPT is more robust to source-policy mismatch"
                    ),
                    terms=(
                        ContrastTerm(
                            "caption_gpt",
                            _observed(target_policy, target_policy),
                            1.0,
                            "GPT matched",
                        ),
                        ContrastTerm(
                            "caption_gpt",
                            _observed(
                                transferred_policy, target_policy
                            ),
                            -1.0,
                            "GPT transferred",
                        ),
                        ContrastTerm(
                            "exposure_logistic",
                            _observed(target_policy, target_policy),
                            -1.0,
                            "exposure logistic matched",
                        ),
                        ContrastTerm(
                            "exposure_logistic",
                            _observed(
                                transferred_policy, target_policy
                            ),
                            1.0,
                            "exposure logistic transferred",
                        ),
                    ),
                )
            )

    # Post-hoc matched-metadata semantic baselines. These diagnostics are never
    # added to the existing primary Holm family.
    if include_text_caption:
        specs.extend(
            _gpt_vs_text_specs(
                gpt_source="caption_gpt",
                text_source="text_caption",
                metadata_label="caption",
            )
        )
        for target_policy in policies:
            transferred_policy = _other_policy(target_policy)
            specs.append(
                ContrastSpec(
                    comparison_id=(
                        "caption_gpt_vs_text_transfer_gap_difference_target_"
                        f"{target_policy}"
                    ),
                    analysis_family="caption_gpt_vs_text_transfer_gap",
                    target_policy=target_policy,
                    estimand=(
                        "(caption GPT matched - caption GPT transferred) - "
                        "(caption text-affinity matched - caption "
                        "text-affinity transferred)"
                    ),
                    positive_meaning=(
                        "caption GPT has a larger transfer penalty; negative "
                        "means caption GPT is more robust than caption "
                        "text-affinity"
                    ),
                    terms=(
                        ContrastTerm(
                            "caption_gpt",
                            _observed(target_policy, target_policy),
                            1.0,
                            "caption GPT matched",
                        ),
                        ContrastTerm(
                            "caption_gpt",
                            _observed(transferred_policy, target_policy),
                            -1.0,
                            "caption GPT transferred",
                        ),
                        ContrastTerm(
                            "text_caption",
                            _observed(target_policy, target_policy),
                            -1.0,
                            "caption text-affinity matched",
                        ),
                        ContrastTerm(
                            "text_caption",
                            _observed(transferred_policy, target_policy),
                            1.0,
                            "caption text-affinity transferred",
                        ),
                    ),
                )
            )
    if include_text_categories:
        specs.extend(
            _gpt_vs_text_specs(
                gpt_source="categories_gpt",
                text_source="text_categories",
                metadata_label="categories",
            )
        )

    # Existing within-GPT history, no-history, and grounding contrasts.
    for target_policy in policies:
        transferred_policy = _other_policy(target_policy)
        within_specs = (
            (
                "matched_minus_transferred",
                _observed(target_policy, target_policy),
                _observed(transferred_policy, target_policy),
                "matched history",
                "transferred history",
            ),
            (
                "matched_minus_no_history",
                _observed(target_policy, target_policy),
                _no_history(target_policy),
                "matched history",
                "no history",
            ),
            (
                "transferred_minus_no_history",
                _observed(transferred_policy, target_policy),
                _no_history(target_policy),
                "transferred history",
                "no history",
            ),
            (
                "matched_original_minus_permuted",
                _observed(target_policy, target_policy),
                _permuted(target_policy, target_policy),
                "matched observed history",
                "matched permuted-association history",
            ),
            (
                "transferred_original_minus_permuted",
                _observed(transferred_policy, target_policy),
                _permuted(transferred_policy, target_policy),
                "transferred observed history",
                "transferred permuted-association history",
            ),
        )
        for suffix, left_selector, right_selector, left_label, right_label in (
            within_specs
        ):
            specs.append(
                _simple_spec(
                    comparison_id=(
                        f"caption_gpt_{suffix}_target_{target_policy}"
                    ),
                    family="within_caption_gpt",
                    target_policy=target_policy,
                    estimand=f"{left_label} minus {right_label}",
                    positive_meaning=f"{left_label} is better",
                    left_source="caption_gpt",
                    right_source="caption_gpt",
                    left_selector=left_selector,
                    right_selector=right_selector,
                    left_label=left_label,
                    right_label=right_label,
                )
            )
    return specs


def run_suite_analysis(
    *,
    gpt_run_dir: Path,
    classical_run_dir: Path,
    output_dir: Path,
    categories_run_dir: Path | None = None,
    text_run_dir: Path | None = None,
    metric: str = DEFAULT_METRIC,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    sign_flip_samples: int = DEFAULT_SIGN_FLIP_SAMPLES,
    seed: int = DEFAULT_SEED,
    skip_caption_ablation: bool = False,
) -> dict[str, Any]:
    """Load completed runs, validate parity, and write suite-level artifacts."""

    gpt_run_dir = gpt_run_dir.expanduser().resolve()
    classical_run_dir = classical_run_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    artifacts: dict[str, MethodArtifact] = {
        "caption_gpt": _load_method_artifact(
            gpt_run_dir / CAPTION_METHOD, expected_method=CAPTION_METHOD
        ),
        "item_ctr": _load_method_artifact(
            classical_run_dir / ITEM_CTR_METHOD,
            expected_method=ITEM_CTR_METHOD,
        ),
    }
    exposure_logistic_root = (
        classical_run_dir / EXPOSURE_LOGISTIC_METHOD
    )
    if exposure_logistic_root.exists():
        artifacts["exposure_logistic"] = _load_method_artifact(
            exposure_logistic_root,
            expected_method=EXPOSURE_LOGISTIC_METHOD,
        )
    resolved_categories_run_dir = (
        categories_run_dir.expanduser().resolve()
        if categories_run_dir is not None
        else gpt_run_dir
    )
    categories_root = resolved_categories_run_dir / CATEGORIES_METHOD
    if not skip_caption_ablation and categories_root.exists():
        artifacts["categories_gpt"] = _load_method_artifact(
            categories_root,
            expected_method=CATEGORIES_METHOD,
        )
    include_caption_ablation = "categories_gpt" in artifacts
    resolved_text_run_dir = (
        text_run_dir.expanduser().resolve()
        if text_run_dir is not None
        else None
    )
    if resolved_text_run_dir is not None:
        artifacts["text_caption"] = _load_method_artifact(
            resolved_text_run_dir / TEXT_CAPTION_METHOD,
            expected_method=TEXT_CAPTION_METHOD,
        )
        if include_caption_ablation:
            artifacts["text_categories"] = _load_method_artifact(
                resolved_text_run_dir / TEXT_CATEGORIES_METHOD,
                expected_method=TEXT_CATEGORIES_METHOD,
            )
    primary_ids = primary_comparison_ids(
        include_caption_ablation=include_caption_ablation
    )
    parity = _validate_input_parity(artifacts)
    natural_protocol = str(parity["run_protocol"]).startswith(
        "test_only_direct_natural_"
    )
    records = analyze_suite(
        caption=artifacts["caption_gpt"].per_user,
        categories=(
            artifacts["categories_gpt"].per_user
            if include_caption_ablation
            else None
        ),
        item_ctr=artifacts["item_ctr"].per_user,
        exposure_logistic=(
            artifacts["exposure_logistic"].per_user
            if "exposure_logistic" in artifacts
            else None
        ),
        text_caption=(
            artifacts["text_caption"].per_user
            if "text_caption" in artifacts
            else None
        ),
        text_categories=(
            artifacts["text_categories"].per_user
            if "text_categories" in artifacts
            else None
        ),
        metric=metric,
        bootstrap_samples=bootstrap_samples,
        sign_flip_samples=sign_flip_samples,
        seed=seed,
    )

    result = {
        "schema_version": 1,
        "analysis": "kuairand_policy_shift_suite_comparisons",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "metric": metric,
        "metric_higher_is_better": _metric_higher_is_better(metric),
        "statistical_protocol": {
            "pairing_keys": list(PAIR_KEYS),
            "seed_handling": (
                "mean contrast over candidate-resample task seeds within user"
            ),
            "bootstrap_unit": "user_after_mean_over_task_seeds",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_ci": "percentile_95",
            "p_value_method": (
                "two-sided paired Monte Carlo sign-flip test on user-level "
                "seed-averaged contrasts"
            ),
            "sign_flip_samples": sign_flip_samples,
            "random_seed": seed,
            "multiplicity": (
                (
                    "Holm family-wise correction over the code-declared "
                    "primary transfer comparisons available in this suite; "
                    "for the natural-exposure task this family was fixed "
                    "before model scoring but was not externally preregistered"
                )
                if natural_protocol
                else (
                    "Holm family-wise correction over the code-declared "
                    "primary transfer comparisons available in this suite; "
                    "for the forced-c3 pilot this family was specified after "
                    "experiment outputs existed and is not preregistered"
                )
            ),
            "analysis_timing": (
                "fixed_before_natural_task_model_scoring"
                if natural_protocol
                else "post_run_forced_c3_or_legacy"
            ),
            "primary_comparison_ids": sorted(primary_ids),
            "primary_family_size": len(primary_ids),
        },
        "caption_ablation": {
            "included": include_caption_ablation,
            "skip_requested": skip_caption_ablation,
            "categories_artifact_found": categories_root.exists(),
            "omission_reason": (
                None
                if include_caption_ablation
                else (
                    "explicitly_skipped"
                    if skip_caption_ablation
                    else "categories_artifact_missing"
                )
            ),
        },
        "exposure_logistic_diagnostics": {
            "included": "exposure_logistic" in artifacts,
            "artifact_found": exposure_logistic_root.exists(),
            "status": "additional_strong_probability_baseline",
            "multiplicity": (
                "raw paired CI and p-values only; excluded from the existing "
                "primary Holm family"
            ),
        },
        "semantic_baseline_diagnostics": {
            "included": resolved_text_run_dir is not None,
            "text_run_dir": (
                str(resolved_text_run_dir)
                if resolved_text_run_dir is not None
                else None
            ),
            "caption_comparisons_included": "text_caption" in artifacts,
            "categories_comparisons_included": (
                "text_categories" in artifacts
            ),
            "status": "post_hoc_diagnostic",
            "multiplicity": (
                "raw paired CI and p-values only; excluded from the existing "
                "primary Holm family"
            ),
        },
        "input_validation": parity,
        "inputs": {
            name: _input_provenance(artifact)
            for name, artifact in artifacts.items()
        },
        "comparisons": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "suite_analysis.json", result)
    _write_csv(output_dir / "suite_comparisons.csv", records)
    (output_dir / "suite_report.md").write_text(
        render_markdown_report(result),
        encoding="utf-8",
    )
    return result


def render_markdown_report(result: Mapping[str, Any]) -> str:
    records = list(result["comparisons"])
    protocol = result["statistical_protocol"]
    lines = [
        "# KuaiRand policy-shift suite analysis",
        "",
        f"Primary metric: `{result['metric']}`.",
        "",
        (
            "All comparisons are paired on `task_seed × user × "
            "candidate_group`. For conditional ranking, each user is averaged "
            "over that user's available mixed-label candidate-resample seeds "
            "before users are bootstrapped; the report exposes the number of "
            "eligible seeds per user."
        ),
        "",
        (
            f"Intervals use {protocol['bootstrap_samples']:,} user-level "
            f"bootstrap samples. Raw p-values use "
            f"{protocol['sign_flip_samples']:,} paired sign flips. Holm "
            "correction is applied only to the "
            f"{protocol['primary_family_size']} code-declared primary transfer "
            "comparisons available in this suite."
        ),
        "",
        (
            "The primary family was fixed before natural-task model scoring "
            "but was not externally preregistered."
            if protocol["analysis_timing"]
            == "fixed_before_natural_task_model_scoring"
            else (
                "The primary family was specified after the forced-c3 "
                "experiment outputs existed; it is a reproducible "
                "multiplicity correction, not a preregistered confirmatory "
                "analysis."
            )
        ),
        "",
        (
            "Caption ablation included: "
            f"`{result['caption_ablation']['included']}`"
            + (
                "."
                if result["caption_ablation"]["included"]
                else (
                    "; omission reason: "
                    f"`{result['caption_ablation']['omission_reason']}`."
                )
            )
        ),
        "",
        (
            "Post-hoc matched-metadata text-affinity diagnostics included: "
            f"`{result['semantic_baseline_diagnostics']['included']}`. These "
            "comparisons report raw paired intervals/p-values and do not "
            "enter the primary Holm family."
        ),
        "",
        (
            "Additional exposure-aware logistic diagnostics included: "
            f"`{result['exposure_logistic_diagnostics']['included']}`. These "
            "comparisons report raw paired intervals/p-values and do not "
            "enter the primary Holm family."
        ),
        "",
        "## Primary transfer family",
        "",
        _markdown_table([row for row in records if row["primary"]], holm=True),
        "",
    ]
    family_titles = (
        ("metadata_ablation", "Caption ablation: all observed/no-history cells"),
        ("gpt_vs_item_ctr", "Caption GPT versus ItemCTR: full 2×2"),
        (
            "difference_in_transfer_gaps",
            "Difference in transfer penalties",
        ),
        (
            "gpt_vs_exposure_logistic",
            "Caption GPT versus exposure-aware logistic: full 2×2",
        ),
        (
            "difference_in_transfer_gaps_exposure_logistic",
            "GPT versus exposure-aware logistic transfer-gap difference",
        ),
        ("within_caption_gpt", "Within-GPT history and grounding controls"),
        (
            "gpt_vs_text_affinity_caption",
            "Caption GPT versus caption text-affinity: full 2×2",
        ),
        (
            "gpt_vs_text_affinity_categories",
            "Categories GPT versus categories text-affinity: full 2×2",
        ),
        (
            "caption_gpt_vs_text_transfer_gap",
            "Caption GPT versus text-affinity transfer-gap difference",
        ),
    )
    for family, title in family_titles:
        family_rows = [
            row for row in records if row["analysis_family"] == family
        ]
        if not family_rows:
            continue
        lines.extend(
            [
                f"## {title}",
                "",
                _markdown_table(
                    family_rows,
                    holm=False,
                ),
                "",
            ]
        )

    significant = [
        row
        for row in records
        if row["primary"] and row["holm_reject_0_05"]
    ]
    lines.extend(
        [
            "## Mechanical significance summary",
            "",
        ]
    )
    if significant:
        for row in significant:
            lines.append(
                f"- `{row['comparison_id']}`: Δ={row['mean_delta']:.4f}, "
                f"Holm p={row['holm_adjusted_p']:.4g}."
            )
    else:
        lines.append(
            "- No primary contrast rejects the null after Holm correction."
        )
    lines.extend(
        [
            "",
            "## Sign conventions",
            "",
            (
                "- Simple contrasts are quality-normalized left minus right; "
                "positive values mean the first named condition/model is "
                "better. For log-loss and Brier, this is raw right minus left "
                "because lower is better."
            ),
            (
                "- For difference-in-transfer-gap contrasts, negative values "
                "mean GPT has the smaller matched-to-transferred penalty and is "
                "therefore more robust than ItemCTR."
            ),
            (
                "- Statistical significance alone does not establish that "
                "history is item-specifically grounded; use the permutation "
                "controls for that diagnostic."
            ),
            "",
            "## Input parity",
            "",
            (
                f"- Task seeds: `{result['input_validation']['task_seeds']}`."
            ),
            (
                f"- Cohort users: "
                f"`{result['input_validation']['n_cohort_users']}`."
            ),
            (
                "- Exact target rows across all included method artifacts: "
                f"`{result['input_validation']['exact_target_rows_match']}`."
            ),
            (
                "- Canonical target SHA-256: "
                f"`{result['input_validation']['canonical_target_sha256']}`."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in original order."""

    if any(not 0.0 <= value <= 1.0 for value in p_values):
        raise ValueError("p-values must lie in [0, 1]")
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [math.nan] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _estimate_contrast(
    frames: Mapping[str, pd.DataFrame],
    spec: ContrastSpec,
    *,
    metric: str,
    higher_is_better: bool,
    bootstrap_samples: int,
    sign_flip_samples: int,
    seed: int,
) -> dict[str, Any]:
    selected_terms: list[pd.DataFrame] = []
    value_columns: list[str] = []
    term_metadata: list[dict[str, Any]] = []
    for index, term in enumerate(spec.terms):
        selected = _select_cell(
            frames[term.source],
            selector=term.selector,
            metric=metric,
            source=term.source,
        )
        value_column = f"value_{index}"
        value_columns.append(value_column)
        selected_terms.append(selected.rename(columns={metric: value_column}))
        term_metadata.append(
            {
                "source": term.source,
                "selector": dict(term.selector),
                "coefficient": term.coefficient,
                "label": term.label,
            }
        )

    # Paid permutation controls may deliberately use a strict subset of the
    # main cohort. In that one case, compare the observed condition only on
    # the exact seed-user-group keys for which a permuted prompt was run.
    # Every other contrast remains strict and rejects missing method/task rows.
    has_permuted_term = any(
        term.selector.get("history_control") == "permuted"
        for term in spec.terms
    )
    if has_permuted_term:
        permuted_key_sets = [
            frame.loc[:, list(PAIR_KEYS)].drop_duplicates()
            for term, frame in zip(spec.terms, selected_terms, strict=True)
            if term.selector.get("history_control") == "permuted"
        ]
        control_keys = permuted_key_sets[0]
        for keys in permuted_key_sets[1:]:
            before = len(control_keys)
            control_keys = control_keys.merge(
                keys,
                on=list(PAIR_KEYS),
                how="inner",
                validate="one_to_one",
            )
            if len(control_keys) != before or len(control_keys) != len(keys):
                raise ValueError(
                    f"Unpaired permutation rows in contrast "
                    f"{spec.comparison_id!r}"
                )
        selected_terms = [
            frame.merge(
                control_keys,
                on=list(PAIR_KEYS),
                how="inner",
                validate="one_to_one",
            )
            for frame in selected_terms
        ]

    merged: pd.DataFrame | None = None
    for selected in selected_terms:
        if merged is None:
            merged = selected
            continue
        before = len(merged)
        merged = merged.merge(
            selected,
            on=list(PAIR_KEYS),
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != before or len(merged) != len(selected):
            raise ValueError(
                f"Unpaired rows in contrast {spec.comparison_id!r}: "
                f"left={before}, right={len(selected)}, paired={len(merged)}"
            )
    if merged is None or merged.empty:
        raise ValueError(f"Contrast {spec.comparison_id!r} has no paired rows")

    merged["raw_contrast"] = sum(
        term.coefficient * merged[value_column]
        for term, value_column in zip(spec.terms, value_columns, strict=True)
    )
    quality_direction_multiplier = 1.0 if higher_is_better else -1.0
    merged["contrast"] = (
        quality_direction_multiplier * merged["raw_contrast"]
    )
    term_mean_scores = [
        {
            **metadata,
            "mean_score_on_paired_rows": float(merged[value_column].mean()),
        }
        for metadata, value_column in zip(
            term_metadata,
            value_columns,
            strict=True,
        )
    ]
    per_user = merged.groupby("user_id", sort=False)["contrast"].mean()
    seeds_per_user = merged.groupby("user_id", sort=False)["task_seed"].nunique()
    available_seed_count = int(merged["task_seed"].nunique())
    values = per_user.to_numpy(dtype=float)
    ci_low, ci_high = _bootstrap_ci(
        values,
        samples=bootstrap_samples,
        seed=seed,
    )
    p_value = _sign_flip_p_value(
        values,
        samples=sign_flip_samples,
        seed=seed + 1,
    )
    return {
        "comparison_id": spec.comparison_id,
        "analysis_family": spec.analysis_family,
        "target_policy": spec.target_policy,
        "metric": metric,
        "metric_higher_is_better": higher_is_better,
        "quality_direction_multiplier": quality_direction_multiplier,
        "estimand": spec.estimand,
        "positive_meaning": spec.positive_meaning,
        "terms": term_metadata,
        "term_mean_scores": term_mean_scores,
        "n_users": int(len(per_user)),
        "n_seed_user_pairs": int(len(merged)),
        "task_seeds": sorted(
            int(value) for value in merged["task_seed"].unique()
        ),
        "metric_eligible_seeds_per_user_min": int(seeds_per_user.min()),
        "metric_eligible_seeds_per_user_mean": float(
            seeds_per_user.mean()
        ),
        "metric_eligible_seeds_per_user_max": int(seeds_per_user.max()),
        "users_with_all_metric_eligible_seeds": int(
            seeds_per_user.eq(available_seed_count).sum()
        ),
        "available_task_seed_count": available_seed_count,
        "mean_delta": float(values.mean()),
        "median_user_delta": float(np.median(values)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "raw_p_value": p_value,
        "p_value_method": "paired_user_sign_flip_after_seed_mean",
        "bootstrap_unit": "user_after_mean_over_task_seeds",
        "bootstrap_samples": bootstrap_samples,
        "holm_family": None,
        "holm_adjusted_p": None,
        "holm_reject_0_05": None,
    }


def _metric_higher_is_better(metric: str) -> bool:
    return metric not in {"log_loss", "brier"}


def _bootstrap_ci(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    batch_size = min(2_048, samples)
    offset = 0
    while offset < samples:
        size = min(batch_size, samples - offset)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means[offset : offset + size] = values[indices].mean(axis=1)
        offset += size
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _sign_flip_p_value(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> float:
    observed = abs(float(values.mean()))
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    batch_size = min(2_048, samples)
    while completed < samples:
        size = min(batch_size, samples - completed)
        signs = rng.integers(0, 2, size=(size, len(values)), dtype=np.int8)
        signs = signs * 2 - 1
        null_means = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(null_means >= observed))
        completed += size
    return float((extreme + 1) / (samples + 1))


def _apply_primary_holm(
    records: list[dict[str, Any]],
    *,
    primary_ids: frozenset[str],
) -> None:
    primary_indices = [
        index for index, record in enumerate(records) if record["primary"]
    ]
    if {records[index]["comparison_id"] for index in primary_indices} != set(
        primary_ids
    ):
        raise AssertionError("Primary comparison family is incomplete")
    adjusted = holm_adjust(
        [float(records[index]["raw_p_value"]) for index in primary_indices]
    )
    for index, adjusted_p in zip(primary_indices, adjusted, strict=True):
        records[index]["holm_family"] = "primary_transfer"
        records[index]["holm_adjusted_p"] = adjusted_p
        records[index]["holm_reject_0_05"] = bool(adjusted_p < 0.05)


def _validate_per_user_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    metric: str,
) -> None:
    required = {*PAIR_KEYS, *CELL_COLUMNS, metric}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    missing_metric = frame[metric].isna()
    if missing_metric.any():
        if "metric_eligible" not in frame.columns:
            raise ValueError(
                f"{name} has missing {metric}; rerun failed groups or "
                "explicitly analyze a failure-as-zero metric"
            )
        eligible = frame["metric_eligible"].astype(bool)
        if (missing_metric & eligible).any() or (
            ~missing_metric & ~eligible
        ).any():
            raise ValueError(
                f"{name} missing {metric} values do not exactly match "
                "metric_eligible=False rows"
            )
    duplicate_columns = [*PAIR_KEYS, *CELL_COLUMNS]
    if frame.duplicated(duplicate_columns).any():
        raise ValueError(f"{name} has duplicate per-user cell rows")


def _select_cell(
    frame: pd.DataFrame,
    *,
    selector: Mapping[str, str],
    metric: str,
    source: str,
) -> pd.DataFrame:
    selected = frame
    for column, value in selector.items():
        selected = selected[selected[column].eq(value)]
    selected = selected.loc[:, [*PAIR_KEYS, metric]].copy()
    selected = selected.dropna(subset=[metric])
    if selected.empty:
        raise ValueError(f"No rows for {source} selector={dict(selector)}")
    if selected.duplicated(PAIR_KEYS).any():
        raise ValueError(
            f"Multiple rows for {source} selector={dict(selector)}"
        )
    return selected


def _simple_spec(
    *,
    comparison_id: str,
    family: str,
    target_policy: str,
    estimand: str,
    positive_meaning: str,
    left_source: str,
    right_source: str,
    left_selector: Mapping[str, str],
    right_selector: Mapping[str, str],
    left_label: str,
    right_label: str,
) -> ContrastSpec:
    return ContrastSpec(
        comparison_id=comparison_id,
        analysis_family=family,
        target_policy=target_policy,
        estimand=estimand,
        positive_meaning=positive_meaning,
        terms=(
            ContrastTerm(left_source, left_selector, 1.0, left_label),
            ContrastTerm(right_source, right_selector, -1.0, right_label),
        ),
    )


def _gpt_vs_text_specs(
    *,
    gpt_source: str,
    text_source: str,
    metadata_label: str,
) -> list[ContrastSpec]:
    specs = []
    family = f"gpt_vs_text_affinity_{metadata_label}"
    for target_policy in ("standard", "random"):
        for history_policy in ("standard", "random"):
            specs.append(
                _simple_spec(
                    comparison_id=(
                        f"{metadata_label}_gpt_minus_text_affinity_"
                        f"{history_policy}_to_{target_policy}"
                    ),
                    family=family,
                    target_policy=target_policy,
                    estimand=(
                        f"{metadata_label} GPT minus matched-metadata "
                        f"text-affinity for {history_policy} history"
                    ),
                    positive_meaning=f"{metadata_label} GPT is better",
                    left_source=gpt_source,
                    right_source=text_source,
                    left_selector=_observed(history_policy, target_policy),
                    right_selector=_observed(history_policy, target_policy),
                    left_label=f"{metadata_label} GPT",
                    right_label=f"{metadata_label} text-affinity",
                )
            )
    return specs


def _observed(history_policy: str, target_policy: str) -> dict[str, str]:
    return {
        "history_policy": history_policy,
        "target_policy": target_policy,
        "history_control": "observed",
    }


def _permuted(history_policy: str, target_policy: str) -> dict[str, str]:
    return {
        "history_policy": history_policy,
        "target_policy": target_policy,
        "history_control": "permuted",
    }


def _no_history(target_policy: str) -> dict[str, str]:
    return {
        "history_policy": "none",
        "target_policy": target_policy,
        "history_control": "none",
    }


def _other_policy(policy: str) -> str:
    if policy == "standard":
        return "random"
    if policy == "random":
        return "standard"
    raise ValueError(f"Unknown policy: {policy!r}")


def _derived_seed(seed: int, comparison_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{comparison_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**32)


def _load_method_artifact(
    root: Path,
    *,
    expected_method: str,
) -> MethodArtifact:
    required = (
        root / "manifest.json",
        root / "per_user_metrics.parquet",
        root / "predictions.parquet",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing method artifacts: {missing}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"{root} is not a complete run")
    manifest_method = manifest.get("method")
    if isinstance(manifest_method, Mapping):
        manifest_name = manifest_method.get("name")
    else:
        manifest_name = manifest_method
    if manifest_name != expected_method:
        raise ValueError(
            f"Expected method {expected_method!r} at {root}, "
            f"found {manifest_name!r}"
        )
    return MethodArtifact(
        name=expected_method,
        root=root,
        manifest=manifest,
        per_user=pd.read_parquet(root / "per_user_metrics.parquet"),
        predictions=pd.read_parquet(root / "predictions.parquet"),
    )


def _validate_input_parity(
    artifacts: Mapping[str, MethodArtifact],
) -> dict[str, Any]:
    task_seed_sets = {
        name: tuple(int(seed) for seed in artifact.manifest["task_seeds"])
        for name, artifact in artifacts.items()
    }
    if len(set(task_seed_sets.values())) != 1:
        raise ValueError(f"Task-seed mismatch: {task_seed_sets}")
    cohort_hashes = {
        name: artifact.manifest["cohort"]["user_ids_sha256"]
        for name, artifact in artifacts.items()
    }
    if len(set(cohort_hashes.values())) != 1:
        raise ValueError(f"Cohort mismatch: {cohort_hashes}")
    cohort_sizes = {
        name: int(artifact.manifest["cohort"]["n_users"])
        for name, artifact in artifacts.items()
    }
    if len(set(cohort_sizes.values())) != 1:
        raise ValueError(f"Cohort-size mismatch: {cohort_sizes}")
    run_protocols = {
        name: artifact.manifest.get("protocol")
        for name, artifact in artifacts.items()
    }
    if len(set(run_protocols.values())) != 1:
        raise ValueError(f"Run-protocol mismatch: {run_protocols}")

    canonical_targets = {
        name: _canonical_targets(artifact.predictions)
        for name, artifact in artifacts.items()
    }
    first_name = next(iter(canonical_targets))
    first = canonical_targets[first_name]
    for name, targets in canonical_targets.items():
        if not first.equals(targets):
            raise ValueError(
                f"Exact target-row mismatch between {first_name} and {name}"
            )
    target_sha = _frame_sha256(first)
    return {
        "task_seeds": list(next(iter(task_seed_sets.values()))),
        "n_cohort_users": next(iter(cohort_sizes.values())),
        "cohort_user_ids_sha256": next(iter(cohort_hashes.values())),
        "run_protocol": next(iter(run_protocols.values())),
        "exact_target_rows_match": True,
        "canonical_target_rows": int(len(first)),
        "canonical_target_sha256": target_sha,
    }


def _canonical_targets(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "task_seed",
        "target_policy",
        "user_id",
        "candidate_group",
        "item_id",
        "target",
    ]
    missing = sorted(set(columns) - set(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing target columns: {missing}")
    targets = predictions.loc[:, columns].drop_duplicates()
    conflict_keys = columns[:-1]
    if targets.duplicated(conflict_keys).any():
        raise ValueError("Predictions contain conflicting target labels")
    return targets.sort_values(columns, kind="stable").reset_index(drop=True)


def _frame_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_provenance(artifact: MethodArtifact) -> dict[str, Any]:
    return {
        "method": artifact.name,
        "root": str(artifact.root),
        "manifest_sha256": _file_sha256(artifact.root / "manifest.json"),
        "per_user_metrics_sha256": _file_sha256(
            artifact.root / "per_user_metrics.parquet"
        ),
        "predictions_sha256": _file_sha256(
            artifact.root / "predictions.parquet"
        ),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for record in records:
        row = dict(record)
        row["terms"] = json.dumps(
            row["terms"], sort_keys=True, ensure_ascii=False
        )
        row["term_mean_scores"] = json.dumps(
            row["term_mean_scores"], sort_keys=True, ensure_ascii=False
        )
        row["task_seeds"] = json.dumps(row["task_seeds"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _markdown_table(
    records: Iterable[Mapping[str, Any]],
    *,
    holm: bool,
) -> str:
    rows = list(records)
    if not rows:
        return "_No comparisons._"
    headers = [
        "Comparison",
        "Target",
        "Δ",
        "95% CI",
        "Raw p",
    ]
    if holm:
        headers.extend(["Holm p", "Reject"])
    output = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        values = [
            f"`{row['comparison_id']}`",
            str(row["target_policy"]),
            f"{row['mean_delta']:.4f}",
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]",
            f"{row['raw_p_value']:.4g}",
        ]
        if holm:
            values.extend(
                [
                    f"{row['holm_adjusted_p']:.4g}",
                    "yes" if row["holm_reject_0_05"] else "no",
                ]
            )
        output.append("| " + " | ".join(values) + " |")
    return "\n".join(output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze paired KuaiRand policy-shift GPT and classical artifacts"
        )
    )
    parser.add_argument(
        "--gpt-run-dir",
        type=Path,
        required=True,
        help=(
            "Suite root containing caption GPT and, when available, "
            "categories-only GPT method directories."
        ),
    )
    parser.add_argument(
        "--classical-run-dir",
        type=Path,
        required=True,
        help="Suite root containing the item_ctr method directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--categories-run-dir",
        type=Path,
        help=(
            "Optional separate suite root containing the categories-only GPT "
            "method directory. Defaults to --gpt-run-dir."
        ),
    )
    parser.add_argument(
        "--text-run-dir",
        type=Path,
        help=(
            "Optional suite root containing matched-metadata caption and "
            "categories text-affinity methods. Comparisons are post-hoc "
            "diagnostics outside the primary Holm family."
        ),
    )
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument(
        "--skip-caption-ablation",
        action="store_true",
        help=(
            "Do not require or analyze the categories-only GPT artifact. "
            "Caption ablation is also skipped automatically when that method "
            "directory is absent."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument(
        "--sign-flip-samples",
        type=int,
        default=DEFAULT_SIGN_FLIP_SAMPLES,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_suite_analysis(
        gpt_run_dir=args.gpt_run_dir,
        classical_run_dir=args.classical_run_dir,
        output_dir=args.output_dir,
        categories_run_dir=args.categories_run_dir,
        text_run_dir=args.text_run_dir,
        metric=args.metric,
        bootstrap_samples=args.bootstrap_samples,
        sign_flip_samples=args.sign_flip_samples,
        seed=args.seed,
        skip_caption_ablation=args.skip_caption_ablation,
    )
    summary = {
        "status": "complete",
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "comparisons": len(result["comparisons"]),
        "primary_comparisons": sum(
            bool(record["primary"]) for record in result["comparisons"]
        ),
        "holm_rejections": sum(
            bool(record["holm_reject_0_05"])
            for record in result["comparisons"]
            if record["primary"]
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
