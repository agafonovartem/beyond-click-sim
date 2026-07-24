from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Literal

import pandas as pd

from beyond_click_sim.llm_clients import make_llm_client
from beyond_click_sim.scorers import Scorer
from beyond_click_sim.tasks import KuaiRandPolicyShiftTask
from runners.ood.kuairand_policy_shift.checkpoint import (
    GroupCheckpointStore,
    build_group_checkpoint_key,
)
from runners.ood.kuairand_policy_shift.cost import (
    CostEstimate,
    TokenCounter,
    enforce_hard_budget,
    estimate_prompt_plan_cost,
    token_cost,
)
from runners.ood.kuairand_policy_shift.methods import (
    METHOD_REGISTRY,
    MethodSpec,
    make_scorer,
)
from runners.ood.kuairand_policy_shift.metrics import (
    build_paired_comparisons,
    evaluate_prediction_cells,
)
from runners.ood.kuairand_policy_shift.task_builders import (
    DEFAULT_CANDIDATE_SEEDS,
    DEFAULT_COHORT_SEED,
    DEFAULT_NATURAL_CANDIDATE_SEEDS,
    DEFAULT_SOURCE_SEED,
    build_natural_policy_shift_task,
    build_policy_shift_task,
    repo_root,
    select_user_cohort,
    shared_test_users,
)


Policy = Literal["standard", "random"]
Control = Literal["observed", "permuted", "none"]
POLICIES: tuple[Policy, ...] = ("standard", "random")
CONTROLS: tuple[Control, ...] = ("observed", "permuted", "none")
DEFAULT_METHODS = (
    "global_mean",
    "item_ctr",
    "user_ctr",
    "exposure_logistic_bias",
    "item_knn",
    "bpr",
    "text_affinity_caption_categories",
    "llm_openai_vk_gpt54_mini_probability_caption_categories",
)
DEFAULT_OUTPUT_DIR = (
    repo_root() / "outputs" / "ood" / "kuairand_policy_shift"
)


@dataclass(frozen=True)
class CellSpec:
    history_policy: Policy | None
    target_policy: Policy
    history_control: Control

    @property
    def display_history_policy(self) -> str:
        return "none" if self.history_control == "none" else str(self.history_policy)

    @property
    def task_history_policy(self) -> Policy:
        # The specialized Task API needs a policy even for an empty history.
        # Either policy produces the same schema-preserving empty frame.
        return "standard" if self.history_policy is None else self.history_policy

    @property
    def matched(self) -> bool:
        return (
            self.history_control == "observed"
            and self.history_policy == self.target_policy
        )


def planned_cells(controls: Iterable[str]) -> list[CellSpec]:
    normalized = tuple(dict.fromkeys(str(control) for control in controls))
    unknown = sorted(set(normalized) - set(CONTROLS))
    if unknown:
        raise ValueError(f"Unknown history controls: {unknown}")
    cells: list[CellSpec] = []
    for control in normalized:
        if control == "none":
            cells.extend(
                CellSpec(None, target_policy, "none")
                for target_policy in POLICIES
            )
            continue
        cells.extend(
            CellSpec(history_policy, target_policy, control)  # type: ignore[arg-type]
            for history_policy in POLICIES
            for target_policy in POLICIES
        )
    return cells


def prepare_llm_preflight(
    tasks_by_seed: dict[int, KuaiRandPolicyShiftTask],
    *,
    spec: MethodSpec,
    user_ids: Sequence[object],
    control_user_ids: Sequence[object] | None = None,
    controls: Iterable[str],
    safety_factor: float = 1.10,
    token_counter: TokenCounter | None = None,
) -> tuple[CostEstimate, list[list[dict[str, str]]]]:
    """Materialize and price every planned test prompt without API access."""

    if not spec.is_llm:
        raise ValueError("Only LLM methods have prompt preflights")
    main_users, control_users = _validated_user_cohorts(
        user_ids,
        control_user_ids,
    )
    prompts: list[list[dict[str, str]]] = []
    no_call_client = object()
    for task_seed, task in sorted(tasks_by_seed.items()):
        for cell_spec in planned_cells(controls):
            if not spec.supports_control(cell_spec.history_control):
                continue
            cell = task.cell(
                history_policy=cell_spec.task_history_policy,
                target_policy=cell_spec.target_policy,
                split="test",
                history_control=cell_spec.history_control,
            )
            cell_users = _users_for_cell(
                cell_spec,
                main_users=main_users,
                control_users=control_users,
            )
            history = _history_for_method(
                cell.history,
                spec,
                cell_users,
                training_history=cell.training_history,
            )
            X_history, y_history = _split_model_xy(
                history, target_column=cell.schema.target_column
            )
            targets = _select_target_users(cell.targets, cell_users)
            X_target, _ = _split_model_xy(
                targets, target_column=cell.schema.target_column
            )
            scorer = make_scorer(
                spec,
                seed=task_seed,
                client=no_call_client,
            ).fit(X_history, y_history)
            build_messages = getattr(scorer, "_build_messages")
            for _, group in X_target.groupby("candidate_group", sort=False):
                labels = [f"C{position}" for position in range(1, len(group) + 1)]
                user_ids_in_group = group["user_id"].drop_duplicates()
                if len(user_ids_in_group) != 1:
                    raise ValueError("Candidate group contains multiple users")
                prompts.append(
                    build_messages(
                        user_id=user_ids_in_group.iloc[0],
                        candidates=group,
                        labels=labels,
                    )
                )
    estimate = estimate_prompt_plan_cost(
        prompts,
        max_completion_tokens=spec.max_completion_tokens,
        expected_output_tokens=spec.expected_output_tokens,
        input_price_per_million=spec.input_price_per_million,
        output_price_per_million=spec.output_price_per_million,
        safety_factor=safety_factor,
        token_counter=token_counter,
    )
    return estimate, prompts


def run_method_suite(
    tasks_by_seed: dict[int, KuaiRandPolicyShiftTask],
    *,
    method_name: str,
    output_dir: Path,
    user_ids: Sequence[object],
    control_user_ids: Sequence[object] | None = None,
    controls: Iterable[str] = ("observed",),
    hard_budget_usd: float | None = None,
    safety_factor: float = 1.10,
    execute_llm: bool = False,
    client: Any | None = None,
    token_counter: TokenCounter | None = None,
    preflight: CostEstimate | None = None,
    max_attempts: int = 1,
    max_workers: int = 1,
) -> dict[str, object]:
    """Run one method across task seeds and policy-shift cells.

    The LLM path is test-only direct ranking. It intentionally makes no
    validation calls and does not fit a pointwise/F1 threshold.
    """

    if method_name not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method {method_name!r}; available={sorted(METHOD_REGISTRY)}"
        )
    if not tasks_by_seed:
        raise ValueError("tasks_by_seed must not be empty")
    if not user_ids:
        raise ValueError("user_ids must not be empty")
    main_users, control_users = _validated_user_cohorts(
        user_ids,
        control_user_ids,
    )
    if max_attempts < 1 or max_workers < 1:
        raise ValueError("max_attempts and max_workers must be positive")

    spec = METHOD_REGISTRY[method_name]
    evaluation_protocol = _evaluation_protocol(tasks_by_seed)
    ranking_k = int(evaluation_protocol["ranking_k"])
    ndcg_metric = f"ndcg@{ranking_k}"
    strict_ndcg_metric = f"failure_as_zero_{ndcg_metric}"
    if spec.is_paid and max_attempts != 1:
        raise ValueError("Paid methods are locked to max_attempts=1")
    cell_specs = [
        cell
        for cell in planned_cells(controls)
        if spec.supports_control(cell.history_control)
    ]
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cost_estimate: CostEstimate | None = preflight
    if spec.is_llm:
        if cost_estimate is None:
            cost_estimate, _ = prepare_llm_preflight(
                tasks_by_seed,
                spec=spec,
                user_ids=main_users,
                control_user_ids=control_users,
                controls=controls,
                safety_factor=safety_factor,
                token_counter=token_counter,
            )
        if spec.is_paid and hard_budget_usd is not None and hard_budget_usd > 0:
            enforce_hard_budget(
                cost_estimate,
                hard_budget_usd=hard_budget_usd,
            )
        _write_json(
            output_dir / "preflight_cost.json",
            cost_estimate.to_dict(),
        )
        if not execute_llm:
            result = {
                "status": "preflight_only",
                "method": method_name,
                "test_only": True,
                "cost": cost_estimate.to_dict(),
                "planned_cells": [_cell_dict(cell) for cell in cell_specs],
                "task_seeds": sorted(tasks_by_seed),
                "n_users": len(main_users),
                "n_control_users": len(control_users),
            }
            _write_json(output_dir / "manifest.json", result)
            return result
        # Crucially, client construction occurs only after the complete prompt
        # plan is known and any explicitly enabled hard-budget guard has passed.
        active_client = client or make_llm_client(str(spec.client_name))
    else:
        active_client = None

    prediction_parts: list[pd.DataFrame] = []
    errors: list[dict[str, object]] = []
    fit_diagnostics: list[dict[str, object]] = []
    score_diagnostics: list[dict[str, object]] = []
    actual_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "requests": 0,
    }
    cache_stats = {
        "hits": 0,
        "misses": 0,
        "writes": 0,
        "failed_misses": 0,
        "hits_without_usage": 0,
        "cached_prompt_tokens": 0,
        "cached_completion_tokens": 0,
        "cached_total_tokens": 0,
        "cached_requests": 0,
    }
    checkpoint_store = (
        GroupCheckpointStore(output_dir / "group_checkpoints")
        if spec.is_llm
        else None
    )
    for task_seed, task in sorted(tasks_by_seed.items()):
        for cell_spec in cell_specs:
            cell = task.cell(
                history_policy=cell_spec.task_history_policy,
                target_policy=cell_spec.target_policy,
                split="test",
                history_control=cell_spec.history_control,
            )
            cell_users = _users_for_cell(
                cell_spec,
                main_users=main_users,
                control_users=control_users,
            )
            history = _history_for_method(
                cell.history,
                spec,
                cell_users,
                training_history=cell.training_history,
            )
            targets = _select_target_users(cell.targets, cell_users)
            X_history, y_history = _split_model_xy(
                history, target_column=cell.schema.target_column
            )
            X_target, y_target = _split_model_xy(
                targets, target_column=cell.schema.target_column
            )
            scorer = make_scorer(
                spec,
                seed=task_seed,
                client=active_client,
            ).fit(X_history, y_history)
            scorer_fit_diagnostics = getattr(
                scorer, "fit_diagnostics_", None
            )
            if scorer_fit_diagnostics is not None:
                target_users = set(X_target["user_id"])
                diagnostic_entry = {
                    "task_seed": task_seed,
                    **_cell_dict(cell_spec),
                    **dict(scorer_fit_diagnostics),
                    "n_target_users": len(target_users),
                }
                profile_by_user = getattr(
                    scorer, "profile_by_user_", None
                )
                known_users = getattr(scorer, "known_users_", None)
                if profile_by_user is not None:
                    profile_users = set(profile_by_user)
                    n_target_profile_users = len(
                        target_users & profile_users
                    )
                    diagnostic_entry.update(
                        {
                            "n_target_users_with_profile": (
                                n_target_profile_users
                            ),
                            "target_profile_user_fraction": (
                                n_target_profile_users / len(target_users)
                                if target_users
                                else 0.0
                            ),
                        }
                    )
                elif known_users is not None:
                    n_known_target_users = len(
                        target_users & set(known_users)
                    )
                    diagnostic_entry.update(
                        {
                            "n_target_users_known_to_model": (
                                n_known_target_users
                            ),
                            "target_user_known_fraction": (
                                n_known_target_users / len(target_users)
                                if target_users
                                else 0.0
                            ),
                        }
                    )
                fit_diagnostics.append(diagnostic_entry)
            if spec.is_llm:
                scores, cell_errors, cell_cache = _score_llm_groups(
                    scorer,
                    X_target,
                    max_attempts=max_attempts,
                    max_workers=max_workers,
                    checkpoint_store=checkpoint_store,
                    task_identity=_task_identity(task),
                    cell={
                        "task_seed": task_seed,
                        **_cell_dict(cell_spec),
                    },
                    model=str(spec.model),
                    request_config=_llm_request_config(scorer),
                )
                for key in cache_stats:
                    cache_stats[key] += int(cell_cache[key])
                errors.extend(
                    {
                        "task_seed": task_seed,
                        **_cell_dict(cell_spec),
                        **error,
                    }
                    for error in cell_errors
                )
                scorer_usage = getattr(scorer, "usage_", {})
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    actual_usage[key] += int(scorer_usage.get(key, 0))
                actual_usage["requests"] += int(
                    getattr(scorer, "request_count_", 0)
                )
            else:
                scores = scorer.score(X_target)
            scorer_score_diagnostics = getattr(
                scorer,
                "coverage_diagnostics_",
                None,
            )
            if scorer_score_diagnostics is not None:
                score_diagnostics.append(
                    {
                        "task_seed": task_seed,
                        **_cell_dict(cell_spec),
                        **dict(scorer_score_diagnostics),
                    }
                )

            prediction = targets.copy()
            prediction["target"] = y_target.to_numpy()
            prediction["score"] = scores.to_numpy()
            prediction.insert(0, "method", method_name)
            prediction.insert(1, "task_seed", task_seed)
            prediction.insert(2, "split", "test")
            prediction["history_policy"] = cell_spec.display_history_policy
            prediction["target_policy"] = cell_spec.target_policy
            prediction["history_control"] = cell_spec.history_control
            prediction["metadata_condition"] = spec.metadata_condition
            prediction["matched"] = cell_spec.matched
            prediction_parts.append(prediction)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    probability_scores = spec.score_semantics == "probability"
    cell_metrics, per_user = evaluate_prediction_cells(
        predictions,
        ranking_k=ranking_k,
        require_mixed_labels=bool(
            evaluation_protocol["require_mixed_labels"]
        ),
        probability_scores=probability_scores,
    )
    comparison_metrics = [
        ndcg_metric,
        "hit_rate@1",
        strict_ndcg_metric,
        "failure_as_zero_hit_rate@1",
    ]
    if probability_scores:
        comparison_metrics = ["log_loss", "brier", *comparison_metrics]
    comparisons = build_paired_comparisons(
        per_user,
        metrics=comparison_metrics,
    )
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    per_user.to_parquet(output_dir / "per_user_metrics.parquet", index=False)
    _write_jsonl(output_dir / "llm_errors.jsonl", errors)
    _write_json(output_dir / "metrics.json", {"cells": cell_metrics})
    _write_json(output_dir / "comparisons.json", {"comparisons": comparisons})

    actual_cost = (
        token_cost(
            actual_usage["prompt_tokens"],
            actual_usage["completion_tokens"],
            spec.input_price_per_million,
            spec.output_price_per_million,
        )
        if spec.is_llm
        else 0.0
    )
    cached_cost = (
        token_cost(
            cache_stats["cached_prompt_tokens"],
            cache_stats["cached_completion_tokens"],
            spec.input_price_per_million,
            spec.output_price_per_million,
        )
        if spec.is_llm
        else 0.0
    )
    manifest = {
        "status": "complete",
        "experiment": "kuairand_tab1_standard_random_policy_shift",
        "protocol": evaluation_protocol["run_protocol"],
        "method": spec.manifest_config(),
        "target": evaluation_protocol["target_description"],
        "metrics": {
            "primary": (
                ["log_loss", "brier"]
                if probability_scores
                else [ndcg_metric]
            ),
            "secondary": (
                [ndcg_metric, "hit_rate@1"]
                if probability_scores
                else ["hit_rate@1"]
            ),
            "pointwise_scope": (
                "all label-blind natural target rows"
                if probability_scores
                else None
            ),
            "ranking_eligibility": evaluation_protocol[
                "ranking_eligibility"
            ],
            "tie_policy": "average",
            "failed_llm_group_policy": "zero in failure_as_zero metrics",
        },
        "test_only": True,
        "validation_llm_calls": 0,
        "fit_diagnostics": fit_diagnostics or None,
        "score_diagnostics": score_diagnostics or None,
        "task_seeds": sorted(tasks_by_seed),
        "cohort": {
            "n_users": len(main_users),
            "user_ids_sha256": _values_sha256(main_users),
            "same_users_across_task_seeds": True,
            "permuted_control_users": len(control_users),
            "permuted_control_user_ids_sha256": _values_sha256(
                control_users
            ),
            "permuted_control_is_main_cohort_subset": True,
        },
        "cells": [_cell_dict(cell) for cell in cell_specs],
        "task_manifests": {
            str(seed): task.manifest for seed, task in sorted(tasks_by_seed.items())
        },
        "llm": {
            "preflight": (
                cost_estimate.to_dict() if cost_estimate is not None else None
            ),
            "new_api_usage": actual_usage if spec.is_llm else None,
            "new_api_cost_usd": actual_cost if spec.is_llm else None,
            # Backwards-compatible aliases. They deliberately count only calls
            # made by this invocation; cache hits incur no new API cost.
            "actual_usage": actual_usage if spec.is_llm else None,
            "actual_cost_usd": actual_cost if spec.is_llm else None,
            "group_checkpoint_cache": (
                {
                    **cache_stats,
                    "root": "group_checkpoints",
                    "content_addressed": True,
                }
                if spec.is_llm
                else None
            ),
            "cached_provider_usage": (
                {
                    "prompt_tokens": cache_stats["cached_prompt_tokens"],
                    "completion_tokens": cache_stats[
                        "cached_completion_tokens"
                    ],
                    "total_tokens": cache_stats["cached_total_tokens"],
                    "requests": cache_stats["cached_requests"],
                    "hits_without_usage": cache_stats["hits_without_usage"],
                }
                if spec.is_llm
                else None
            ),
            "provider_usage_including_cached": (
                {
                    "prompt_tokens": (
                        actual_usage["prompt_tokens"]
                        + cache_stats["cached_prompt_tokens"]
                    ),
                    "completion_tokens": (
                        actual_usage["completion_tokens"]
                        + cache_stats["cached_completion_tokens"]
                    ),
                    "total_tokens": (
                        actual_usage["total_tokens"]
                        + cache_stats["cached_total_tokens"]
                    ),
                    "requests": (
                        actual_usage["requests"]
                        + cache_stats["cached_requests"]
                    ),
                }
                if spec.is_llm
                else None
            ),
            "cached_provider_cost_usd": (
                cached_cost if spec.is_llm else None
            ),
            "provider_cost_including_cached_usd": (
                actual_cost + cached_cost if spec.is_llm else None
            ),
            "hard_budget_usd": hard_budget_usd if spec.is_paid else None,
            "max_attempts": max_attempts if spec.is_llm else None,
            "max_workers": max_workers if spec.is_llm else None,
            "errors": len(errors),
        },
        "artifacts": {
            "predictions": "predictions.parquet",
            "per_user_metrics": "per_user_metrics.parquet",
            "cell_metrics": "metrics.json",
            "paired_comparisons": "comparisons.json",
            "llm_errors": "llm_errors.jsonl",
            "group_checkpoints": (
                "group_checkpoints/" if spec.is_llm else None
            ),
        },
        "git": _git_provenance(),
        "request_config_sha256": _request_config_sha256(spec),
        "prompt_plan_sha256": (
            cost_estimate.prompt_plan_sha256
            if cost_estimate is not None
            else None
        ),
        "known_caveats": [
            *evaluation_protocol["known_caveats"],
                (
                    "Repeated candidate seeds for one user are correlated; paired "
                    "comparisons average available metric-eligible seeds within "
                    "user before bootstrapping users."
                ),
            (
                "Standard-target and random-target metric levels are not "
                "directly comparable because their observed candidate "
                "distributions differ."
            ),
            (
                "Pretrained LLMs may know released KuaiRand captions or their "
                "underlying public videos; this run does not estimate "
                "pretraining memorization."
            ),
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return {
        "status": "complete",
        "method": method_name,
        "n_predictions": int(len(predictions)),
        "n_errors": len(errors),
        "cells": cell_metrics,
        "comparisons": comparisons,
        # ``actual_cost_usd`` is retained as the cost of this invocation for
        # backward compatibility.  The cumulative field prevents a resumed
        # suite_results.json from making an already-paid run look free.
        "actual_cost_usd": actual_cost,
        "new_api_cost_usd": actual_cost,
        "provider_cost_including_cached_usd": actual_cost + cached_cost,
    }


def main() -> None:
    args = _parse_args()
    method_names = _parse_names(args.methods, choices=METHOD_REGISTRY)
    candidate_seeds = (
        _parse_ints(args.task_seeds)
        if args.task_seeds is not None
        else list(
            DEFAULT_NATURAL_CANDIDATE_SEEDS
            if args.protocol == "natural"
            else DEFAULT_CANDIDATE_SEEDS
        )
    )
    explicit_controls = (
        None
        if args.history_controls == "auto"
        else _parse_controls(args.history_controls)
    )
    controls_by_method = {
        name: (
            _default_controls(METHOD_REGISTRY[name])
            if explicit_controls is None
            else explicit_controls
        )
        for name in method_names
    }
    if args.protocol == "natural":
        tasks_by_seed = {
            seed: build_natural_policy_shift_task(candidate_seed=seed)
            for seed in candidate_seeds
        }
    else:
        tasks_by_seed = {
            seed: build_policy_shift_task(
                candidate_seed=seed,
                source_seed=args.source_seed,
                min_positive_source_users_per_policy=(
                    args.min_positive_source_users_per_policy
                ),
            )
            for seed in candidate_seeds
        }
    available_users = shared_test_users(tasks_by_seed)
    cohort = select_user_cohort(
        available_users,
        max_users=args.max_users,
        cohort_seed=args.cohort_seed,
    )
    if not cohort:
        raise RuntimeError("No users remain in the cross-seed test intersection")
    control_cohort = select_user_cohort(
        cohort,
        max_users=args.max_control_users,
        cohort_seed=args.cohort_seed,
    )

    preflights: dict[str, CostEstimate] = {}
    for method_name in method_names:
        spec = METHOD_REGISTRY[method_name]
        if not spec.is_llm:
            continue
        estimate, _ = prepare_llm_preflight(
            tasks_by_seed,
            spec=spec,
            user_ids=cohort,
            control_user_ids=control_cohort,
            controls=controls_by_method[method_name],
            safety_factor=args.budget_safety_factor,
        )
        preflights[method_name] = estimate
    total_paid_guard = sum(
        preflights[name].guarded_cost_usd
        for name in preflights
        if METHOD_REGISTRY[name].is_paid
    )
    if args.hard_budget_usd > 0 and total_paid_guard > args.hard_budget_usd:
        raise RuntimeError(
            "Combined paid-method preflight exceeds the invocation budget "
            f"before any API call: ${total_paid_guard:.4f} > "
            f"${args.hard_budget_usd:.4f}"
        )

    if args.run_dir is not None:
        run_root = Path(args.run_dir).expanduser().resolve()
        run_root.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_root = (
            Path(args.output_dir).expanduser().resolve()
            / f"{timestamp}_{args.protocol}_candidates"
            f"{'-'.join(str(seed) for seed in candidate_seeds)}"
        )
        run_root.mkdir(parents=True, exist_ok=False)
    _write_json(
        run_root / "suite_plan.json",
        {
            "protocol": args.protocol,
            "candidate_seeds": candidate_seeds,
            "source_seed": (
                args.source_seed if args.protocol == "forced-c3" else None
            ),
            "min_positive_source_users_per_policy": (
                args.min_positive_source_users_per_policy
                if args.protocol == "forced-c3"
                else None
            ),
            "controls_by_method": controls_by_method,
            "methods": method_names,
            "available_cross_seed_users": len(available_users),
            "selected_users": len(cohort),
            "selected_permuted_control_users": len(control_cohort),
            "cohort_seed": args.cohort_seed,
            "combined_paid_guarded_cost_usd": total_paid_guard,
            "hard_budget_usd": args.hard_budget_usd,
            "execute_llm": args.execute_llm,
            "preflights": {
                name: estimate.to_dict()
                for name, estimate in preflights.items()
            },
        },
    )
    results = {}
    for method_name in method_names:
        results[method_name] = run_method_suite(
            tasks_by_seed,
            method_name=method_name,
            output_dir=run_root / method_name,
            user_ids=cohort,
            control_user_ids=control_cohort,
            controls=controls_by_method[method_name],
            hard_budget_usd=(
                args.hard_budget_usd if args.hard_budget_usd > 0 else None
            ),
            safety_factor=args.budget_safety_factor,
            execute_llm=args.execute_llm,
            preflight=preflights.get(method_name),
            max_attempts=args.max_attempts,
            max_workers=args.max_workers,
        )
    _write_json(run_root / "suite_results.json", results)
    print(json.dumps(_json_safe(results), indent=2, sort_keys=True))


def _score_llm_groups(
    scorer: Scorer,
    X: pd.DataFrame,
    *,
    max_attempts: int,
    max_workers: int,
    checkpoint_store: GroupCheckpointStore | None,
    task_identity: str,
    cell: dict[str, object],
    model: str,
    request_config: dict[str, object],
) -> tuple[
    pd.Series,
    list[dict[str, object]],
    dict[str, int],
]:
    groups = list(X.groupby("candidate_group", sort=False))
    scores = pd.Series(index=X.index, dtype=float, name="score")
    errors: list[dict[str, object]] = []
    cache_stats = {
        "hits": 0,
        "misses": 0,
        "writes": 0,
        "failed_misses": 0,
        "hits_without_usage": 0,
        "cached_prompt_tokens": 0,
        "cached_completion_tokens": 0,
        "cached_total_tokens": 0,
        "cached_requests": 0,
    }
    if max_workers == 1:
        for group_id, group in groups:
            group_scores, error, cache_event, cached_usage = _score_one_group(
                scorer,
                group_id,
                group,
                max_attempts=max_attempts,
                checkpoint_store=checkpoint_store,
                task_identity=task_identity,
                cell=cell,
                model=model,
                request_config=request_config,
            )
            scores.loc[group_scores.index] = group_scores
            if error is not None:
                errors.append(error)
            _record_cache_event(cache_stats, cache_event, cached_usage)
        return scores, errors, cache_stats

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _score_one_group,
                scorer,
                group_id,
                group,
                max_attempts=max_attempts,
                checkpoint_store=checkpoint_store,
                task_identity=task_identity,
                cell=cell,
                model=model,
                request_config=request_config,
            ): group_id
            for group_id, group in groups
        }
        for future in as_completed(futures):
            group_scores, error, cache_event, cached_usage = future.result()
            scores.loc[group_scores.index] = group_scores
            if error is not None:
                errors.append(error)
            _record_cache_event(cache_stats, cache_event, cached_usage)
    return scores, errors, cache_stats


def _score_one_group(
    scorer: Scorer,
    group_id: object,
    group: pd.DataFrame,
    *,
    max_attempts: int,
    checkpoint_store: GroupCheckpointStore | None,
    task_identity: str,
    cell: dict[str, object],
    model: str,
    request_config: dict[str, object],
) -> tuple[
    pd.Series,
    dict[str, object] | None,
    Literal["hit", "write", "failed_miss"],
    dict[str, int] | None,
]:
    user_ids = group["user_id"].drop_duplicates()
    if len(user_ids) != 1:
        raise ValueError("Candidate group contains multiple users")
    labels = [f"C{position}" for position in range(1, len(group) + 1)]
    messages = getattr(scorer, "_build_messages")(
        user_id=user_ids.iloc[0],
        candidates=group,
        labels=labels,
    )
    item_ids = group["item_id"].tolist()
    checkpoint_key = build_group_checkpoint_key(
        task_identity=task_identity,
        cell=cell,
        user_id=user_ids.iloc[0],
        candidate_group=group_id,
        candidate_item_ids=item_ids,
        prompt_messages=messages,
        model=model,
        request_config=request_config,
    )
    if checkpoint_store is not None:
        cached = checkpoint_store.load(
            checkpoint_key,
            expected_item_ids=item_ids,
        )
        if cached is not None:
            return (
                pd.Series(cached.scores, index=group.index, name="score"),
                None,
                "hit",
                cached.usage,
            )

    attempt_errors: list[str] = []
    for _ in range(max_attempts):
        try:
            group_scores = scorer.score(group)
        except Exception as error:  # noqa: BLE001 - preserve long-run progress.
            attempt_errors.append(repr(error))
            continue
        # A persistence failure after a successful paid response must abort
        # loudly. Treating it as a model failure could retry and pay twice.
        if checkpoint_store is not None:
            last_usage = getattr(scorer, "last_request_usage", lambda: None)()
            checkpoint_store.save(
                checkpoint_key,
                item_ids=item_ids,
                scores=group_scores.tolist(),
                usage=last_usage,
            )
        return group_scores, None, "write", None
    return (
        pd.Series(index=group.index, dtype=float, name="score"),
        {
            "candidate_group": str(group_id),
            "checkpoint_key": checkpoint_key,
            "attempts": max_attempts,
            "errors": attempt_errors,
        },
        "failed_miss",
        None,
    )


def _record_cache_event(
    cache_stats: dict[str, int],
    event: Literal["hit", "write", "failed_miss"],
    cached_usage: dict[str, int] | None,
) -> None:
    if event == "hit":
        cache_stats["hits"] += 1
        if cached_usage is None:
            cache_stats["hits_without_usage"] += 1
        else:
            cache_stats["cached_prompt_tokens"] += int(
                cached_usage.get("prompt_tokens", 0)
            )
            cache_stats["cached_completion_tokens"] += int(
                cached_usage.get("completion_tokens", 0)
            )
            cache_stats["cached_total_tokens"] += int(
                cached_usage.get("total_tokens", 0)
            )
            cache_stats["cached_requests"] += 1
        return
    cache_stats["misses"] += 1
    if event == "write":
        cache_stats["writes"] += 1
    else:
        cache_stats["failed_misses"] += 1


def _history_for_method(
    history: pd.DataFrame,
    spec: MethodSpec,
    user_ids: Sequence[object],
    *,
    training_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if spec.kind in {"user_mean", "user_ctr_smoothed", "llm"}:
        return history[history["user_id"].isin(user_ids)].copy()
    return (
        history.copy()
        if training_history is None
        else training_history.copy()
    )


def _select_target_users(
    targets: pd.DataFrame,
    user_ids: Sequence[object],
) -> pd.DataFrame:
    selected = targets[targets["user_id"].isin(user_ids)].copy()
    found = set(selected["user_id"])
    missing = [user_id for user_id in user_ids if user_id not in found]
    if missing:
        raise ValueError(
            f"Task seed is missing {len(missing)} selected target users"
        )
    return selected


def _validated_user_cohorts(
    user_ids: Sequence[object],
    control_user_ids: Sequence[object] | None,
) -> tuple[list[object], list[object]]:
    main_users = list(dict.fromkeys(user_ids))
    control_users = (
        list(main_users)
        if control_user_ids is None
        else list(dict.fromkeys(control_user_ids))
    )
    if not main_users:
        raise ValueError("user_ids must not be empty")
    unknown = set(control_users) - set(main_users)
    if unknown:
        raise ValueError(
            "control_user_ids must be a subset of user_ids; "
            f"unknown={len(unknown)}"
        )
    return main_users, control_users


def _users_for_cell(
    cell: CellSpec,
    *,
    main_users: Sequence[object],
    control_users: Sequence[object],
) -> Sequence[object]:
    if cell.history_control == "permuted":
        return control_users
    return main_users


def _split_model_xy(
    frame: pd.DataFrame,
    *,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Strip audit/provenance fields before every scorer sees the frame."""

    model_columns = [
        column
        for column in (
            "user_id",
            "item_id",
            "item_description",
            "item_category_path",
            "history_click",
            "candidate_group",
        )
        if column in frame.columns
    ]
    return (
        frame.loc[:, model_columns].copy(),
        frame[target_column].copy(),
    )


def _cell_dict(cell: CellSpec) -> dict[str, object]:
    return {
        "history_policy": cell.display_history_policy,
        "target_policy": cell.target_policy,
        "history_control": cell.history_control,
        "matched": cell.matched,
    }


def _values_sha256(values: Sequence[object]) -> str:
    import hashlib

    payload = json.dumps(
        [repr(value) for value in values],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def _current_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _git_provenance() -> dict[str, object]:
    root = repo_root()
    status = _git_output(["git", "status", "--porcelain=v1"], cwd=root)
    unstaged = _git_output(["git", "diff", "--binary"], cwd=root)
    staged = _git_output(["git", "diff", "--cached", "--binary"], cwd=root)
    return {
        "commit": _current_git_commit(),
        "dirty": bool(status.strip()) if status is not None else None,
        "status_porcelain": status.splitlines() if status is not None else None,
        "unstaged_diff_sha256": _sha256_text(unstaged) if unstaged is not None else None,
        "staged_diff_sha256": _sha256_text(staged) if staged is not None else None,
    }


def _git_output(command: list[str], *, cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout


def _request_config_sha256(spec: MethodSpec) -> str:
    payload = {
        "client_name": spec.client_name,
        "model": spec.model,
        "metadata_condition": spec.metadata_condition,
        "max_completion_tokens": spec.max_completion_tokens,
        "extra_body": spec.extra_body,
        "temperature": (
            None
            if (
                spec.client_name == "openai_vk_proxy"
                and str(spec.model).startswith("gpt-5")
            )
            else 0.0
        ),
        "reasoning_effort": (
            "none"
            if (
                spec.client_name == "openai_vk_proxy"
                and str(spec.model).startswith("gpt-5")
            )
            else None
        ),
    }
    return _sha256_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _task_identity(task: KuaiRandPolicyShiftTask) -> str:
    hashes = task.manifest.get("hashes", {})
    if isinstance(hashes, dict):
        identity = hashes.get("task_identity_sha256")
        if isinstance(identity, str) and identity:
            return identity
    # Toy and legacy tasks may not carry the canonical frame hash yet. Hashing
    # their complete manifest still prevents silent cross-task checkpoint reuse.
    return _sha256_text(
        json.dumps(
            _json_safe(task.manifest),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )


def _evaluation_protocol(
    tasks_by_seed: dict[int, KuaiRandPolicyShiftTask],
) -> dict[str, object]:
    kinds = {
        str(
            task.manifest.get("candidate_construction", {}).get(
                "kind",
                "unknown",
            )
        )
        for task in tasks_by_seed.values()
    }
    if len(kinds) != 1:
        raise ValueError(
            f"Task seeds mix candidate constructions: {sorted(kinds)}"
        )
    group_sizes: set[int] = set()
    for task in tasks_by_seed.values():
        for frame in (
            task.val_standard_targets,
            task.val_random_targets,
            task.test_standard_targets,
            task.test_random_targets,
        ):
            group_sizes.update(
                int(value)
                for value in frame.groupby("candidate_group").size().unique()
            )
    if len(group_sizes) != 1:
        raise ValueError(
            f"Task seeds do not have one fixed candidate size: {group_sizes}"
        )
    ranking_k = group_sizes.pop()
    kind = next(iter(kinds))
    natural = kind == "label_blind_natural_exposure_group"
    if natural:
        return {
            "candidate_construction": kind,
            "ranking_k": ranking_k,
            "require_mixed_labels": True,
            "run_protocol": (
                f"test_only_direct_natural_m{ranking_k}_response_prediction"
            ),
            "target_description": (
                "released binary is_click on label-blind natural exposure "
                f"groups of size {ranking_k}"
            ),
            "ranking_eligibility": (
                "conditional ranking diagnostics use groups containing both "
                "click labels; all constructed groups remain saved, enter "
                "probability metrics, and have ranking coverage reported"
            ),
            "known_caveats": [
                (
                    "Candidate rows are separately logged natural exposures, "
                    "not a simultaneous logged recommendation slate."
                ),
                (
                    "All-zero and all-one groups are retained but are "
                    "uninformative for conditional ranking metrics."
                ),
                (
                    "Ranking-only scorers cannot support pointwise calibration; "
                    "probability-producing scorers are evaluated on every row."
                ),
            ],
        }
    return {
        "candidate_construction": kind,
        "ranking_k": ranking_k,
        "require_mixed_labels": False,
        "run_protocol": "test_only_direct_c3_ranking",
        "target_description": (
            "released binary is_click; rank one observed click above two "
            "observed non-click exposures"
        ),
        "ranking_eligibility": (
            "all forced one-click/two-non-click groups are ranking-eligible"
        ),
        "known_caveats": [
            (
                "Candidate groups are c3 pseudo-lists assembled from separate "
                "observed exposures, not simultaneous logged slates."
            ),
            (
                "One-click/two-non-click conditioning supports ranking metrics "
                "but not natural-policy CTR calibration."
            ),
            (
                "The shared cohort is outcome-selected: users must have enough "
                "observed clicks and non-clicks to form c3 under both target "
                "policies."
            ),
        ],
    }


def _llm_request_config(scorer: Scorer) -> dict[str, object]:
    """Capture the exact non-message completion request for checkpoint keys."""

    config: dict[str, object] = {}
    temperature = getattr(scorer, "temperature", None)
    max_tokens = getattr(scorer, "max_tokens", None)
    if temperature is not None:
        config["temperature"] = temperature
    if max_tokens is not None:
        config["max_tokens"] = max_tokens
    config.update(dict(getattr(scorer, "request_kwargs", {})))
    extra_body = getattr(scorer, "extra_body", None)
    if extra_body:
        config["extra_body"] = extra_body
    return config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the KuaiRand standard/random exposure-policy shift benchmark."
        )
    )
    parser.add_argument(
        "--protocol",
        choices=("natural", "forced-c3"),
        default="natural",
        help=(
            "Natural is the locked label-blind H=5/m=5 task. forced-c3 "
            "reproduces the exploratory one-click/two-non-click pilot."
        ),
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help=f"Comma-separated methods. Available: {sorted(METHOD_REGISTRY)}",
    )
    parser.add_argument(
        "--task-seeds",
        default=None,
        help=(
            "Comma-separated candidate/task seeds. Defaults to 0,1,2 for "
            "natural and 42,43,44 for forced-c3."
        ),
    )
    parser.add_argument("--source-seed", type=int, default=DEFAULT_SOURCE_SEED)
    parser.add_argument(
        "--min-positive-source-users-per-policy",
        type=int,
        default=1,
        help=(
            "Require every candidate item to have at least this many positive "
            "source users under each exposure policy."
        ),
    )
    parser.add_argument(
        "--history-controls",
        default="auto",
        help=(
            "Comma-separated observed,permuted,none, or auto. Auto runs full "
            "controls for caption LLMs, observed+none for category-only LLMs, "
            "and observed for classical methods."
        ),
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=0,
        help="Outcome-blind shared-user cap; 0 keeps the full cross-seed cohort.",
    )
    parser.add_argument(
        "--max-control-users",
        type=int,
        default=0,
        help=(
            "Outcome-blind prefix used only for permuted-history cells; "
            "0 uses the complete selected main cohort."
        ),
    )
    parser.add_argument("--cohort-seed", type=int, default=DEFAULT_COHORT_SEED)
    parser.add_argument(
        "--hard-budget-usd",
        type=float,
        default=0.0,
        help=(
            "Optional strict guarded-cost stop; 0 disables blocking while still "
            "writing the complete preflight estimate."
        ),
    )
    parser.add_argument("--budget-safety-factor", type=float, default=1.10)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--execute-llm",
        action="store_true",
        help=(
            "Actually call configured LLM clients after full prompt/cost preflight. "
            "Without this flag LLM methods are preflight-only."
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Optional exact run directory. Reusing it resumes LLM groups from "
            "content-addressed checkpoints instead of paying for them again."
        ),
    )
    return parser.parse_args()


def _parse_names(raw: str, *, choices: dict[str, object]) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = sorted(set(names) - set(choices))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    return names


def _parse_ints(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one task seed is required")
    return list(dict.fromkeys(values))


def _parse_controls(raw: str) -> list[str]:
    controls = [value.strip() for value in raw.split(",") if value.strip()]
    planned_cells(controls)
    return controls


def _default_controls(spec: MethodSpec) -> list[str]:
    if not spec.is_llm:
        return ["observed"]
    if spec.metadata_condition == "caption_categories":
        return ["observed", "none", "permuted"]
    return ["observed", "none"]


if __name__ == "__main__":
    main()
