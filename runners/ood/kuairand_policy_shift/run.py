from __future__ import annotations

import argparse
from collections.abc import Sequence
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
from beyond_click_sim.tasks import KuaiRandStandardToRandomTask
from runners.ood.kuairand_policy_shift.checkpoint import (
    CHECKPOINT_IDENTITY_VERSION,
    GroupCheckpointStore,
    build_group_checkpoint_key,
    build_legacy_group_checkpoint_key,
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
    PARSER_FORMAT_VERSION,
    SCORER_FORMAT_VERSION,
    MethodSpec,
    make_scorer,
)
from runners.ood.kuairand_policy_shift.metrics import (
    build_paired_comparisons,
    evaluate_predictions,
)
from runners.ood.kuairand_policy_shift.task_builders import (
    DEFAULT_CANDIDATE_SEEDS,
    build_tasks,
    load_canonical_dataset,
    repo_root,
    select_test_users,
    shared_test_users,
)


TargetPolicy = Literal["standard", "random"]
HistoryControl = Literal["observed", "none"]
TARGET_POLICIES: tuple[TargetPolicy, ...] = ("standard", "random")
DEFAULT_METHODS = tuple(METHOD_REGISTRY)
DEFAULT_OUTPUT_DIR = (
    repo_root() / "outputs" / "ood" / "kuairand_policy_shift_v2"
)
FIRST_SUITE_TARGET_POOL = "common_warm"
LEGACY_CHECKPOINT_CLIENT_PROVIDER = "openai_vk_proxy"


@dataclass(frozen=True)
class CellSpec:
    target_policy: TargetPolicy
    history_control: HistoryControl


def planned_cells(spec: MethodSpec) -> list[CellSpec]:
    """Return only cells supported by one fixed method specification."""

    return [
        CellSpec(target_policy, control)
        for control in spec.supported_controls
        for target_policy in TARGET_POLICIES
    ]


def prepare_llm_preflight(
    tasks_by_seed: dict[int, KuaiRandStandardToRandomTask],
    *,
    spec: MethodSpec,
    user_ids: Sequence[object],
    safety_factor: float = 1.10,
    token_counter: TokenCounter | None = None,
) -> tuple[CostEstimate, list[list[dict[str, str]]]]:
    """Materialize and price every test prompt without constructing a client."""

    if not spec.is_llm:
        raise ValueError("Only LLM methods have prompt preflights")
    users = _validated_users(user_ids)
    prompts: list[list[dict[str, str]]] = []
    no_call_client = object()
    for task_seed, task in sorted(tasks_by_seed.items()):
        for cell_spec in planned_cells(spec):
            cell = task.cell(
                split="test",
                target_policy=cell_spec.target_policy,
                history_control=cell_spec.history_control,
            )
            history = _history_for_method(cell, spec, users)
            targets = _select_target_users(cell.targets, users)
            X_history, y_history = _split_model_xy(
                history,
                target_column=cell.schema.target_column,
            )
            X_target, _ = _split_model_xy(
                targets,
                target_column=cell.schema.target_column,
            )
            scorer = make_scorer(
                spec,
                seed=task_seed,
                client=no_call_client,
            ).fit(X_history, y_history)
            build_messages = getattr(scorer, "_build_messages")
            for _, group in X_target.groupby("candidate_group", sort=False):
                labels = [
                    f"C{position}" for position in range(1, len(group) + 1)
                ]
                user_values = group["user_id"].drop_duplicates()
                if len(user_values) != 1:
                    raise ValueError("Candidate group contains multiple users")
                prompts.append(
                    build_messages(
                        user_id=user_values.iloc[0],
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
    tasks_by_seed: dict[int, KuaiRandStandardToRandomTask],
    *,
    method_name: str,
    output_dir: Path,
    user_ids: Sequence[object],
    execute_llm: bool = False,
    client: Any | None = None,
    hard_budget_usd: float | None = None,
    safety_factor: float = 1.10,
    token_counter: TokenCounter | None = None,
    preflight: CostEstimate | None = None,
    max_attempts: int = 1,
    max_workers: int = 1,
) -> dict[str, object]:
    """Run one method on the fixed test cohort and write reproducible artifacts."""

    if method_name not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method {method_name!r}; available={sorted(METHOD_REGISTRY)}"
        )
    if not tasks_by_seed:
        raise ValueError("tasks_by_seed must not be empty")
    users = _validated_users(user_ids)
    if max_attempts <= 0 or max_workers <= 0:
        raise ValueError("max_attempts and max_workers must be positive")

    spec = METHOD_REGISTRY[method_name]
    if spec.is_paid and max_attempts != 1:
        raise ValueError("Paid methods are locked to max_attempts=1")
    target_pools = {task.target_pool for task in tasks_by_seed.values()}
    if len(target_pools) != 1:
        raise ValueError("Task seeds mix target pools")
    target_pool = next(iter(target_pools))
    if target_pool != FIRST_SUITE_TARGET_POOL:
        raise ValueError(
            "The first KuaiRand method suite supports only the "
            f"{FIRST_SUITE_TARGET_POOL!r} target pool"
        )
    if not spec.supports_target_pool(target_pool):
        raise ValueError(
            f"Method {method_name!r} does not support pool {target_pool!r}"
        )
    if spec.is_llm and (
        not spec.client_name
        or not spec.scorer_format_version
        or not spec.parser_format_version
    ):
        raise ValueError(
            "LLM checkpoint identity requires explicit client, scorer, "
            "and parser versions"
        )

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = planned_cells(spec)
    cost_estimate = preflight
    prompts: list[list[dict[str, str]]] = []
    if spec.is_llm:
        if cost_estimate is None:
            cost_estimate, prompts = prepare_llm_preflight(
                tasks_by_seed,
                spec=spec,
                user_ids=users,
                safety_factor=safety_factor,
                token_counter=token_counter,
            )
        if hard_budget_usd is not None and hard_budget_usd > 0:
            enforce_hard_budget(
                cost_estimate,
                hard_budget_usd=hard_budget_usd,
            )
        _write_json(
            output_dir / "preflight_cost.json",
            cost_estimate.to_dict(),
        )
        if prompts:
            _write_json(
                output_dir / "prompt_example.json",
                {"messages": prompts[0]},
            )
        if not execute_llm:
            result = {
                "status": "preflight_only",
                "experiment": "kuairand_standard_to_random_v2",
                "method": spec.manifest_config(),
                "target_pool": target_pool,
                "test_only": True,
                "task_seeds": sorted(tasks_by_seed),
                "cohort": _cohort_manifest(users),
                "cells": [_cell_dict(cell) for cell in cells],
                "cost": cost_estimate.to_dict(),
            }
            _write_json(output_dir / "manifest.json", result)
            return result
        # Client construction happens only after every prompt has been built
        # and the cost estimate has been persisted.
        active_client = client or make_llm_client(str(spec.client_name))
    else:
        active_client = None

    prediction_parts: list[pd.DataFrame] = []
    errors: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    actual_usage = _empty_usage()
    cache_stats = _empty_cache_stats()
    checkpoint_store = (
        GroupCheckpointStore(output_dir / "group_checkpoints")
        if spec.is_llm
        else None
    )

    for task_seed, task in sorted(tasks_by_seed.items()):
        for cell_spec in cells:
            cell = task.cell(
                split="test",
                target_policy=cell_spec.target_policy,
                history_control=cell_spec.history_control,
            )
            history = _history_for_method(cell, spec, users)
            targets = _select_target_users(cell.targets, users)
            X_history, y_history = _split_model_xy(
                history,
                target_column=cell.schema.target_column,
            )
            X_target, y_target = _split_model_xy(
                targets,
                target_column=cell.schema.target_column,
            )
            scorer = make_scorer(
                spec,
                seed=task_seed,
                client=active_client,
            ).fit(X_history, y_history)

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
                        "target_pool": target_pool,
                        **_cell_dict(cell_spec),
                    },
                    model=str(spec.model),
                    request_config=_llm_request_config(scorer),
                    client_provider=str(spec.client_name),
                    scorer_format_version=str(spec.scorer_format_version),
                    parser_format_version=str(spec.parser_format_version),
                )
                errors.extend(
                    {
                        "task_seed": task_seed,
                        "target_pool": target_pool,
                        **_cell_dict(cell_spec),
                        **error,
                    }
                    for error in cell_errors
                )
                _add_counts(cache_stats, cell_cache)
                scorer_usage = getattr(scorer, "usage_", {})
                for field in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                ):
                    actual_usage[field] += int(scorer_usage.get(field, 0))
                actual_usage["requests"] += int(
                    getattr(scorer, "request_count_", 0)
                )
                fallback_rows = pd.Series(
                    False,
                    index=X_target.index,
                    dtype=bool,
                )
            else:
                scores = scorer.score(X_target)
                known_items = set(getattr(scorer, "item_scores_").index)
                fallback_rows = ~X_target["item_id"].isin(known_items)
                if target_pool == "common_warm" and fallback_rows.any():
                    raise AssertionError(
                        "Popularity encountered a cold item in common_warm"
                    )

            coverage.append(
                {
                    "task_seed": task_seed,
                    "target_pool": target_pool,
                    **_cell_dict(cell_spec),
                    "target_rows": int(len(X_target)),
                    "fallback_rows": int(fallback_rows.sum()),
                    "fallback_fraction": float(fallback_rows.mean()),
                }
            )
            prediction = targets.copy()
            prediction["target"] = y_target.to_numpy()
            prediction["score"] = scores.to_numpy()
            prediction["method_fallback"] = fallback_rows.to_numpy()
            prediction.insert(0, "method", method_name)
            prediction.insert(1, "task_seed", task_seed)
            prediction.insert(2, "split", "test")
            prediction["history_control"] = cell_spec.history_control
            prediction_parts.append(prediction)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    if spec.is_llm:
        successful_groups = cache_stats["hits"] + cache_stats["writes"]
        failed_groups = cache_stats["failed_misses"]
        planned_groups = successful_groups + failed_groups
        if errors and successful_groups == 0:
            run_status = "failed"
        elif errors:
            run_status = "complete_with_errors"
        else:
            run_status = "complete"
        group_outcomes: dict[str, int] | None = {
            "planned": planned_groups,
            "successful": successful_groups,
            "failed": failed_groups,
        }
    else:
        run_status = "complete"
        group_outcomes = None
    cell_metrics, per_group = evaluate_predictions(predictions)
    comparisons = build_paired_comparisons(per_group)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    per_group.to_parquet(
        output_dir / "per_group_metrics.parquet",
        index=False,
    )
    _write_json(output_dir / "metrics.json", {"cells": cell_metrics})
    _write_json(
        output_dir / "comparisons.json",
        {"comparisons": comparisons},
    )
    _write_jsonl(output_dir / "llm_errors.jsonl", errors)

    new_api_cost = (
        token_cost(
            actual_usage["prompt_tokens"],
            actual_usage["completion_tokens"],
            spec.input_price_per_million,
            spec.output_price_per_million,
        )
        if spec.is_llm
        else 0.0
    )
    cached_provider_cost = (
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
        "status": run_status,
        "experiment": "kuairand_standard_to_random_v2",
        "protocol": "kuairand-standard-to-random-v2",
        "method": spec.manifest_config(),
        "target_pool": target_pool,
        "target": (
            "released binary positive-interaction response after exposure"
        ),
        "source": "pre-intervention standard tab-1 exposures",
        "test_only": True,
        "validation_llm_calls": 0,
        "task_seeds": sorted(tasks_by_seed),
        "cohort": _cohort_manifest(users),
        "cells": [_cell_dict(cell) for cell in cells],
        "metrics": {
            "primary": ["ndcg@5", "hit_rate@1"],
            "secondary": ["ndcg@10", "hit_rate@5"],
            "eligibility": "mixed-label candidate groups only",
            "all_zero_and_all_positive": (
                "materialized but excluded from ranking metrics"
            ),
            "tie_policy": "average over score-consistent permutations",
            "seed_aggregation": (
                "candidate seeds averaged within user before inference"
            ),
            "failed_llm_group_sensitivity": "failure_as_zero metrics",
        },
        "coverage": coverage,
        "task_manifests": {
            str(seed): task.manifest
            for seed, task in sorted(tasks_by_seed.items())
        },
        "llm": {
            "preflight": (
                cost_estimate.to_dict()
                if cost_estimate is not None
                else None
            ),
            "new_api_usage": actual_usage if spec.is_llm else None,
            "new_api_cost_usd": new_api_cost if spec.is_llm else None,
            "group_checkpoint_cache": (
                {
                    **cache_stats,
                    "root": "group_checkpoints",
                    "content_addressed": True,
                    "identity_version": CHECKPOINT_IDENTITY_VERSION,
                    "legacy_read_fallback": (
                        "exact-v1-key migration for the declared "
                        "provider/scorer/parser only"
                    ),
                }
                if spec.is_llm
                else None
            ),
            "group_outcomes": group_outcomes,
            "cached_provider_cost_usd": (
                cached_provider_cost if spec.is_llm else None
            ),
            "provider_cost_including_cached_usd": (
                new_api_cost + cached_provider_cost
                if spec.is_llm
                else None
            ),
            "hard_budget_usd": (
                hard_budget_usd if spec.is_paid else None
            ),
            "max_attempts": max_attempts if spec.is_llm else None,
            "max_workers": max_workers if spec.is_llm else None,
            "errors": len(errors),
        },
        "artifacts": {
            "predictions": "predictions.parquet",
            "per_group_metrics": "per_group_metrics.parquet",
            "cell_metrics": "metrics.json",
            "paired_comparisons": "comparisons.json",
            "llm_errors": "llm_errors.jsonl",
            "group_checkpoints": (
                "group_checkpoints/" if spec.is_llm else None
            ),
        },
        "request_config_sha256": _request_config_sha256(spec),
        "prompt_plan_sha256": (
            cost_estimate.prompt_plan_sha256
            if cost_estimate is not None
            else None
        ),
        "git": _git_provenance(),
        "known_caveats": [
            (
                "Candidate groups contain separate logged exposures rather "
                "than simultaneous production slates."
            ),
            (
                "Random targets differ from source both temporally and by "
                "exposure regime."
            ),
            (
                "Standard and random raw metric levels are not directly "
                "comparable because their candidate distributions differ."
            ),
            (
                "Popularity is the source positive-count statistic, not CTR."
            ),
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return {
        "status": run_status,
        "method": method_name,
        "n_predictions": int(len(predictions)),
        "n_errors": len(errors),
        "cells": cell_metrics,
        "new_api_cost_usd": new_api_cost,
        "provider_cost_including_cached_usd": (
            new_api_cost + cached_provider_cost
        ),
    }


def write_combined_analysis(
    run_root: Path,
    *,
    method_names: Sequence[str] = DEFAULT_METHODS,
) -> dict[str, object]:
    """Validate target alignment and compare GPT with Pop on the same groups."""

    method_statuses = _method_run_statuses(run_root, method_names)
    incomplete = {
        method: status
        for method, status in method_statuses.items()
        if status != "complete"
    }
    if incomplete:
        raise ValueError(
            "Combined analysis requires complete method runs; "
            f"incomplete={incomplete}"
        )

    parts = []
    for method_name in method_names:
        path = run_root / method_name / "predictions.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        parts.append(pd.read_parquet(path))
    predictions = pd.concat(parts, ignore_index=True)
    _validate_cross_method_targets(predictions)
    cells, per_group = evaluate_predictions(predictions)
    comparisons = build_paired_comparisons(per_group)
    per_group.to_parquet(
        run_root / "combined_per_group_metrics.parquet",
        index=False,
    )
    result = {
        "status": "complete",
        "methods": list(method_names),
        "metrics": {
            "primary": ["ndcg@5", "hit_rate@1"],
            "secondary": ["ndcg@10", "hit_rate@5"],
            "cells": cells,
        },
        "comparisons": comparisons,
    }
    _write_json(run_root / "combined_analysis.json", result)
    return result


def _method_run_statuses(
    run_root: Path,
    method_names: Sequence[str],
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for method_name in method_names:
        manifest_path = run_root / method_name / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        status = payload.get("status")
        statuses[method_name] = (
            status if isinstance(status, str) else "missing"
        )
    return statuses


def main() -> None:
    args = _parse_args()
    method_names = _parse_names(args.methods, choices=METHOD_REGISTRY)
    candidate_seeds = _parse_ints(args.candidate_seeds)
    dataset = load_canonical_dataset(
        args.canonical_root
        if args.canonical_root is not None
        else None
    )
    tasks_by_seed = build_tasks(
        candidate_seeds=candidate_seeds,
        target_pool=FIRST_SUITE_TARGET_POOL,
        dataset=dataset,
        translation_dir=args.translation_dir,
    )
    cohort = select_test_users(
        shared_test_users(tasks_by_seed),
        max_users=args.max_users,
        seed=args.subset_seed,
    )
    if not cohort:
        raise RuntimeError("No test users selected")

    run_root = args.output_dir.expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_root / "task_audit.json",
        {
            "status": "materialized_before_model_scoring",
            "protocol": "kuairand-standard-to-random-v2",
            "target_pool": FIRST_SUITE_TARGET_POOL,
            "task_seeds": sorted(tasks_by_seed),
            "selected_test_cohort": _cohort_manifest(cohort),
            "task_manifests": {
                str(seed): task.manifest
                for seed, task in sorted(tasks_by_seed.items())
            },
        },
    )
    results: dict[str, object] = {}
    for method_name in method_names:
        results[method_name] = run_method_suite(
            tasks_by_seed,
            method_name=method_name,
            output_dir=run_root / method_name,
            user_ids=cohort,
            execute_llm=args.execute_llm,
            hard_budget_usd=(
                args.hard_budget_usd
                if args.hard_budget_usd > 0
                else None
            ),
            safety_factor=args.budget_safety_factor,
            max_attempts=args.max_attempts,
            max_workers=args.max_workers,
        )
    completed = all(
        result.get("status") == "complete"
        for result in results.values()
        if isinstance(result, dict)
    )
    if completed and set(method_names) == set(DEFAULT_METHODS):
        results["combined_analysis"] = write_combined_analysis(
            run_root,
            method_names=DEFAULT_METHODS,
        )
    elif set(method_names) == set(DEFAULT_METHODS):
        skipped = {
            "status": "skipped",
            "reason": "combined analysis requires every method status=complete",
            "method_statuses": {
                method_name: (
                    result.get("status")
                    if isinstance(result, dict)
                    else "unknown"
                )
                for method_name, result in results.items()
            },
        }
        results["combined_analysis"] = skipped
        _write_json(run_root / "combined_analysis.json", skipped)
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
    client_provider: str,
    scorer_format_version: str,
    parser_format_version: str,
) -> tuple[pd.Series, list[dict[str, object]], dict[str, int]]:
    groups = list(X.groupby("candidate_group", sort=False))
    scores = pd.Series(index=X.index, dtype=float, name="score")
    errors: list[dict[str, object]] = []
    cache_stats = _empty_cache_stats()
    if max_workers == 1:
        for group_id, group in groups:
            group_scores, error, event, cached_usage = _score_one_group(
                scorer,
                group_id,
                group,
                max_attempts=max_attempts,
                checkpoint_store=checkpoint_store,
                task_identity=task_identity,
                cell=cell,
                model=model,
                request_config=request_config,
                client_provider=client_provider,
                scorer_format_version=scorer_format_version,
                parser_format_version=parser_format_version,
            )
            scores.loc[group_scores.index] = group_scores
            if error is not None:
                errors.append(error)
            _record_cache_event(cache_stats, event, cached_usage)
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
                client_provider=client_provider,
                scorer_format_version=scorer_format_version,
                parser_format_version=parser_format_version,
            ): group_id
            for group_id, group in groups
        }
        for future in as_completed(futures):
            group_scores, error, event, cached_usage = future.result()
            scores.loc[group_scores.index] = group_scores
            if error is not None:
                errors.append(error)
            _record_cache_event(cache_stats, event, cached_usage)
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
    client_provider: str,
    scorer_format_version: str,
    parser_format_version: str,
) -> tuple[
    pd.Series,
    dict[str, object] | None,
    Literal["hit", "legacy_migration", "write", "failed_miss"],
    dict[str, int] | None,
]:
    user_values = group["user_id"].drop_duplicates()
    if len(user_values) != 1:
        raise ValueError("Candidate group contains multiple users")
    labels = [f"C{position}" for position in range(1, len(group) + 1)]
    messages = getattr(scorer, "_build_messages")(
        user_id=user_values.iloc[0],
        candidates=group,
        labels=labels,
    )
    item_ids = group["item_id"].tolist()
    checkpoint_key = build_group_checkpoint_key(
        task_identity=task_identity,
        cell=cell,
        user_id=user_values.iloc[0],
        candidate_group=group_id,
        candidate_item_ids=item_ids,
        prompt_messages=messages,
        model=model,
        request_config=request_config,
        client_provider=client_provider,
        scorer_format_version=scorer_format_version,
        parser_format_version=parser_format_version,
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
        if _legacy_checkpoint_read_is_safe(
            client_provider=client_provider,
            scorer_format_version=scorer_format_version,
            parser_format_version=parser_format_version,
        ):
            legacy_key = build_legacy_group_checkpoint_key(
                task_identity=task_identity,
                cell=cell,
                user_id=user_values.iloc[0],
                candidate_group=group_id,
                candidate_item_ids=item_ids,
                prompt_messages=messages,
                model=model,
                request_config=request_config,
            )
            legacy = checkpoint_store.load(
                legacy_key,
                expected_item_ids=item_ids,
            )
            if legacy is not None:
                migrated = checkpoint_store.save(
                    checkpoint_key,
                    item_ids=item_ids,
                    scores=legacy.scores,
                    usage=legacy.usage,
                )
                return (
                    pd.Series(
                        migrated.scores,
                        index=group.index,
                        name="score",
                    ),
                    None,
                    "legacy_migration",
                    migrated.usage,
                )

    attempt_errors: list[str] = []
    for _ in range(max_attempts):
        try:
            group_scores = scorer.score(group)
        except Exception as error:  # noqa: BLE001 - preserve long-run progress.
            attempt_errors.append(repr(error))
            continue
        if checkpoint_store is not None:
            usage = getattr(scorer, "last_request_usage", lambda: None)()
            checkpoint_store.save(
                checkpoint_key,
                item_ids=item_ids,
                scores=group_scores.tolist(),
                usage=usage,
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
    event: Literal["hit", "legacy_migration", "write", "failed_miss"],
    cached_usage: dict[str, int] | None,
) -> None:
    if event in {"hit", "legacy_migration"}:
        cache_stats["hits"] += 1
        if event == "legacy_migration":
            cache_stats["legacy_hits"] += 1
            cache_stats["legacy_migrations"] += 1
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
    cell: Any,
    spec: MethodSpec,
    user_ids: Sequence[object],
) -> pd.DataFrame:
    if spec.history_scope == "full_standard_source":
        return cell.training_source.copy()
    return cell.history.loc[cell.history["user_id"].isin(user_ids)].copy()


def _select_target_users(
    targets: pd.DataFrame,
    user_ids: Sequence[object],
) -> pd.DataFrame:
    selected = targets.loc[targets["user_id"].isin(user_ids)].copy()
    found = set(selected["user_id"])
    missing = [user_id for user_id in user_ids if user_id not in found]
    if missing:
        raise ValueError(
            f"Task is missing {len(missing)} selected target users"
        )
    return selected


def _split_model_xy(
    frame: pd.DataFrame,
    *,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Pass only explicit model-visible columns to every scorer."""

    model_columns = [
        column
        for column in (
            "user_id",
            "item_id",
            "item_caption_en",
            "item_cover_text_en",
            "item_category_path_en",
            "history_click",
            "candidate_group",
        )
        if column in frame.columns
    ]
    return (
        frame.loc[:, model_columns].copy(),
        frame[target_column].copy(),
    )


def _validate_cross_method_targets(predictions: pd.DataFrame) -> None:
    identity = [
        "task_seed",
        "split",
        "target_pool",
        "target_policy",
        "user_id",
        "candidate_group",
        "item_id",
        "target",
    ]
    reference = (
        predictions.loc[
            predictions["method"].eq("popularity"),
            identity,
        ]
        .sort_values(identity[:-1], kind="mergesort")
        .reset_index(drop=True)
    )
    if reference.empty:
        raise ValueError("Combined analysis requires popularity predictions")
    conditions = (
        ("llm_openai_vk_gpt54_mini_listwise_english", "observed"),
        ("llm_openai_vk_gpt54_mini_listwise_english", "none"),
    )
    for method, control in conditions:
        candidate = (
            predictions.loc[
                predictions["method"].eq(method)
                & predictions["history_control"].eq(control),
                identity,
            ]
            .sort_values(identity[:-1], kind="mergesort")
            .reset_index(drop=True)
        )
        try:
            pd.testing.assert_frame_equal(reference, candidate)
        except AssertionError as error:
            raise ValueError(
                f"Target rows differ for {method}/{control}"
            ) from error


def _validated_users(user_ids: Sequence[object]) -> list[object]:
    users = list(dict.fromkeys(user_ids))
    if not users:
        raise ValueError("user_ids must not be empty")
    return users


def _cell_dict(cell: CellSpec) -> dict[str, object]:
    return {
        "source_policy": "standard",
        "target_policy": cell.target_policy,
        "history_control": cell.history_control,
        "target_interpretation": (
            "temporal_reference"
            if cell.target_policy == "standard"
            else "temporal_plus_exposure_regime_ood"
        ),
    }


def _cohort_manifest(users: Sequence[object]) -> dict[str, object]:
    return {
        "n_users": len(users),
        "user_ids_sha256": _values_sha256(users),
        "selection_uses_target_labels": False,
    }


def _empty_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "requests": 0,
    }


def _empty_cache_stats() -> dict[str, int]:
    return {
        "hits": 0,
        "misses": 0,
        "writes": 0,
        "failed_misses": 0,
        "legacy_hits": 0,
        "legacy_migrations": 0,
        "hits_without_usage": 0,
        "cached_prompt_tokens": 0,
        "cached_completion_tokens": 0,
        "cached_total_tokens": 0,
        "cached_requests": 0,
    }


def _add_counts(
    destination: dict[str, int],
    source: dict[str, int],
) -> None:
    for key, value in source.items():
        destination[key] += int(value)


def _task_identity(task: KuaiRandStandardToRandomTask) -> str:
    identity = task.manifest.get("hashes", {}).get("task_identity_sha256")
    if not isinstance(identity, str) or not identity:
        raise ValueError("Task manifest lacks task_identity_sha256")
    return identity


def _llm_request_config(scorer: Scorer) -> dict[str, object]:
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


def _legacy_checkpoint_read_is_safe(
    *,
    client_provider: str,
    scorer_format_version: str,
    parser_format_version: str,
) -> bool:
    """Allow legacy reads only for the exact paid-run implementation."""

    return (
        client_provider == LEGACY_CHECKPOINT_CLIENT_PROVIDER
        and scorer_format_version == SCORER_FORMAT_VERSION
        and parser_format_version == PARSER_FORMAT_VERSION
    )


def _request_config_sha256(spec: MethodSpec) -> str:
    payload = {
        "client_name": spec.client_name,
        "model": spec.model,
        "scorer_format_version": spec.scorer_format_version,
        "parser_format_version": spec.parser_format_version,
        "history_scope": spec.history_scope,
        "metadata_condition": spec.metadata_condition,
        "max_completion_tokens": spec.max_completion_tokens,
        "extra_body": spec.extra_body,
        "reasoning_effort": "none" if spec.is_llm else None,
    }
    return _sha256_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )


def _git_provenance() -> dict[str, object]:
    root = repo_root()
    status = _git_output(["git", "status", "--porcelain=v1"], cwd=root)
    return {
        "commit": _git_output(["git", "rev-parse", "HEAD"], cwd=root),
        "dirty": bool(status.strip()) if status is not None else None,
        "status_porcelain": status.splitlines() if status is not None else None,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
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
    return completed.stdout.strip()


def _values_sha256(values: Sequence[object]) -> str:
    payload = json.dumps(
        [repr(value) for value in values],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    records: Sequence[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run KuaiRand pre-intervention standard source against future "
            "standard/random natural exposures."
        )
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help=f"Comma-separated methods: {sorted(METHOD_REGISTRY)}",
    )
    parser.add_argument(
        "--candidate-seeds",
        default=",".join(str(seed) for seed in DEFAULT_CANDIDATE_SEEDS),
    )
    parser.add_argument("--canonical-root", type=Path)
    parser.add_argument("--translation-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-users",
        type=int,
        default=0,
        help="Outcome-blind test-user cap; 0 keeps all 500 users.",
    )
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--budget-safety-factor", type=float, default=1.10)
    parser.add_argument(
        "--hard-budget-usd",
        type=float,
        default=0.0,
        help="Optional explicit preflight guard; 0 disables it.",
    )
    parser.add_argument(
        "--execute-llm",
        action="store_true",
        help="Call VK proxy after full prompt/cost preflight.",
    )
    return parser.parse_args()


def _parse_names(
    raw: str,
    *,
    choices: dict[str, object],
) -> list[str]:
    names = list(dict.fromkeys(part.strip() for part in raw.split(",")))
    unknown = sorted(set(names) - set(choices))
    if unknown:
        raise ValueError(f"Unknown names: {unknown}")
    return names


def _parse_ints(raw: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in raw.split(",")]
    except ValueError as error:
        raise ValueError(f"Invalid integer list: {raw!r}") from error
    values = list(dict.fromkeys(values))
    if not values:
        raise ValueError("Integer list must not be empty")
    return values


if __name__ == "__main__":
    main()
