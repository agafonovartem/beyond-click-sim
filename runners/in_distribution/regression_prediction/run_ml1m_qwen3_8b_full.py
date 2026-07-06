from __future__ import annotations

"""Run the reduced ML-1M rating-regression grid with Qwen3-8B simulators."""

import argparse
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
import json
from pathlib import Path
from statistics import mean, stdev
import traceback
from typing import Any

from beyond_click_sim.tasks import Task
from runners.in_distribution.regression_prediction.methods import METHOD_RUNNERS
from runners.in_distribution.regression_prediction.task_builders import (
    TASK_BUILDERS,
    repo_root,
)


SEEDS = (0, 1, 2)
BASELINE_METHODS = (
    "mean_regressor",
    "mode_regressor",
    "item_mean_regressor",
    "item_mode_regressor",
    "user_mean_regressor",
    "user_mode_regressor",
)
QWEN_METHODS = (
    "llm_regressor_vllm_qwen3_8b_with_item_stats_full",
    "agent4rec_regressor_vllm_qwen3_8b_traits_full",
    "agent4rec_regressor_vllm_qwen3_8b_taste_gpt4o_mini_full",
    "agent4rec_regressor_vllm_qwen3_8b_traits_taste_gpt4o_mini_full",
)
DEFAULT_METHODS = (*BASELINE_METHODS, *QWEN_METHODS)
DEFAULT_OUTPUT_ROOT = (
    repo_root() / "outputs" / "in_distribution" / "regression_prediction"
)


def task_names(*, seeds: tuple[int, ...] = SEEDS) -> list[str]:
    return [
        f"ml-1m_rating_item_stats_eval_users1000_rows_per_user5_seed{seed}"
        for seed in seeds
    ]


def method_runners() -> dict[str, Callable[[Task, Path], dict[str, Any]]]:
    return {name: METHOD_RUNNERS[name] for name in DEFAULT_METHODS}


def main() -> None:
    args = parse_args()
    selected_tasks = _selected_task_names(args)
    methods = method_runners()
    selected_methods = _selected_methods(args, methods)
    output_root = Path(args.output_root).expanduser().resolve()
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    batch_root = output_root / f"{run_id}_ml1m_qwen3_8b_regression_full"

    print(f"batch_root={batch_root}", flush=True)
    print(f"tasks={selected_tasks}", flush=True)
    print(f"methods={selected_methods}", flush=True)
    if args.dry_run:
        return

    batch_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    total_runs = len(selected_tasks) * len(selected_methods)
    run_index = 0

    for task_name in selected_tasks:
        print(f"BUILD task={task_name}", flush=True)
        task = TASK_BUILDERS[task_name]()
        for method_name in selected_methods:
            run_index += 1
            output_dir = batch_root / task_name / method_name
            print(
                f"RUN {run_index}/{total_runs} task={task_name} method={method_name}",
                flush=True,
            )
            if args.skip_existing and (output_dir / "metrics.json").exists():
                metrics = read_existing_metrics(output_dir)
                row = summarize_metrics(
                    task_name=task_name,
                    method_name=method_name,
                    output_dir=output_dir,
                    metrics=metrics,
                )
                row["reused_existing"] = True
                summary.append(row)
                write_summary(batch_root, summary)
                write_aggregates(batch_root, summary)
                print(json.dumps(row, sort_keys=True), flush=True)
                continue

            try:
                metrics = methods[method_name](task, output_dir)
                row = summarize_metrics(
                    task_name=task_name,
                    method_name=method_name,
                    output_dir=output_dir,
                    metrics=metrics,
                )
            except Exception as error:  # noqa: BLE001 - keep batch alive.
                output_dir.mkdir(parents=True, exist_ok=True)
                error_text = "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                )
                (output_dir / "run_error.txt").write_text(error_text, encoding="utf-8")
                row = {
                    "task": task_name,
                    "method": method_name,
                    "status": "error",
                    "output_dir": str(output_dir),
                    "error": repr(error),
                }
            summary.append(row)
            write_summary(batch_root, summary)
            write_aggregates(batch_root, summary)
            print(json.dumps(row, sort_keys=True), flush=True)

    write_summary(batch_root, summary)
    write_aggregates(batch_root, summary)
    print(f"summary={batch_root / 'summary.json'}", flush=True)
    print(f"aggregate_summary={batch_root / 'aggregate_summary.json'}", flush=True)


def summarize_metrics(
    *,
    task_name: str,
    method_name: str,
    output_dir: Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    test = metrics["test"]
    micro = test["micro"]
    macro_user = test["macro_by_user_mean"]
    return {
        "task": task_name,
        "method": method_name,
        "status": "ok",
        "output_dir": str(output_dir),
        "llm_errors": metrics.get("llm_errors", 0),
        "requested_rows": metrics.get("requested_rows"),
        "scored_rows": metrics.get("scored_rows"),
        "coverage": metrics.get("coverage"),
        "macro_user_mae": macro_user["mae"],
        "macro_user_rmse": macro_user["rmse"],
        "micro_mae": micro["mae"],
        "micro_rmse": micro["rmse"],
    }


def read_existing_metrics(output_dir: Path) -> dict[str, Any]:
    return json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))


def write_summary(batch_root: Path, summary: list[dict[str, Any]]) -> None:
    (batch_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_aggregates(batch_root: Path, summary: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary:
        if row.get("status") == "ok":
            grouped[str(row["method"])].append(row)

    aggregates = []
    for method_name, rows in grouped.items():
        aggregate = {
            "method": method_name,
            "n": len(rows),
            "macro_user_mae_mean": _mean(row["macro_user_mae"] for row in rows),
            "macro_user_mae_std": _std(row["macro_user_mae"] for row in rows),
            "macro_user_rmse_mean": _mean(row["macro_user_rmse"] for row in rows),
            "macro_user_rmse_std": _std(row["macro_user_rmse"] for row in rows),
            "micro_mae_mean": _mean(row["micro_mae"] for row in rows),
            "micro_mae_std": _std(row["micro_mae"] for row in rows),
            "micro_rmse_mean": _mean(row["micro_rmse"] for row in rows),
            "micro_rmse_std": _std(row["micro_rmse"] for row in rows),
        }
        coverages = [
            float(row["coverage"])
            for row in rows
            if row.get("coverage") is not None
        ]
        if coverages:
            aggregate["coverage_mean"] = mean(coverages)
            aggregate["coverage_std"] = stdev(coverages) if len(coverages) > 1 else 0.0
        aggregates.append(aggregate)

    (batch_root / "aggregate_summary.json").write_text(
        json.dumps(aggregates, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in SEEDS),
        help="Comma-separated split/sampling seeds to run.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated explicit task names. Overrides --seeds.",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated method names. Default: baselines plus Qwen3-8B methods.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for batch artifacts.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional stable run id. Default: current UTC timestamp.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip method outputs that already contain metrics.json.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _selected_task_names(args: argparse.Namespace) -> list[str]:
    if args.tasks:
        selected = _parse_str_list(args.tasks)
    else:
        selected = task_names(seeds=tuple(_parse_int_list(args.seeds)))
    unknown = [name for name in selected if name not in TASK_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown task names: {unknown}")
    return selected


def _selected_methods(
    args: argparse.Namespace,
    methods: dict[str, Callable[[Task, Path], dict[str, Any]]],
) -> list[str]:
    selected = list(DEFAULT_METHODS) if args.methods is None else _parse_str_list(args.methods)
    unknown = [name for name in selected if name not in methods]
    if unknown:
        raise ValueError(f"Unknown method names: {unknown}. Available: {list(methods)}")
    return selected


def _parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(item) for item in _parse_str_list(raw)]


def _mean(values: Any) -> float:
    numbers = [float(value) for value in values]
    return mean(numbers)


def _std(values: Any) -> float:
    numbers = [float(value) for value in values]
    return stdev(numbers) if len(numbers) > 1 else 0.0


if __name__ == "__main__":
    main()
