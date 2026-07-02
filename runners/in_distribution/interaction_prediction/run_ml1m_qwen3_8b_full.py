from __future__ import annotations

"""Run the MovieLens-1M interaction grid with Qwen3-8B LLM simulators.

This batch is intentionally explicit: all methods use the same item-stats task
variants so candidate sets/splits are matched, while methods that should not see
item statistics simply omit those prompt columns.
"""

import argparse
from collections.abc import Callable
from datetime import UTC, datetime
import json
from pathlib import Path
import traceback
from typing import Any

from beyond_click_sim.tasks import Task
from runners.in_distribution.interaction_prediction.methods import agent4rec_yes_no
from runners.in_distribution.interaction_prediction.methods import llm_yes_no
from runners.in_distribution.interaction_prediction.task_builders import (
    TASK_BUILDERS,
    repo_root,
)


RATIOS = (1, 2, 3, 9, 19)
SEEDS = (0, 1, 2)
QWEN_CLIENT = "vllm_local"
QWEN_MODEL = "Qwen/Qwen3-8B"
QWEN_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
TASTE_CLIENT = "openai"
TASTE_MODEL = "gpt-4o-mini"
MAX_WORKERS = 32

DEFAULT_OUTPUT_ROOT = (
    repo_root() / "outputs" / "in_distribution" / "interaction_prediction"
)


def task_names(*, ratios: tuple[int, ...] = RATIOS, seeds: tuple[int, ...] = SEEDS) -> list[str]:
    return [
        f"ml-1m_item_stats_cap20_eval_users1000_cg5_m{ratio}_seed{seed}"
        for ratio in ratios
        for seed in seeds
    ]


def method_runners() -> dict[str, Callable[[Task, Path], dict[str, Any]]]:
    return {
        "llm_yes_no_vllm_qwen3_8b_history_full": (
            lambda task, output_dir: llm_yes_no.run_method(
                task,
                output_dir,
                method_name="llm_yes_no_vllm_qwen3_8b_history_full",
                client_name=QWEN_CLIENT,
                model=QWEN_MODEL,
                max_candidate_groups=None,
                max_workers=MAX_WORKERS,
                use_item_stats=False,
                extra_body=QWEN_EXTRA_BODY,
            )
        ),
        "llm_yes_no_vllm_qwen3_8b_history_item_stats_full": (
            lambda task, output_dir: llm_yes_no.run_method(
                task,
                output_dir,
                method_name="llm_yes_no_vllm_qwen3_8b_history_item_stats_full",
                client_name=QWEN_CLIENT,
                model=QWEN_MODEL,
                max_candidate_groups=None,
                max_workers=MAX_WORKERS,
                use_item_stats=True,
                extra_body=QWEN_EXTRA_BODY,
            )
        ),
        "agent4rec_yes_no_vllm_qwen3_8b_taste_gpt4o_mini_full": (
            lambda task, output_dir: agent4rec_yes_no.run_method(
                task,
                output_dir,
                method_name="agent4rec_yes_no_vllm_qwen3_8b_taste_gpt4o_mini_full",
                client_name=QWEN_CLIENT,
                model=QWEN_MODEL,
                max_candidate_groups=None,
                max_workers=MAX_WORKERS,
                extra_body=QWEN_EXTRA_BODY,
                profile_components=("taste",),
                taste_client_name=TASTE_CLIENT,
                taste_model=TASTE_MODEL,
                taste_temperature=0.0,
                taste_max_tokens=None,
            )
        ),
        "agent4rec_yes_no_vllm_qwen3_8b_traits_full": (
            lambda task, output_dir: agent4rec_yes_no.run_method(
                task,
                output_dir,
                method_name="agent4rec_yes_no_vllm_qwen3_8b_traits_full",
                client_name=QWEN_CLIENT,
                model=QWEN_MODEL,
                max_candidate_groups=None,
                max_workers=MAX_WORKERS,
                extra_body=QWEN_EXTRA_BODY,
                profile_components=("traits",),
            )
        ),
        "agent4rec_yes_no_vllm_qwen3_8b_traits_taste_gpt4o_mini_full": (
            lambda task, output_dir: agent4rec_yes_no.run_method(
                task,
                output_dir,
                method_name=(
                    "agent4rec_yes_no_vllm_qwen3_8b_"
                    "traits_taste_gpt4o_mini_full"
                ),
                client_name=QWEN_CLIENT,
                model=QWEN_MODEL,
                max_candidate_groups=None,
                max_workers=MAX_WORKERS,
                extra_body=QWEN_EXTRA_BODY,
                profile_components=("traits", "taste"),
                taste_client_name=TASTE_CLIENT,
                taste_model=TASTE_MODEL,
                taste_temperature=0.0,
                taste_max_tokens=None,
            )
        ),
    }


def main() -> None:
    args = parse_args()
    selected_tasks = _selected_task_names(args)
    methods = method_runners()
    selected_methods = _selected_methods(args, methods)
    output_root = Path(args.output_root).expanduser().resolve()
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    batch_root = output_root / f"{run_id}_ml1m_qwen3_8b_interaction_full"

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
                row = {
                    "task": task_name,
                    "method": method_name,
                    "status": "skipped_existing",
                    "output_dir": str(output_dir),
                }
                summary.append(row)
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
            print(json.dumps(row, sort_keys=True), flush=True)

    write_summary(batch_root, summary)
    print(f"summary={batch_root / 'summary.json'}", flush=True)


def summarize_metrics(
    *,
    task_name: str,
    method_name: str,
    output_dir: Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    test = metrics["test"]
    micro = test["micro"]
    macro_user = test["macro_by_user_group_mean"]
    strict = metrics.get("test_failure_as_negative", {})
    strict_micro = strict.get("micro", {})
    coverage = metrics.get("coverage", {})
    return {
        "task": task_name,
        "method": method_name,
        "status": "ok",
        "output_dir": str(output_dir),
        "llm_errors": metrics.get("llm_errors", 0),
        "requested_rows": metrics.get("requested_rows"),
        "scored_rows": metrics.get("scored_rows"),
        "coverage": coverage,
        "macro_user_f1": macro_user["f1"],
        "micro_f1": micro["f1"],
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "strict_micro_f1": strict_micro.get("f1"),
    }


def write_summary(batch_root: Path, summary: list[dict[str, Any]]) -> None:
    (batch_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ratios",
        default=",".join(str(ratio) for ratio in RATIOS),
        help="Comma-separated negative ratios to run.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in SEEDS),
        help="Comma-separated split/sampling seeds to run.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated explicit task names. Overrides --ratios/--seeds.",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated method names. Default: all Qwen3-8B methods.",
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
        selected = task_names(
            ratios=tuple(_parse_int_list(args.ratios)),
            seeds=tuple(_parse_int_list(args.seeds)),
        )
    unknown = [name for name in selected if name not in TASK_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown task names: {unknown}")
    return selected


def _selected_methods(
    args: argparse.Namespace,
    methods: dict[str, Callable[[Task, Path], dict[str, Any]]],
) -> list[str]:
    selected = list(methods) if args.methods is None else _parse_str_list(args.methods)
    unknown = [name for name in selected if name not in methods]
    if unknown:
        raise ValueError(f"Unknown method names: {unknown}. Available: {list(methods)}")
    return selected


def _parse_str_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(part) for part in _parse_str_list(raw)]


if __name__ == "__main__":
    main()
