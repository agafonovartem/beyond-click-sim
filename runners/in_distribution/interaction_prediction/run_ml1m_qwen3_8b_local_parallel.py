from __future__ import annotations

"""Run the local MovieLens-1M Qwen3-8B ablation on multiple vLLM ports.

The method order is sequential. Within each method, task variants are run in
parallel and each active task checks out one local vLLM client from the pool.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from queue import Queue
import traceback
from typing import Any

from beyond_click_sim.tasks import Task
from runners.in_distribution.interaction_prediction.methods import agent4rec_yes_no
from runners.in_distribution.interaction_prediction.methods import llm_yes_no
from runners.in_distribution.interaction_prediction.run_ml1m_qwen3_8b_full import (
    DEFAULT_OUTPUT_ROOT,
    QWEN_EXTRA_BODY,
    QWEN_MODEL,
    TASTE_CLIENT,
    TASTE_MODEL,
    summarize_metrics,
    task_names,
    write_summary,
)
from runners.in_distribution.interaction_prediction.task_builders import (
    TASK_BUILDERS,
)


METHOD_ORDER = (
    "llm_yes_no_vllm_qwen3_8b_history_item_stats_full",
    "agent4rec_yes_no_vllm_qwen3_8b_taste_gpt4o_mini_full",
    "agent4rec_yes_no_vllm_qwen3_8b_traits_full",
    "agent4rec_yes_no_vllm_qwen3_8b_traits_taste_gpt4o_mini_full",
)
DEFAULT_RATIOS = (3, 9, 19)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_VLLM_CLIENTS = (
    "vllm_local",
    "vllm_local_8001",
    "vllm_local_8002",
    "vllm_local_8003",
)
DEFAULT_MAX_WORKERS_PER_TASK = 32


def main() -> None:
    args = parse_args()
    selected_tasks = _selected_task_names(args)
    selected_methods = _selected_methods(args)
    vllm_clients = tuple(_parse_str_list(args.vllm_clients))
    if not vllm_clients:
        raise ValueError("--vllm-clients must contain at least one client")
    if args.task_workers < 1:
        raise ValueError("--task-workers must be positive")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be positive")

    output_root = Path(args.output_root).expanduser().resolve()
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    batch_root = output_root / f"{run_id}_ml1m_qwen3_8b_local_parallel"

    print(f"batch_root={batch_root}", flush=True)
    print(f"tasks={selected_tasks}", flush=True)
    print(f"methods={selected_methods}", flush=True)
    print(f"vllm_clients={vllm_clients}", flush=True)
    print(
        f"task_workers={args.task_workers} max_workers_per_task={args.max_workers}",
        flush=True,
    )
    if args.dry_run:
        return

    batch_root.mkdir(parents=True, exist_ok=True)
    _write_batch_config(
        batch_root,
        {
            "run_id": run_id,
            "tasks": selected_tasks,
            "methods": selected_methods,
            "vllm_clients": list(vllm_clients),
            "task_workers": args.task_workers,
            "max_workers_per_task": args.max_workers,
            "qwen_model": QWEN_MODEL,
            "qwen_extra_body": QWEN_EXTRA_BODY,
            "taste_client": TASTE_CLIENT,
            "taste_model": TASTE_MODEL,
            "vllm_base_url_env": _vllm_base_url_env(),
            "method_order": "sequential",
            "task_order": "parallel_with_checked_out_vllm_client",
        },
    )

    summary: list[dict[str, Any]] = []
    total_runs = len(selected_tasks) * len(selected_methods)
    completed_runs = 0

    for method_name in selected_methods:
        print(f"METHOD start method={method_name}", flush=True)
        client_queue: Queue[str] = Queue()
        for client_name in vllm_clients:
            client_queue.put(client_name)

        with ThreadPoolExecutor(max_workers=args.task_workers) as executor:
            futures = [
                executor.submit(
                    _run_one_task_method,
                    task_name,
                    method_name,
                    batch_root,
                    client_queue,
                    args.skip_existing,
                    args.max_workers,
                )
                for task_name in selected_tasks
            ]
            for future in as_completed(futures):
                completed_runs += 1
                row = future.result()
                summary.append(row)
                write_summary(batch_root, summary)
                print(
                    f"RUN {completed_runs}/{total_runs} "
                    + json.dumps(row, sort_keys=True),
                    flush=True,
                )
        print(f"METHOD done method={method_name}", flush=True)

    write_summary(batch_root, summary)
    print(f"summary={batch_root / 'summary.json'}", flush=True)


def _run_one_task_method(
    task_name: str,
    method_name: str,
    batch_root: Path,
    client_queue: Queue[str],
    skip_existing: bool,
    max_workers: int,
) -> dict[str, Any]:
    output_dir = batch_root / task_name / method_name
    if skip_existing and (output_dir / "metrics.json").exists():
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        row = summarize_metrics(
            task_name=task_name,
            method_name=method_name,
            output_dir=output_dir,
            metrics=metrics,
        )
        row["status"] = "skipped_existing"
        return row

    client_name = client_queue.get()
    try:
        print(
            f"RUN start task={task_name} method={method_name} client={client_name}",
            flush=True,
        )
        task = TASK_BUILDERS[task_name]()
        metrics = _run_method(
            task,
            output_dir,
            method_name=method_name,
            client_name=client_name,
            max_workers=max_workers,
        )
        row = summarize_metrics(
            task_name=task_name,
            method_name=method_name,
            output_dir=output_dir,
            metrics=metrics,
        )
        row["client_name"] = client_name
        return row
    except Exception as error:  # noqa: BLE001 - keep the batch alive.
        output_dir.mkdir(parents=True, exist_ok=True)
        error_text = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        (output_dir / "run_error.txt").write_text(error_text, encoding="utf-8")
        return {
            "task": task_name,
            "method": method_name,
            "status": "error",
            "output_dir": str(output_dir),
            "client_name": client_name,
            "error": repr(error),
        }
    finally:
        client_queue.put(client_name)


def _run_method(
    task: Task,
    output_dir: Path,
    *,
    method_name: str,
    client_name: str,
    max_workers: int,
) -> dict[str, Any]:
    if method_name == "llm_yes_no_vllm_qwen3_8b_history_item_stats_full":
        return llm_yes_no.run_method(
            task,
            output_dir,
            method_name=method_name,
            client_name=client_name,
            model=QWEN_MODEL,
            max_candidate_groups=None,
            max_workers=max_workers,
            use_item_stats=True,
            extra_body=QWEN_EXTRA_BODY,
        )
    if method_name == "agent4rec_yes_no_vllm_qwen3_8b_taste_gpt4o_mini_full":
        return agent4rec_yes_no.run_method(
            task,
            output_dir,
            method_name=method_name,
            client_name=client_name,
            model=QWEN_MODEL,
            max_candidate_groups=None,
            max_workers=max_workers,
            extra_body=QWEN_EXTRA_BODY,
            profile_components=("taste",),
            taste_client_name=TASTE_CLIENT,
            taste_model=TASTE_MODEL,
            taste_temperature=0.0,
            taste_max_tokens=None,
        )
    if method_name == "agent4rec_yes_no_vllm_qwen3_8b_traits_full":
        return agent4rec_yes_no.run_method(
            task,
            output_dir,
            method_name=method_name,
            client_name=client_name,
            model=QWEN_MODEL,
            max_candidate_groups=None,
            max_workers=max_workers,
            extra_body=QWEN_EXTRA_BODY,
            profile_components=("traits",),
        )
    if method_name == "agent4rec_yes_no_vllm_qwen3_8b_traits_taste_gpt4o_mini_full":
        return agent4rec_yes_no.run_method(
            task,
            output_dir,
            method_name=method_name,
            client_name=client_name,
            model=QWEN_MODEL,
            max_candidate_groups=None,
            max_workers=max_workers,
            extra_body=QWEN_EXTRA_BODY,
            profile_components=("traits", "taste"),
            taste_client_name=TASTE_CLIENT,
            taste_model=TASTE_MODEL,
            taste_temperature=0.0,
            taste_max_tokens=None,
        )
    raise ValueError(f"Unsupported method: {method_name!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ratios",
        default=",".join(str(ratio) for ratio in DEFAULT_RATIOS),
        help="Comma-separated negative ratios to run.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated split/sampling seeds to run.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated explicit task names. Overrides --ratios/--seeds.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(METHOD_ORDER),
        help="Comma-separated method names. Default: requested four-method ablation.",
    )
    parser.add_argument(
        "--vllm-clients",
        default=",".join(DEFAULT_VLLM_CLIENTS),
        help="Comma-separated local vLLM client names to use as a checked-out pool.",
    )
    parser.add_argument(
        "--task-workers",
        type=int,
        default=len(DEFAULT_VLLM_CLIENTS),
        help="Maximum number of task variants to run concurrently per method.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS_PER_TASK,
        help="Candidate-group worker count inside each task variant.",
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


def _selected_methods(args: argparse.Namespace) -> list[str]:
    selected = _parse_str_list(args.methods)
    unknown = [name for name in selected if name not in METHOD_ORDER]
    if unknown:
        raise ValueError(f"Unknown method names: {unknown}. Available: {METHOD_ORDER}")
    return selected


def _parse_str_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(part) for part in _parse_str_list(raw)]


def _write_batch_config(batch_root: Path, payload: dict[str, Any]) -> None:
    (batch_root / "batch_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _vllm_base_url_env() -> dict[str, str | None]:
    return {
        "BEYOND_CLICK_SIM_VLLM_LOCAL_BASE_URL": os.environ.get(
            "BEYOND_CLICK_SIM_VLLM_LOCAL_BASE_URL"
        ),
        "BEYOND_CLICK_SIM_VLLM_LOCAL_8001_BASE_URL": os.environ.get(
            "BEYOND_CLICK_SIM_VLLM_LOCAL_8001_BASE_URL"
        ),
        "BEYOND_CLICK_SIM_VLLM_LOCAL_8002_BASE_URL": os.environ.get(
            "BEYOND_CLICK_SIM_VLLM_LOCAL_8002_BASE_URL"
        ),
        "BEYOND_CLICK_SIM_VLLM_LOCAL_8003_BASE_URL": os.environ.get(
            "BEYOND_CLICK_SIM_VLLM_LOCAL_8003_BASE_URL"
        ),
    }


if __name__ == "__main__":
    main()
