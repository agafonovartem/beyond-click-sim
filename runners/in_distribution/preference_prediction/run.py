from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from runners.in_distribution.preference_prediction.methods import METHOD_RUNNERS
from runners.in_distribution.preference_prediction.task_builders import (
    DEFAULT_TASK_NAMES,
    TASK_BUILDERS,
    repo_root,
)


DEFAULT_TASKS = DEFAULT_TASK_NAMES
DEFAULT_METHODS = ["popularity_f1_threshold"]
DEFAULT_OUTPUT_DIR = repo_root() / "outputs" / "in_distribution" / "preference_prediction"


def run_one(
    task_name: str,
    method_name: str,
    output_root: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    task_start = perf_counter()
    print(f"Building task: {task_name}", flush=True)
    task = TASK_BUILDERS[task_name]()
    print(f"Built task: {task_name} in {_elapsed(task_start)}s", flush=True)
    output_dir = make_output_dir(
        output_root,
        task_name=task_name,
        method_name=method_name,
    )
    method_start = perf_counter()
    print(f"Running method: {method_name} -> {output_dir}", flush=True)
    result = METHOD_RUNNERS[method_name](task, output_dir)
    print(f"Finished method: {method_name} in {_elapsed(method_start)}s", flush=True)
    return result


def main() -> None:
    args = parse_args()
    tasks = parse_names(args.tasks, default=DEFAULT_TASKS, choices=TASK_BUILDERS)
    methods = parse_names(args.methods, default=DEFAULT_METHODS, choices=METHOD_RUNNERS)
    output_root = Path(args.output_dir).expanduser().resolve()
    total_runs = len(tasks) * len(methods)
    run_index = 0

    for task_name in tasks:
        task_start = perf_counter()
        print(f"Building task: {task_name}", flush=True)
        task = TASK_BUILDERS[task_name]()
        print(f"Built task: {task_name} in {_elapsed(task_start)}s", flush=True)
        for method_name in methods:
            run_index += 1
            print(
                f"Run {run_index}/{total_runs}: "
                f"task={task_name}, method={method_name}",
                flush=True,
            )
            output_dir = make_output_dir(
                output_root,
                task_name=task_name,
                method_name=method_name,
            )
            method_start = perf_counter()
            print(f"Running method: {method_name} -> {output_dir}", flush=True)
            result = METHOD_RUNNERS[method_name](task, output_dir)
            print(
                f"Finished method: {method_name} in {_elapsed(method_start)}s",
                flush=True,
            )
            print(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task names. Default: all preference tasks.",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated method names. Default: popularity_f1_threshold.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for run artifacts.",
    )
    return parser.parse_args()


def parse_names(
    raw: str | None,
    *,
    default: list[str],
    choices: dict[str, object],
) -> list[str]:
    names = default if raw is None else [
        name.strip()
        for name in raw.split(",")
        if name.strip()
    ]
    unknown = [name for name in names if name not in choices]
    if unknown:
        raise ValueError(f"Unknown names: {unknown}. Available: {sorted(choices)}")
    return names


def make_output_dir(output_root: Path, *, task_name: str, method_name: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return output_root / f"{timestamp}_{task_name}_{method_name}"


def _elapsed(start: float) -> float:
    return round(perf_counter() - start, 3)


if __name__ == "__main__":
    main()
