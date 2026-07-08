from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from time import perf_counter

from runners.in_distribution.policy_ranking_agreement.methods import METHOD_RUNNERS
from runners.in_distribution.policy_ranking_agreement.task_builders import (
    DEFAULT_TASK_NAMES,
    TASK_BUILDERS,
    repo_root,
)


DEFAULT_TASKS = DEFAULT_TASK_NAMES
DEFAULT_METHODS = ["popularity_scorer"]
DEFAULT_OUTPUT_DIR = repo_root() / "outputs" / "in_distribution" / "policy_ranking_agreement"


def run_one(task_name: str, method_name: str, output_root: Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    task_start = perf_counter()
    print(f"Building task: {task_name}", flush=True)
    task = TASK_BUILDERS[task_name]()
    print(f"Built task: {task_name} in {_elapsed(task_start)}s", flush=True)
    write_task_policy_metrics(output_root, task_name=task_name, task_manifest=task.manifest)
    output_dir = make_output_dir(output_root, task_name=task_name, method_name=method_name)
    method_start = perf_counter()
    print(f"Running method: {method_name} -> {output_dir}", flush=True)
    result = METHOD_RUNNERS[method_name](task, output_dir)
    print(f"Finished method: {method_name} in {_elapsed(method_start)}s", flush=True)
    return result


def is_completed(output_root: Path, task_name: str, method_name: str) -> bool:
    for d in output_root.glob(f"*_{task_name}_{method_name}"):
        if not d.is_dir():
            continue
        if (d / "metrics.json").exists():
            return True
    return False


def main() -> None:
    args = parse_args()
    tasks = parse_names(args.tasks, default=DEFAULT_TASKS, choices=TASK_BUILDERS)
    methods = parse_names(args.methods, default=DEFAULT_METHODS, choices=METHOD_RUNNERS)
    output_root = Path(args.output_dir).expanduser().resolve()
    total_runs = len(tasks) * len(methods)
    run_index = 0

    for task_name in tasks:
        task_start = perf_counter()
        task = None
        for method_name in methods:
            run_index += 1
            if args.resume and is_completed(output_root, task_name, method_name):
                print(f"Skipping (already done) {run_index}/{total_runs}: task={task_name}, method={method_name}", flush=True)
                continue
            if task is None:
                print(f"Building task: {task_name}", flush=True)
                task = TASK_BUILDERS[task_name]()
                print(f"Built task: {task_name} in {_elapsed(task_start)}s", flush=True)
                write_task_policy_metrics(
                    output_root,
                    task_name=task_name,
                    task_manifest=task.manifest,
                )
            print(f"Run {run_index}/{total_runs}: task={task_name}, method={method_name}", flush=True)
            output_dir = make_output_dir(
                output_root,
                task_name=task_name,
                method_name=method_name,
            )
            method_start = perf_counter()
            print(f"Running method: {method_name} -> {output_dir}", flush=True)
            result = METHOD_RUNNERS[method_name](task, output_dir)
            print(f"Finished method: {method_name} in {_elapsed(method_start)}s", flush=True)
            print(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run policy ranking agreement experiments.")
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task names. Default: all eval1000 policy ranking tasks.",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated method names. Default: popularity_scorer.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for run artifacts.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip task-method pairs that already have output in --output-dir.",
    )
    return parser.parse_args()


def parse_names(
    raw: str | None,
    *,
    default: list[str],
    choices: dict[str, object],
) -> list[str]:
    names = default if raw is None else [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in names if name not in choices]
    if unknown:
        raise ValueError(f"Unknown names: {unknown}. Available: {sorted(choices)}")
    return names


def make_output_dir(output_root: Path, *, task_name: str, method_name: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return output_root / f"{timestamp}_{task_name}_{method_name}"


def write_task_policy_metrics(
    output_root: Path,
    *,
    task_name: str,
    task_manifest: dict[str, object],
) -> None:
    payload = {
        "task": task_name,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "policy_recommendation_metrics": task_manifest.get("policy_recommendation_metrics"),
    }
    task_artifacts_dir = output_root / "_task_artifacts" / task_name
    task_artifacts_dir.mkdir(parents=True, exist_ok=True)
    (task_artifacts_dir / "policy_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _elapsed(start: float) -> float:
    return round(perf_counter() - start, 3)


if __name__ == "__main__":
    main()
