from __future__ import annotations

import argparse
import re
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from runners.in_distribution.cold_start.methods import METHOD_RUNNERS
from runners.in_distribution.cold_start.task_builders import (
    DEFAULT_TASK_NAMES,
    TASK_BUILDERS,
)
from runners.in_distribution.interaction_prediction.task_builders import repo_root


DEFAULT_METHODS = ["popularity_f1_threshold"]
DEFAULT_OUTPUT_DIR = repo_root() / "outputs" / "in_distribution" / "cold_start"
_RUN_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def run_one(
    task_name: str,
    method_name: str,
    output_root: Path = DEFAULT_OUTPUT_DIR,
    *,
    run_token: str | None = None,
) -> dict[str, object]:
    task_start = perf_counter()
    print(f"Building task: {task_name}", flush=True)
    task = TASK_BUILDERS[task_name]()
    print(f"Built task: {task_name} in {_elapsed(task_start)}s", flush=True)
    output_dir = make_output_dir(
        output_root,
        task_name=task_name,
        method_name=method_name,
        run_token=run_token,
    )
    method_start = perf_counter()
    print(f"Running method: {method_name} -> {output_dir}", flush=True)
    print(f"OUTPUT_DIR={output_dir}", flush=True)
    result = METHOD_RUNNERS[method_name](task, output_dir)
    print(f"Finished method: {method_name} in {_elapsed(method_start)}s", flush=True)
    return result


def required_completion_files(method_name: str) -> tuple[str, ...]:
    """Method-aware completion artifacts for resume / orchestration."""
    if method_name.startswith(
        (
            "llm_preference_listwise_ranking_",
            "agent4rec_preference_listwise_ranking_",
        )
    ):
        return ("manifest.json", "metrics_ranking.json")
    if method_name.endswith("_ranking") or "_ranking_" in method_name:
        return ("manifest.json", "metrics_ranking.json")
    if method_name.startswith(("llm_yes_no_", "agent4rec_yes_no_")):
        return ("manifest.json", "metrics.json", "metrics_ranking.json")
    return ("manifest.json", "metrics.json")


def is_completed(output_root: Path, task_name: str, method_name: str) -> bool:
    required = required_completion_files(method_name)
    for d in output_root.glob(f"*_{task_name}_{method_name}"):
        if not d.is_dir():
            continue
        if all((d / filename).exists() for filename in required):
            return True
    return False


def main() -> None:
    args = parse_args()
    tasks = parse_names(args.tasks, default=DEFAULT_TASK_NAMES, choices=TASK_BUILDERS)
    methods = parse_names(args.methods, default=DEFAULT_METHODS, choices=METHOD_RUNNERS)
    output_root = Path(args.output_dir).expanduser().resolve()
    run_token = validate_run_token(args.run_token)
    total_runs = len(tasks) * len(methods)
    run_index = 0

    for task_name in tasks:
        task_start = perf_counter()
        task = None
        for method_name in methods:
            run_index += 1
            if args.resume and is_completed(output_root, task_name, method_name):
                print(
                    f"Skipping (already done) {run_index}/{total_runs}: "
                    f"task={task_name}, method={method_name}",
                    flush=True,
                )
                continue
            if task is None:
                print(f"Building task: {task_name}", flush=True)
                task = TASK_BUILDERS[task_name]()
                print(f"Built task: {task_name} in {_elapsed(task_start)}s", flush=True)
            print(
                f"Run {run_index}/{total_runs}: task={task_name}, method={method_name}",
                flush=True,
            )
            output_dir = make_output_dir(
                output_root,
                task_name=task_name,
                method_name=method_name,
                run_token=run_token,
            )
            method_start = perf_counter()
            print(f"Running method: {method_name} -> {output_dir}", flush=True)
            print(f"OUTPUT_DIR={output_dir}", flush=True)
            result = METHOD_RUNNERS[method_name](task, output_dir)
            print(f"Finished method: {method_name} in {_elapsed(method_start)}s", flush=True)
            print(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task names. Default: all cold-start tasks.",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip task-method pairs that already have output in --output-dir.",
    )
    parser.add_argument(
        "--run-token",
        default=None,
        help=(
            "Optional unique token inserted into the output directory name as "
            "{timestamp}_{run_token}_{task}_{method}. Use to prevent same-second "
            "collisions under concurrent retries."
        ),
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


def validate_run_token(run_token: str | None) -> str | None:
    if run_token is None:
        return None
    token = run_token.strip()
    if not _RUN_TOKEN_RE.fullmatch(token):
        raise ValueError(
            "run_token must be 1-128 chars of [A-Za-z0-9._-] and start with "
            f"an alphanumeric character; got {run_token!r}"
        )
    return token


def make_output_dir(
    output_root: Path,
    *,
    task_name: str,
    method_name: str,
    run_token: str | None = None,
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    token = validate_run_token(run_token)
    if token is None:
        dirname = f"{timestamp}_{task_name}_{method_name}"
    else:
        dirname = f"{timestamp}_{token}_{task_name}_{method_name}"
    return output_root / dirname


def _elapsed(start: float) -> float:
    return round(perf_counter() - start, 3)


if __name__ == "__main__":
    main()
