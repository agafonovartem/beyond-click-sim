from __future__ import annotations

"""Snapshot the exact requests rendered by the pre-refactor Ollama smoke10."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

from runners.in_distribution.regression_prediction.methods import llm_regressor
from runners.in_distribution.regression_prediction.methods.common import (
    current_git_commit,
    json_safe,
    task_xy,
)
from runners.in_distribution.regression_prediction.task_builders import (
    TASK_BUILDERS,
    repo_root,
)


TASK_NAME = "ml-1m_rating_item_stats_eval_users1000_rows_per_user5_seed0"
SMOKE_OUTPUT_NAME = "20260708T_ollama_smoke10_ml1m_ollama_llama31_8b_smoke10"
DEFAULT_OUTPUT = (
    repo_root()
    / "outputs"
    / "in_distribution"
    / "regression_prediction"
    / SMOKE_OUTPUT_NAME
    / "prompt_snapshot.jsonl"
)

SNAPSHOT_CASES = (
    (
        "history_no_summary",
        "llm_regressor_ollama_llama31_8b_smoke10",
        llm_regressor.run_llama31_8b_smoke10,
    ),
    (
        "history_item_stats_no_summary",
        "llm_regressor_ollama_llama31_8b_with_item_stats_smoke10",
        llm_regressor.run_llama31_8b_with_item_stats_smoke10,
    ),
    (
        "history_item_stats_both_summaries",
        "llm_regressor_ollama_llama31_8b_with_item_stats_summary_smoke10",
        llm_regressor.run_llama31_8b_with_item_stats_summary_smoke10,
    ),
)


class RecordingCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(json_safe(kwargs))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="3"))]
        )


class RecordingClient:
    def __init__(self) -> None:
        completions = RecordingCompletions()
        self.chat = SimpleNamespace(completions=completions)
        self.completions = completions


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    records, method_names = build_snapshot_records()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )

    manifest_path = output_path.with_name("prompt_snapshot_manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "source_git_commit": current_git_commit(repo_root()),
                "task": TASK_NAME,
                "cases": method_names,
                "rows_per_case": llm_regressor.SMOKE10_ROWS,
                "requests": len(records),
                "snapshot_file": output_path.name,
                "snapshot_sha256": _sha256_file(output_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"snapshot={output_path}")
    print(f"manifest={manifest_path}")
    print(f"sha256={_sha256_file(output_path)}")


def build_snapshot_records() -> tuple[list[dict[str, Any]], dict[str, str]]:
    task = TASK_BUILDERS[TASK_NAME]()
    X_test, y_test = task_xy(task)["test"]
    X_test = X_test.head(llm_regressor.SMOKE10_ROWS).copy()
    y_test = y_test.loc[X_test.index].copy()
    row_metadata = [
        {
            "position": position,
            "row_index": str(index),
            "user_id": json_safe(row["user_id"]),
            "item_id": json_safe(row["item_id"]),
            "target": json_safe(y_test.loc[index]),
        }
        for position, (index, row) in enumerate(X_test.iterrows(), start=1)
    ]

    records: list[dict[str, Any]] = []
    method_names: dict[str, str] = {}
    original_make_client = llm_regressor.make_llm_client
    try:
        for case_name, method_name, runner in SNAPSHOT_CASES:
            client = RecordingClient()
            llm_regressor.make_llm_client = lambda _client_name: client
            with TemporaryDirectory(prefix=f"{case_name}-") as temp_dir:
                runner(task, Path(temp_dir))

            calls = client.completions.calls
            if len(calls) != len(row_metadata):
                raise RuntimeError(
                    f"Expected {len(row_metadata)} requests for {case_name}, "
                    f"recorded {len(calls)}"
                )
            method_names[case_name] = method_name
            for metadata, request in zip(row_metadata, calls, strict=True):
                records.append(
                    {
                        "case": case_name,
                        **metadata,
                        "request_sha256": _request_sha256(request),
                        "request": request,
                    }
                )
    finally:
        llm_regressor.make_llm_client = original_make_client

    return records, method_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path for the ignored row-level JSONL snapshot.",
    )
    return parser.parse_args()


def _request_sha256(request: dict[str, Any]) -> str:
    payload = json.dumps(
        request,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
