from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from beyond_click_sim.llm_clients import make_llm_client
from beyond_click_sim.scorers import Agent4RecProfileGenerator, Agent4RecRegressor
from beyond_click_sim.scorers.agent4rec.prompts import AGENT4REC_TASTE_PROMPT_VERSION
from beyond_click_sim.tasks import PREFIXED_ITEM_RATING_MEAN_COLUMN, Task
from runners.in_distribution.regression_prediction.config import (
    DATASET_TARGET_REGRESSION_CONFIG,
    MAX_HISTORY_ITEMS,
)
from runners.in_distribution.regression_prediction.item_summaries import (
    ITEM_SUMMARY_COLUMN,
    ITEM_SUMMARY_COLUMN_LABEL,
    add_ml1m_item_summaries,
    resolve_item_summary_visibility,
)
from runners.in_distribution.regression_prediction.methods.common import (
    current_git_commit,
    regression_metrics_for_split,
    score_frame,
    task_xy,
    write_json,
)
from runners.in_distribution.regression_prediction.methods.llm_regressor import (
    _score_rows,
    _write_errors,
)
from runners.in_distribution.regression_prediction.metrics import (
    REGRESSION_MAIN_METRIC,
    REGRESSION_METRICS_FILENAME,
)
from runners.in_distribution.regression_prediction.task_builders import repo_root


VLLM_QWEN36_27B_METHOD_NAME = "agent4rec_regressor_vllm_qwen36_27b"
VLLM_QWEN36_27B_CLIENT = "vllm_local"
VLLM_QWEN36_27B_MODEL = "Qwen/Qwen3.6-27B"
VLLM_QWEN3_8B_METHOD_NAME = "agent4rec_regressor_vllm_qwen3_8b"
VLLM_QWEN3_8B_CLIENT = "vllm_local"
VLLM_QWEN3_8B_MODEL = "Qwen/Qwen3-8B"
QWEN_EXTRA_BODY: dict = {"chat_template_kwargs": {"enable_thinking": False}}
OPENAI_CLIENT = "openai"
GPT4O_MINI_TASTE_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_TOKENS = 64
MAX_LLM_ATTEMPTS = 5
SMOKE_ROWS = 25
TASTE_TEMPERATURE = 0.0
TASTE_MAX_TOKENS = None
TASTE_MAX_WORKERS = 32
VLLM_MAX_WORKERS = 32
QWEN3_8B_MAX_WORKERS = 128

DATASET_CANDIDATE_COLUMNS = {
    "ml-1m": ("item_title", PREFIXED_ITEM_RATING_MEAN_COLUMN, "item_genres"),
}
DATASET_COLUMN_LABELS = {
    "ml-1m": {
        "item_title": "movie title",
        PREFIXED_ITEM_RATING_MEAN_COLUMN: "History ratings",
        "item_genres": "genres",
    },
}


def run_qwen36_27b_traits_taste_gpt4o_mini_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=(
            f"{VLLM_QWEN36_27B_METHOD_NAME}_traits_taste_gpt4o_mini_smoke"
        ),
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_rows=SMOKE_ROWS,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("traits", "taste"),
        taste_client_name=OPENAI_CLIENT,
        taste_model=GPT4O_MINI_TASTE_MODEL,
        taste_temperature=TASTE_TEMPERATURE,
        taste_max_tokens=TASTE_MAX_TOKENS,
    )


def run_qwen36_27b_traits_taste_gpt4o_mini_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=(
            f"{VLLM_QWEN36_27B_METHOD_NAME}_traits_taste_gpt4o_mini_full"
        ),
        client_name=VLLM_QWEN36_27B_CLIENT,
        model=VLLM_QWEN36_27B_MODEL,
        max_rows=None,
        max_workers=VLLM_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("traits", "taste"),
        taste_client_name=OPENAI_CLIENT,
        taste_model=GPT4O_MINI_TASTE_MODEL,
        taste_temperature=TASTE_TEMPERATURE,
        taste_max_tokens=TASTE_MAX_TOKENS,
    )


def run_qwen3_8b_traits_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN3_8B_METHOD_NAME}_traits_smoke",
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        max_rows=SMOKE_ROWS,
        max_workers=QWEN3_8B_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("traits",),
    )


def run_qwen3_8b_traits_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN3_8B_METHOD_NAME}_traits_full",
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        max_rows=None,
        max_workers=QWEN3_8B_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("traits",),
    )


def run_qwen3_8b_traits_summary_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN3_8B_METHOD_NAME}_traits_summary_full",
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        max_rows=None,
        max_workers=QWEN3_8B_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("traits",),
        use_item_summaries=True,
    )


def run_qwen3_8b_taste_gpt4o_mini_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN3_8B_METHOD_NAME}_taste_gpt4o_mini_smoke",
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        max_rows=SMOKE_ROWS,
        max_workers=QWEN3_8B_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("taste",),
        taste_client_name=OPENAI_CLIENT,
        taste_model=GPT4O_MINI_TASTE_MODEL,
        taste_temperature=TASTE_TEMPERATURE,
        taste_max_tokens=TASTE_MAX_TOKENS,
    )


def run_qwen3_8b_taste_gpt4o_mini_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN3_8B_METHOD_NAME}_taste_gpt4o_mini_full",
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        max_rows=None,
        max_workers=QWEN3_8B_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("taste",),
        taste_client_name=OPENAI_CLIENT,
        taste_model=GPT4O_MINI_TASTE_MODEL,
        taste_temperature=TASTE_TEMPERATURE,
        taste_max_tokens=TASTE_MAX_TOKENS,
    )


def run_qwen3_8b_taste_gpt4o_mini_summary_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN3_8B_METHOD_NAME}_taste_gpt4o_mini_summary_full",
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        max_rows=None,
        max_workers=QWEN3_8B_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("taste",),
        taste_client_name=OPENAI_CLIENT,
        taste_model=GPT4O_MINI_TASTE_MODEL,
        taste_temperature=TASTE_TEMPERATURE,
        taste_max_tokens=TASTE_MAX_TOKENS,
        use_item_summaries=True,
    )


def run_qwen3_8b_taste_gpt4o_mini_history_summary_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN3_8B_METHOD_NAME}_taste_gpt4o_mini_history_summary_full",
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        max_rows=None,
        max_workers=QWEN3_8B_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("taste",),
        taste_client_name=OPENAI_CLIENT,
        taste_model=GPT4O_MINI_TASTE_MODEL,
        taste_temperature=TASTE_TEMPERATURE,
        taste_max_tokens=TASTE_MAX_TOKENS,
        history_item_summaries=True,
        candidate_item_summaries=False,
    )


def run_qwen3_8b_taste_gpt4o_mini_candidate_summary_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN3_8B_METHOD_NAME}_taste_gpt4o_mini_candidate_summary_full",
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        max_rows=None,
        max_workers=QWEN3_8B_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("taste",),
        taste_client_name=OPENAI_CLIENT,
        taste_model=GPT4O_MINI_TASTE_MODEL,
        taste_temperature=TASTE_TEMPERATURE,
        taste_max_tokens=TASTE_MAX_TOKENS,
        history_item_summaries=False,
        candidate_item_summaries=True,
    )


def run_qwen3_8b_traits_taste_gpt4o_mini_smoke(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN3_8B_METHOD_NAME}_traits_taste_gpt4o_mini_smoke",
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        max_rows=SMOKE_ROWS,
        max_workers=QWEN3_8B_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("traits", "taste"),
        taste_client_name=OPENAI_CLIENT,
        taste_model=GPT4O_MINI_TASTE_MODEL,
        taste_temperature=TASTE_TEMPERATURE,
        taste_max_tokens=TASTE_MAX_TOKENS,
    )


def run_qwen3_8b_traits_taste_gpt4o_mini_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_QWEN3_8B_METHOD_NAME}_traits_taste_gpt4o_mini_full",
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        max_rows=None,
        max_workers=QWEN3_8B_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("traits", "taste"),
        taste_client_name=OPENAI_CLIENT,
        taste_model=GPT4O_MINI_TASTE_MODEL,
        taste_temperature=TASTE_TEMPERATURE,
        taste_max_tokens=TASTE_MAX_TOKENS,
    )


def run_qwen3_8b_traits_taste_gpt4o_mini_summary_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=(
            f"{VLLM_QWEN3_8B_METHOD_NAME}_traits_taste_gpt4o_mini_summary_full"
        ),
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        max_rows=None,
        max_workers=QWEN3_8B_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("traits", "taste"),
        taste_client_name=OPENAI_CLIENT,
        taste_model=GPT4O_MINI_TASTE_MODEL,
        taste_temperature=TASTE_TEMPERATURE,
        taste_max_tokens=TASTE_MAX_TOKENS,
        use_item_summaries=True,
    )


def run_qwen3_8b_traits_taste_gpt4o_mini_history_summary_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=(
            f"{VLLM_QWEN3_8B_METHOD_NAME}_traits_taste_gpt4o_mini_history_summary_full"
        ),
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        max_rows=None,
        max_workers=QWEN3_8B_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("traits", "taste"),
        taste_client_name=OPENAI_CLIENT,
        taste_model=GPT4O_MINI_TASTE_MODEL,
        taste_temperature=TASTE_TEMPERATURE,
        taste_max_tokens=TASTE_MAX_TOKENS,
        history_item_summaries=True,
        candidate_item_summaries=False,
    )


def run_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_full(
    task: Task,
    output_dir: Path,
) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=(
            f"{VLLM_QWEN3_8B_METHOD_NAME}_traits_taste_gpt4o_mini_candidate_summary_full"
        ),
        client_name=VLLM_QWEN3_8B_CLIENT,
        model=VLLM_QWEN3_8B_MODEL,
        max_rows=None,
        max_workers=QWEN3_8B_MAX_WORKERS,
        extra_body=QWEN_EXTRA_BODY,
        profile_components=("traits", "taste"),
        taste_client_name=OPENAI_CLIENT,
        taste_model=GPT4O_MINI_TASTE_MODEL,
        taste_temperature=TASTE_TEMPERATURE,
        taste_max_tokens=TASTE_MAX_TOKENS,
        history_item_summaries=False,
        candidate_item_summaries=True,
    )


def run_method(
    task: Task,
    output_dir: Path,
    *,
    method_name: str,
    client_name: str,
    model: str,
    max_rows: int | None,
    max_history_items: int | None = MAX_HISTORY_ITEMS,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    max_llm_attempts: int = MAX_LLM_ATTEMPTS,
    max_workers: int = 1,
    extra_body: dict | None = None,
    profile_components: tuple[str, ...] = ("traits",),
    taste_client_name: str | None = None,
    taste_model: str | None = None,
    taste_temperature: float = TASTE_TEMPERATURE,
    taste_max_tokens: int | None = TASTE_MAX_TOKENS,
    taste_max_attempts: int = MAX_LLM_ATTEMPTS,
    taste_max_workers: int = TASTE_MAX_WORKERS,
    taste_prompt_version: str = AGENT4REC_TASTE_PROMPT_VERSION,
    taste_cache_path: Path | None = None,
    use_item_summaries: bool = False,
    history_item_summaries: bool | None = None,
    candidate_item_summaries: bool | None = None,
) -> dict[str, object]:
    """Run the Agent4Rec profile-based discrete rating regressor."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if task.schema.candidate_group_column is not None:
        raise ValueError("Agent4Rec regression method requires observed-only task rows")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive when provided")

    dataset_name = str(task.manifest["dataset"])
    if dataset_name not in DATASET_CANDIDATE_COLUMNS:
        raise ValueError(
            "Agent4Rec regression v1 supports only MovieLens-style rating tasks. "
            f"Got dataset: {dataset_name!r}"
        )
    target_source_column = str(task.manifest["target_source_column"])
    target_config = DATASET_TARGET_REGRESSION_CONFIG[dataset_name][target_source_column]
    if target_config["target_name"] != "rating":
        raise ValueError("Agent4Rec regression v1 supports only rating targets")
    uses_taste = "taste" in profile_components
    summary_visibility = resolve_item_summary_visibility(
        use_item_summaries=use_item_summaries,
        history_item_summaries=history_item_summaries,
        candidate_item_summaries=candidate_item_summaries,
    )
    if not uses_taste:
        summary_visibility["history"] = False
        summary_visibility["any"] = summary_visibility["candidate"]

    xy = task_xy(task)
    X_train, y_train = xy["train"]
    X_test, y_test = xy["test"]
    if max_rows is not None:
        X_test = X_test.head(max_rows).copy()
        y_test = y_test.loc[X_test.index].copy()
    X_train, X_test, item_summary_metadata = add_ml1m_item_summaries(
        dataset_name=dataset_name,
        X_train=X_train,
        X_test=X_test,
        use_item_summaries=summary_visibility["any"],
        summary_visibility=summary_visibility,
    )
    candidate_columns = _candidate_columns(
        dataset_name,
        use_item_summaries=summary_visibility["candidate"],
    )
    column_labels = _column_labels(
        dataset_name,
        use_item_summaries=summary_visibility["candidate"],
    )
    profile_user_ids = X_test["user_id"].drop_duplicates().tolist()
    _require_columns(X_test, list(candidate_columns))

    if uses_taste:
        if not taste_client_name:
            raise ValueError("taste_client_name is required for taste profiles")
        if not taste_model:
            raise ValueError("taste_model is required for taste profiles")
        if taste_cache_path is None:
            taste_cache_path = agent4rec_taste_cache_path(
                task,
                taste_model=taste_model,
                taste_prompt_version=taste_prompt_version,
                use_history_item_summaries=summary_visibility["history"],
            )
        taste_client = make_llm_client(taste_client_name)
    else:
        taste_client = None

    profile_generator = Agent4RecProfileGenerator(
        profile_components=profile_components,
        taste_client=taste_client,
        taste_client_name=taste_client_name,
        taste_model=taste_model,
        taste_cache_path=taste_cache_path,
        taste_prompt_version=taste_prompt_version,
        taste_temperature=taste_temperature,
        taste_max_tokens=taste_max_tokens,
        taste_max_attempts=taste_max_attempts,
        taste_max_workers=taste_max_workers,
        summary_column=(
            ITEM_SUMMARY_COLUMN
            if summary_visibility["history"]
            else None
        ),
    )
    scorer = Agent4RecRegressor(
        client=make_llm_client(client_name),
        model=model,
        target_description=str(target_config["target_description"]),
        valid_values=target_config["valid_values"],
        profile_generator=profile_generator,
        candidate_description_columns=candidate_columns,
        column_labels=column_labels,
        max_history_items=max_history_items,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    ).fit(X_train, y_train, profile_user_ids=profile_user_ids)
    if uses_taste:
        scorer.build_taste(X_test)

    scores, errors = _score_rows(
        scorer,
        X_test,
        max_attempts=max_llm_attempts,
        max_workers=max_workers,
    )
    valid = scores.notna()
    score_frame(
        split="test",
        X=X_test,
        y=y_test,
        scores=scores,
    ).to_parquet(output_dir / "predictions.parquet", index=False)
    _write_errors(output_dir / "llm_errors.jsonl", errors)

    if not valid.any():
        raise RuntimeError("Agent4Rec regressor did not produce any valid scores")

    valid_scores = scores.loc[valid]
    valid_X = X_test.loc[valid].copy()
    valid_y = y_test.loc[valid].copy()
    test_metrics = regression_metrics_for_split(
        X=valid_X,
        y=valid_y,
        scores=valid_scores,
    )
    requested_rows = int(len(X_test))
    scored_rows = int(valid.sum())
    coverage = scored_rows / requested_rows if requested_rows else 0.0

    root = repo_root()
    scorer_manifest: dict[str, Any] = {
        "class": "Agent4RecRegressor",
        "client_name": client_name,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_history_items": max_history_items,
        "candidate_description_columns": list(
            candidate_columns
        ),
        "column_labels": column_labels,
        "profile_generator": scorer.profile_generator.manifest(),
        "extra_body": extra_body,
        "uses_item_stats": True,
        "item_summaries": item_summary_metadata,
        "target": {
            "name": target_config["target_name"],
            "description": target_config["target_description"],
            "valid_values": target_config["valid_values"],
            "output_format": "agent4rec_rating_line",
        },
    }
    limits = {
        "max_rows": max_rows,
        "max_llm_attempts": max_llm_attempts,
        "max_workers": max_workers,
        "taste_max_workers": taste_max_workers if uses_taste else None,
    }
    manifest = {
        "method": method_name,
        "protocol": "regression",
        "scorer": scorer_manifest,
        "decision_rule": {
            "kind": "hard_discrete_rating_parser",
            "parser_contract": "agent4rec_rating_line",
            "valid_values": list(target_config["valid_values"]),
        },
        "regression_evaluation": {
            "main_metric": REGRESSION_MAIN_METRIC,
            "aggregations": ["micro", "macro_by_user_mean"],
            "metrics": ["mae", "rmse"],
        },
        "limits": limits,
        "evaluated_splits": ["test"],
        "llm_errors": len(errors),
        "requested_rows": requested_rows,
        "scored_rows": scored_rows,
        "coverage": coverage,
        "task": {
            "name": task.name,
            "manifest": task.manifest,
        },
        "git_commit": current_git_commit(root),
    }
    metrics = {
        "method": method_name,
        "task": task.name,
        "protocol": "regression",
        "main_metric": REGRESSION_MAIN_METRIC,
        "regression_evaluation": {
            "aggregations": ["micro", "macro_by_user_mean"],
            "metrics": ["mae", "rmse"],
        },
        "evaluated_splits": ["test"],
        "test": test_metrics,
        "llm_errors": len(errors),
        "requested_rows": requested_rows,
        "scored_rows": scored_rows,
        "coverage": coverage,
        "max_rows": max_rows,
        "max_workers": max_workers,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / REGRESSION_METRICS_FILENAME, metrics)
    return metrics


def agent4rec_taste_cache_path(
    task: Task,
    *,
    taste_model: str,
    taste_prompt_version: str,
    use_history_item_summaries: bool = False,
) -> Path:
    dataset_name = str(task.manifest["dataset"])
    dataset_version = str(task.manifest["dataset_version"])
    split_seed = task.manifest["splitter"]["seed"]
    model_slug = _cache_slug(taste_model)
    summary_slug = "_summary" if use_history_item_summaries else ""
    return (
        repo_root()
        / "outputs"
        / "agent4rec_taste_cache"
        / (
            f"{dataset_name}_{dataset_version}_seed{split_seed}_"
            f"{model_slug}_{taste_prompt_version}{summary_slug}.jsonl"
        )
    )


def _candidate_columns(
    dataset_name: str,
    *,
    use_item_summaries: bool,
) -> tuple[str, ...]:
    columns = DATASET_CANDIDATE_COLUMNS[dataset_name]
    if not use_item_summaries:
        return columns
    return (*columns, ITEM_SUMMARY_COLUMN)


def _column_labels(
    dataset_name: str,
    *,
    use_item_summaries: bool,
) -> dict[str, str]:
    labels = dict(DATASET_COLUMN_LABELS[dataset_name])
    if use_item_summaries:
        labels[ITEM_SUMMARY_COLUMN] = ITEM_SUMMARY_COLUMN_LABEL
    return labels


def _require_columns(frame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            "Agent4Rec regression requires item-stats task columns for the "
            f"candidate prompt. Missing columns: {missing}"
        )


def _cache_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    if not slug:
        raise ValueError(f"Cannot build cache slug from value: {value!r}")
    return slug
