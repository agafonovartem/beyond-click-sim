from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import pandas as pd
from tqdm import tqdm

from beyond_click_sim.evaluation.policy_ranking import policy_ranking_agreement_metrics
from beyond_click_sim.llm_clients import make_llm_client
from beyond_click_sim.scorers import LLMInteractionYesNoScorer
from beyond_click_sim.tasks import Task

from runners.in_distribution.llm_item_stats import (
    item_rating_column_labels,
    maybe_add_item_rating_prompt_columns,
)
from runners.in_distribution.policy_ranking_agreement.metrics import (
    MAIN_METRIC,
    METRICS_FILENAME,
    UTILITY_AGGREGATION,
)
from runners.in_distribution.policy_ranking_agreement.methods.common import (
    compute_policy_utilities,
    current_git_commit,
    json_safe,
    task_xy,
    write_json,
)
from runners.in_distribution.policy_ranking_agreement.task_builders import repo_root

# ---------------------------------------------------------------------------
# Model / client constants
# ---------------------------------------------------------------------------

OLLAMA_CLIENT = "ollama_local"
OLLAMA_MODEL_LLAMA31_8B = "llama3.1:8b"
OLLAMA_MODEL_LLAMA32 = "llama3.2"
VLLM_CLIENT = "vllm_local"
VLLM_MODEL_LLAMA33_70B = "llama-3.3-70b-instruct"
VLLM_MODEL_QWEN3_8B = "qwen3-8b"
VLLM_MODEL_QWEN36_27B = "Qwen/Qwen3.6-27B"
VLLM_MODEL_QWEN36_35B_A3B = "qwen3.6-35b-a3b"

QWEN_EXTRA_BODY: dict = {"chat_template_kwargs": {"enable_thinking": False}}

MAX_HISTORY_ITEMS = 20
TEMPERATURE = 0.0
MAX_TOKENS = 256
MAX_LLM_ATTEMPTS = 5
OLLAMA_MAX_WORKERS = 1
VLLM_MAX_WORKERS = 32

# ---------------------------------------------------------------------------
# Prompt columns per dataset — same as Q1/Q2
# ---------------------------------------------------------------------------

DATASET_PROMPT_COLUMNS: dict[str, dict[str, tuple[str, ...]]] = {
    "ml-1m": {
        "history_description_columns": ("item_title", "item_genres", "rating"),
        "candidate_description_columns": ("item_title", "item_genres"),
    },
    "steam": {
        "history_description_columns": (
            "item_title",
            "item_genres_json",
            "item_tags_json",
            "playtime_forever",
        ),
        "candidate_description_columns": (
            "item_title",
            "item_genres_json",
            "item_tags_json",
        ),
    },
}

# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_llama31_8b_smoke(task: Task, output_dir: Path) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_ollama_llama31_8b_smoke",
                client_name=OLLAMA_CLIENT,
                model=OLLAMA_MODEL_LLAMA31_8B,
                max_llm_groups=25,
                max_workers=OLLAMA_MAX_WORKERS)


def run_llama31_8b_full(task: Task, output_dir: Path) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_ollama_llama31_8b_full",
                client_name=OLLAMA_CLIENT,
                model=OLLAMA_MODEL_LLAMA31_8B,
                max_llm_groups=None,
                max_workers=OLLAMA_MAX_WORKERS)


def run_llama32_smoke(task: Task, output_dir: Path) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_ollama_llama32_smoke",
                client_name=OLLAMA_CLIENT,
                model=OLLAMA_MODEL_LLAMA32,
                max_llm_groups=25,
                max_workers=OLLAMA_MAX_WORKERS)


def run_llama32_full(task: Task, output_dir: Path) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_ollama_llama32_full",
                client_name=OLLAMA_CLIENT,
                model=OLLAMA_MODEL_LLAMA32,
                max_llm_groups=None,
                max_workers=OLLAMA_MAX_WORKERS)


def run_llama33_70b_smoke(task: Task, output_dir: Path) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_vllm_llama33_70b_smoke",
                client_name=VLLM_CLIENT,
                model=VLLM_MODEL_LLAMA33_70B,
                max_llm_groups=25,
                max_workers=VLLM_MAX_WORKERS)


def run_llama33_70b_full(task: Task, output_dir: Path) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_vllm_llama33_70b_full",
                client_name=VLLM_CLIENT,
                model=VLLM_MODEL_LLAMA33_70B,
                max_llm_groups=None,
                max_workers=VLLM_MAX_WORKERS)


def run_qwen3_8b_smoke(task: Task, output_dir: Path) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_vllm_qwen3_8b_smoke",
                client_name=VLLM_CLIENT,
                model=VLLM_MODEL_QWEN3_8B,
                max_llm_groups=25,
                max_workers=VLLM_MAX_WORKERS,
                extra_body=QWEN_EXTRA_BODY)


def run_qwen3_8b_full(task: Task, output_dir: Path) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_vllm_qwen3_8b_full",
                client_name=VLLM_CLIENT,
                model=VLLM_MODEL_QWEN3_8B,
                max_llm_groups=None,
                max_workers=VLLM_MAX_WORKERS,
                extra_body=QWEN_EXTRA_BODY)


def run_qwen36_27b_smoke(task: Task, output_dir: Path) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_vllm_qwen36_27b_smoke",
                client_name=VLLM_CLIENT,
                model=VLLM_MODEL_QWEN36_27B,
                max_llm_groups=25,
                max_workers=VLLM_MAX_WORKERS,
                extra_body=QWEN_EXTRA_BODY)


def run_qwen36_27b_full(task: Task, output_dir: Path) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_vllm_qwen36_27b_full",
                client_name=VLLM_CLIENT,
                model=VLLM_MODEL_QWEN36_27B,
                max_llm_groups=None,
                max_workers=VLLM_MAX_WORKERS,
                extra_body=QWEN_EXTRA_BODY)


def run_qwen36_35b_a3b_smoke(task: Task, output_dir: Path) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_vllm_qwen36_35b_a3b_smoke",
                client_name=VLLM_CLIENT,
                model=VLLM_MODEL_QWEN36_35B_A3B,
                max_llm_groups=25,
                max_workers=VLLM_MAX_WORKERS,
                extra_body=QWEN_EXTRA_BODY)


def run_qwen36_35b_a3b_full(task: Task, output_dir: Path) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_vllm_qwen36_35b_a3b_full",
                client_name=VLLM_CLIENT,
                model=VLLM_MODEL_QWEN36_35B_A3B,
                max_llm_groups=None,
                max_workers=VLLM_MAX_WORKERS,
                extra_body=QWEN_EXTRA_BODY)

# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------

def _run(
    task: Task,
    output_dir: Path,
    *,
    method_name: str,
    client_name: str,
    model: str,
    max_llm_groups: int | None,
    max_workers: int,
    use_item_stats: bool = False,
    extra_body: dict | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_times: dict[str, float] = {}
    dataset_name = str(task.manifest["dataset"])

    prompt_columns = maybe_add_item_rating_prompt_columns(
        dataset_name,
        DATASET_PROMPT_COLUMNS[dataset_name],
        use_item_stats=use_item_stats,
    )
    column_labels = item_rating_column_labels(dataset_name, use_item_stats=use_item_stats)

    t = perf_counter()
    xy = task_xy(task)
    X_train, y_train = xy["train"]
    X_test, y_test = xy["test"]
    _record(stage_times, "prepare_xy", t)

    # Each (user, policy) pair is one LLM candidate group: all k recommended
    # items for that user under that policy are scored in a single prompt.
    X_test = X_test.copy()
    X_test["_llm_group_"] = (
        X_test["user_id"].astype(str) + "::" + X_test["policy"].astype(str)
    )

    # Optionally cap the number of groups (smoke runs).
    # Sample proportionally per policy so every policy stays represented.
    if max_llm_groups is not None:
        policy_to_groups = (
            X_test.groupby("policy")["_llm_group_"].unique().to_dict()
        )
        n_policies = len(policy_to_groups)
        per_policy = max(1, max_llm_groups // n_policies)
        kept_groups: set[str] = set()
        for groups in policy_to_groups.values():
            kept_groups.update(groups[:per_policy])
        X_test = X_test[X_test["_llm_group_"].isin(kept_groups)].copy()
        y_test = y_test.loc[X_test.index]

    t = perf_counter()
    scorer = LLMInteractionYesNoScorer(
        client=make_llm_client(client_name),
        model=model,
        history_description_columns=prompt_columns["history_description_columns"],
        candidate_description_columns=prompt_columns["candidate_description_columns"],
        candidate_group_column="_llm_group_",
        column_labels=column_labels,
        max_history_items=MAX_HISTORY_ITEMS,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        extra_body=extra_body,
    ).fit(X_train, y_train)
    _record(stage_times, "fit", t)

    t = perf_counter()
    scores, errors = _score_groups(
        scorer,
        X_test,
        candidate_group_column="_llm_group_",
        max_attempts=MAX_LLM_ATTEMPTS,
        max_workers=max_workers,
    )
    _record(stage_times, "score_test", t)

    # Use only rows with valid (non-NaN) scores for utility aggregation.
    valid = scores.notna()
    X_scored = X_test.loc[valid].drop(columns=["_llm_group_"])
    y_scored = y_test.loc[valid]
    scores_scored = scores.loc[valid]

    t = perf_counter()
    simulated_utilities, real_utilities = compute_policy_utilities(
        X_scored, y_scored, scores_scored, policy_column="policy",
    )
    if not simulated_utilities:
        raise RuntimeError(
            f"All {len(errors)} LLM group(s) failed — no valid scores. "
            "Check that the server is reachable and the model name is correct. "
            f"First error: {errors[0] if errors else 'none'}"
        )
    policy_names = sorted(simulated_utilities)
    agreement = policy_ranking_agreement_metrics(
        policy_names,
        [simulated_utilities[p] for p in policy_names],
        [real_utilities[p] for p in policy_names],
    )
    _record(stage_times, "aggregate_and_rank", t)

    t = perf_counter()
    predictions = X_test.drop(columns=["_llm_group_"]).copy()
    predictions.insert(0, "split", "test")
    predictions["target"] = y_test.to_numpy()
    predictions["score"] = scores.to_numpy()
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    _write_errors(output_dir / "llm_errors.jsonl", errors)
    _record(stage_times, "write_predictions", t)

    t = perf_counter()
    # X_test still carries _llm_group_ here (we only dropped it from X_scored and predictions).
    requested_groups = int(X_test["_llm_group_"].nunique())
    scored_groups = int(
        (X_scored["user_id"].astype(str) + "::" + X_scored["policy"].astype(str)).nunique()
    )

    metrics = {
        "method": method_name,
        "task": task.name,
        "protocol": "policy_ranking",
        "main_metric": MAIN_METRIC,
        "utility_aggregation": UTILITY_AGGREGATION,
        "test": agreement,
        "llm_errors": len(errors),
        "scored_rows": int(valid.sum()),
        "requested_rows": int(len(scores)),
        "scored_groups": scored_groups,
        "requested_groups": requested_groups,
        "max_llm_groups": max_llm_groups,
        "stage_times_seconds": stage_times,
    }
    manifest = {
        "method": method_name,
        "protocol": "policy_ranking",
        "scorer": {
            "class": "LLMInteractionYesNoScorer",
            "client_name": client_name,
            "model": model,
            "candidate_group": "user_id::policy",
            "max_history_items": MAX_HISTORY_ITEMS,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "prompt_columns": {k: list(v) for k, v in prompt_columns.items()},
            "column_labels": column_labels,
            "uses_item_stats": use_item_stats,
            "extra_body": extra_body,
        },
        "utility_aggregation": UTILITY_AGGREGATION,
        "limits": {
            "max_llm_groups": max_llm_groups,
            "max_llm_attempts": MAX_LLM_ATTEMPTS,
            "max_workers": max_workers,
        },
        "llm_errors": len(errors),
        "groups": {"requested": requested_groups, "scored": scored_groups},
        "task": {"name": task.name, "manifest": json_safe(task.manifest)},
        "stage_times_seconds": stage_times,
        "git_commit": current_git_commit(repo_root()),
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / METRICS_FILENAME, metrics)
    _record(stage_times, "write_metadata", t)

    return metrics


# ---------------------------------------------------------------------------
# LLM group scoring with retry and optional threading (mirrors Q1/Q2)
# ---------------------------------------------------------------------------

def _score_groups(
    scorer: LLMInteractionYesNoScorer,
    X: pd.DataFrame,
    *,
    candidate_group_column: str,
    max_attempts: int,
    max_workers: int = 1,
) -> tuple[pd.Series, list[dict[str, object]]]:
    scores = pd.Series(index=X.index, dtype=float, name="score")
    errors: list[dict[str, object]] = []
    groups = list(X.groupby(candidate_group_column, sort=False))

    if max_workers == 1:
        progress = tqdm(groups, desc="llm groups", unit="group")
        for group_id, group in progress:
            group_scores, error = _score_one_group(scorer, group_id, group, max_attempts=max_attempts)
            if error is not None:
                errors.append(error)
                progress.set_postfix(errors=len(errors))
            else:
                scores.loc[group_scores.index] = group_scores
        return scores, errors

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_score_one_group, scorer, group_id, group, max_attempts=max_attempts)
            for group_id, group in groups
        ]
        progress = tqdm(as_completed(futures), total=len(futures), desc="llm groups", unit="group")
        for future in progress:
            group_scores, error = future.result()
            if error is not None:
                errors.append(error)
                progress.set_postfix(errors=len(errors))
            else:
                scores.loc[group_scores.index] = group_scores
    return scores, errors


def _score_one_group(
    scorer: LLMInteractionYesNoScorer,
    group_id: object,
    group: pd.DataFrame,
    *,
    max_attempts: int,
) -> tuple[pd.Series, dict[str, object] | None]:
    attempt_errors: list[str] = []
    for _ in range(max_attempts):
        try:
            return scorer.score(group), None
        except Exception as exc:  # noqa: BLE001 — keep long runs alive
            attempt_errors.append(repr(exc))
    empty = pd.Series(index=group.index, dtype=float, name="score")
    return empty, {"candidate_group": str(group_id), "attempts": max_attempts, "errors": attempt_errors}


def _write_errors(path: Path, errors: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for error in errors:
            f.write(json.dumps(error, sort_keys=True) + "\n")


def _record(stage_times: dict[str, float], stage: str, t0: float) -> None:
    from time import perf_counter
    stage_times[stage] = perf_counter() - t0
