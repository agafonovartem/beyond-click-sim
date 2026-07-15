from __future__ import annotations

import json
from collections.abc import Callable
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
    write_policy_metrics,
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
DATASET_JSON_LIST_COLUMNS = {
    "ml-1m": (),
    "steam": ("item_genres_json", "item_tags_json"),
}

# ---------------------------------------------------------------------------
# Tie-break defaults
# ---------------------------------------------------------------------------

def majority_vote(scores: list[float]) -> float:
    """Default tie-break: yes iff strictly more runs returned 1.0 than 0.0.

    With an even number of runs and an equal split, returns 0.0 (no).
    """
    yes = sum(s > 0.5 for s in scores)
    return 1.0 if yes > len(scores) / 2 else 0.0

# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_llama31_8b_smoke(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_ollama_llama31_8b_smoke",
                client_name=OLLAMA_CLIENT, model=OLLAMA_MODEL_LLAMA31_8B,
                max_llm_groups=25, max_workers=OLLAMA_MAX_WORKERS,
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)

def run_llama31_8b_full(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_ollama_llama31_8b_full",
                client_name=OLLAMA_CLIENT, model=OLLAMA_MODEL_LLAMA31_8B,
                max_llm_groups=None, max_workers=OLLAMA_MAX_WORKERS,
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)

def run_llama31_8b_itemwise_smoke(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_itemwise_ollama_llama31_8b_smoke",
                client_name=OLLAMA_CLIENT, model=OLLAMA_MODEL_LLAMA31_8B,
                max_llm_groups=25, max_workers=OLLAMA_MAX_WORKERS, scoring="itemwise",
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)

def run_llama31_8b_itemwise_full(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_itemwise_ollama_llama31_8b_full",
                client_name=OLLAMA_CLIENT, model=OLLAMA_MODEL_LLAMA31_8B,
                max_llm_groups=None, max_workers=OLLAMA_MAX_WORKERS, scoring="itemwise",
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)


def run_llama32_smoke(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_ollama_llama32_smoke",
                client_name=OLLAMA_CLIENT, model=OLLAMA_MODEL_LLAMA32,
                max_llm_groups=25, max_workers=OLLAMA_MAX_WORKERS,
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)

def run_llama32_full(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_ollama_llama32_full",
                client_name=OLLAMA_CLIENT, model=OLLAMA_MODEL_LLAMA32,
                max_llm_groups=None, max_workers=OLLAMA_MAX_WORKERS,
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)

def run_llama32_itemwise_smoke(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_itemwise_ollama_llama32_smoke",
                client_name=OLLAMA_CLIENT, model=OLLAMA_MODEL_LLAMA32,
                max_llm_groups=25, max_workers=OLLAMA_MAX_WORKERS, scoring="itemwise",
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)

def run_llama32_itemwise_full(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_itemwise_ollama_llama32_full",
                client_name=OLLAMA_CLIENT, model=OLLAMA_MODEL_LLAMA32,
                max_llm_groups=None, max_workers=OLLAMA_MAX_WORKERS, scoring="itemwise",
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)


def run_llama33_70b_smoke(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_vllm_llama33_70b_smoke",
                client_name=VLLM_CLIENT, model=VLLM_MODEL_LLAMA33_70B,
                max_llm_groups=25, max_workers=VLLM_MAX_WORKERS,
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)

def run_llama33_70b_full(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_vllm_llama33_70b_full",
                client_name=VLLM_CLIENT, model=VLLM_MODEL_LLAMA33_70B,
                max_llm_groups=None, max_workers=VLLM_MAX_WORKERS,
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)

def run_llama33_70b_itemwise_smoke(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_itemwise_vllm_llama33_70b_smoke",
                client_name=VLLM_CLIENT, model=VLLM_MODEL_LLAMA33_70B,
                max_llm_groups=25, max_workers=VLLM_MAX_WORKERS, scoring="itemwise",
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)

def run_llama33_70b_itemwise_full(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_itemwise_vllm_llama33_70b_full",
                client_name=VLLM_CLIENT, model=VLLM_MODEL_LLAMA33_70B,
                max_llm_groups=None, max_workers=VLLM_MAX_WORKERS, scoring="itemwise",
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)


def run_qwen3_8b_smoke(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_vllm_qwen3_8b_smoke",
                client_name=VLLM_CLIENT, model=VLLM_MODEL_QWEN3_8B,
                max_llm_groups=25, max_workers=VLLM_MAX_WORKERS, extra_body=QWEN_EXTRA_BODY,
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)

def run_qwen3_8b_full(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_vllm_qwen3_8b_full",
                client_name=VLLM_CLIENT, model=VLLM_MODEL_QWEN3_8B,
                max_llm_groups=None, max_workers=VLLM_MAX_WORKERS, extra_body=QWEN_EXTRA_BODY,
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)

def run_qwen3_8b_itemwise_smoke(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_itemwise_vllm_qwen3_8b_smoke",
                client_name=VLLM_CLIENT, model=VLLM_MODEL_QWEN3_8B,
                max_llm_groups=25, max_workers=VLLM_MAX_WORKERS,
                extra_body=QWEN_EXTRA_BODY, scoring="itemwise",
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)

def run_qwen3_8b_itemwise_full(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_itemwise_vllm_qwen3_8b_full",
                client_name=VLLM_CLIENT, model=VLLM_MODEL_QWEN3_8B,
                max_llm_groups=None, max_workers=VLLM_MAX_WORKERS,
                extra_body=QWEN_EXTRA_BODY, scoring="itemwise",
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)


def run_qwen36_27b_smoke(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_vllm_qwen36_27b_smoke",
                client_name=VLLM_CLIENT, model=VLLM_MODEL_QWEN36_27B,
                max_llm_groups=25, max_workers=VLLM_MAX_WORKERS, extra_body=QWEN_EXTRA_BODY,
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)

def run_qwen36_27b_full(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_vllm_qwen36_27b_full",
                client_name=VLLM_CLIENT, model=VLLM_MODEL_QWEN36_27B,
                max_llm_groups=None, max_workers=VLLM_MAX_WORKERS, extra_body=QWEN_EXTRA_BODY,
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)

def run_qwen36_27b_itemwise_smoke(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_itemwise_vllm_qwen36_27b_smoke",
                client_name=VLLM_CLIENT, model=VLLM_MODEL_QWEN36_27B,
                max_llm_groups=25, max_workers=VLLM_MAX_WORKERS,
                extra_body=QWEN_EXTRA_BODY, scoring="itemwise",
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)

def run_qwen36_27b_itemwise_full(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_itemwise_vllm_qwen36_27b_full",
                client_name=VLLM_CLIENT, model=VLLM_MODEL_QWEN36_27B,
                max_llm_groups=None, max_workers=VLLM_MAX_WORKERS,
                extra_body=QWEN_EXTRA_BODY, scoring="itemwise",
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)


def run_qwen36_35b_a3b_smoke(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_vllm_qwen36_35b_a3b_smoke",
                client_name=VLLM_CLIENT, model=VLLM_MODEL_QWEN36_35B_A3B,
                max_llm_groups=25, max_workers=VLLM_MAX_WORKERS, extra_body=QWEN_EXTRA_BODY,
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)

def run_qwen36_35b_a3b_full(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_vllm_qwen36_35b_a3b_full",
                client_name=VLLM_CLIENT, model=VLLM_MODEL_QWEN36_35B_A3B,
                max_llm_groups=None, max_workers=VLLM_MAX_WORKERS, extra_body=QWEN_EXTRA_BODY,
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)

def run_qwen36_35b_a3b_itemwise_smoke(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_itemwise_vllm_qwen36_35b_a3b_smoke",
                client_name=VLLM_CLIENT, model=VLLM_MODEL_QWEN36_35B_A3B,
                max_llm_groups=25, max_workers=VLLM_MAX_WORKERS,
                extra_body=QWEN_EXTRA_BODY, scoring="itemwise",
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)

def run_qwen36_35b_a3b_itemwise_full(
    task: Task,
    output_dir: Path,
    *,
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    return _run(task, output_dir,
                method_name="llm_yes_no_itemwise_vllm_qwen36_35b_a3b_full",
                client_name=VLLM_CLIENT, model=VLLM_MODEL_QWEN36_35B_A3B,
                max_llm_groups=None, max_workers=VLLM_MAX_WORKERS,
                extra_body=QWEN_EXTRA_BODY, scoring="itemwise",
                run_count=run_count, tie_break=tie_break, batch_dedup_mode=batch_dedup_mode)

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
    scoring: str = "batch",
    run_count: int = 1,
    tie_break: Callable[[list[float]], float] | None = None,
    batch_dedup_mode: str = "ordered",
) -> dict[str, object]:
    if scoring not in ("batch", "itemwise"):
        raise ValueError(f"scoring must be 'batch' or 'itemwise', got {scoring!r}")
    if run_count < 1:
        raise ValueError(f"run_count must be >= 1, got {run_count}")
    if batch_dedup_mode not in ("ordered", "unordered"):
        raise ValueError(f"batch_dedup_mode must be 'ordered' or 'unordered', got {batch_dedup_mode!r}")
    tie_break_fn = tie_break if tie_break is not None else majority_vote

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

    # Assign the per-policy LLM group key for smoke-cap proportionality tracking.
    # batch:    one group per (user, policy) — all k items sent in a single prompt.
    # itemwise: one group per (user, policy, item) — eliminates positional bias.
    X_test = X_test.copy()
    if scoring == "itemwise":
        X_test["_llm_group_"] = (
            X_test["user_id"].astype(str) + "::"
            + X_test["policy"].astype(str) + "::"
            + X_test["item_id"].astype(str)
        )
    else:
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
        json_list_columns=DATASET_JSON_LIST_COLUMNS[dataset_name],
        max_history_items=MAX_HISTORY_ITEMS,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        extra_body=extra_body,
        prompt_style=scoring,
    ).fit(X_train, y_train)
    _record(stage_times, "fit", t)

    # ------------------------------------------------------------------
    # Deduplication: compute a scoring key that collapses cross-policy
    # redundancy, build a deduplicated scoring set, then join scores back.
    # ------------------------------------------------------------------
    if scoring == "itemwise":
        # One LLM call per unique (user, item) pair — independent of which
        # policy recommended the item. Broadcast the score to all policy rows.
        X_test["_score_key_"] = (
            X_test["user_id"].astype(str) + "::" + X_test["item_id"].astype(str)
        )
        X_to_score = X_test.drop_duplicates(subset=["_score_key_"]).copy()
        X_to_score["_llm_group_"] = X_to_score["_score_key_"]
    else:
        # One LLM call per unique batch. Two (user, policy) groups with the same
        # item set share one call only if their content matches per batch_dedup_mode.
        group_sigs: dict[str, str] = {}
        for gid, grp in X_test.groupby("_llm_group_", sort=False):
            uid = str(grp["user_id"].iloc[0])
            if batch_dedup_mode == "ordered":
                items = tuple(grp.sort_values("rank")["item_id"].tolist())
            else:  # unordered
                items = tuple(sorted(grp["item_id"].astype(str).tolist()))
            group_sigs[str(gid)] = f"{uid}::{items}"
        X_test["_score_key_"] = X_test["_llm_group_"].map(group_sigs)
        seen_sigs: set[str] = set()
        kept_gids: list[str] = []
        for gid, sig in group_sigs.items():
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                kept_gids.append(gid)
        X_to_score = X_test[X_test["_llm_group_"].isin(kept_gids)].copy()
        gid_to_sig = {gid: group_sigs[gid] for gid in kept_gids}
        X_to_score["_llm_group_"] = X_to_score["_llm_group_"].map(gid_to_sig)

    t = perf_counter()
    unique_scores, errors = _score_groups(
        scorer,
        X_to_score,
        candidate_group_column="_llm_group_",
        max_attempts=MAX_LLM_ATTEMPTS,
        max_workers=max_workers,
        run_count=run_count,
        tie_break_fn=tie_break_fn,
    )
    _record(stage_times, "score_test", t)

    # Join scores back to the full X_test (including rows filtered by dedup).
    if scoring == "itemwise":
        score_key_to_score = dict(
            zip(X_to_score["_score_key_"].tolist(), unique_scores.tolist())
        )
        scores = X_test["_score_key_"].map(score_key_to_score).rename("score")
    else:
        # Batch: each item within a unique batch has an independent score.
        # Build (batch_sig, item_id) → score and map back row by row.
        scored_dedup = X_to_score[["item_id"]].copy()
        scored_dedup["_sig_"] = X_to_score["_llm_group_"].values
        scored_dedup["_s_"] = unique_scores.reindex(X_to_score.index).values
        lookup = scored_dedup.set_index(["_sig_", "item_id"])["_s_"].to_dict()
        scores = pd.Series(
            [lookup.get((sk, iid), float("nan"))
             for sk, iid in zip(X_test["_score_key_"], X_test["item_id"])],
            index=X_test.index,
            name="score",
        )

    # Use only rows with valid (non-NaN) scores for utility aggregation.
    valid = scores.notna()
    X_scored = X_test.loc[valid].drop(columns=["_llm_group_", "_score_key_"])
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
    predictions = X_test.drop(columns=["_llm_group_", "_score_key_"]).copy()
    predictions.insert(0, "split", "test")
    predictions["target"] = y_test.to_numpy()
    predictions["score"] = scores.to_numpy()
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    _write_errors(output_dir / "llm_errors.jsonl", errors)
    _record(stage_times, "write_predictions", t)

    t = perf_counter()
    requested_groups = int(X_test["_llm_group_"].nunique())
    unique_score_units = int(X_to_score["_llm_group_"].nunique())

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
        "unique_score_units": unique_score_units,
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
            "json_list_columns": list(DATASET_JSON_LIST_COLUMNS[dataset_name]),
            "uses_item_stats": use_item_stats,
            "extra_body": extra_body,
            "scoring": scoring,
            "run_count": run_count,
            "tie_break": getattr(tie_break_fn, "__name__", "custom"),
            "dedup_strategy": "user_item_pair" if scoring == "itemwise" else f"batch_signature_{batch_dedup_mode}",
            "batch_dedup_mode": batch_dedup_mode,
        },
        "utility_aggregation": UTILITY_AGGREGATION,
        "limits": {
            "max_llm_groups": max_llm_groups,
            "max_llm_attempts": MAX_LLM_ATTEMPTS,
            "max_workers": max_workers,
        },
        "llm_errors": len(errors),
        "groups": {"requested": requested_groups, "unique_score_units": unique_score_units},
        "task": {"name": task.name, "manifest": json_safe(task.manifest)},
        "stage_times_seconds": stage_times,
        "git_commit": current_git_commit(repo_root()),
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / METRICS_FILENAME, metrics)
    write_policy_metrics(task, output_dir)
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
    run_count: int = 1,
    tie_break_fn: Callable[[list[float]], float] = majority_vote,
) -> tuple[pd.Series, list[dict[str, object]]]:
    scores = pd.Series(index=X.index, dtype=float, name="score")
    errors: list[dict[str, object]] = []
    groups = list(X.groupby(candidate_group_column, sort=False))

    if max_workers == 1:
        progress = tqdm(groups, desc="llm groups", unit="group")
        for group_id, group in progress:
            group_scores, error = _score_one_group(
                scorer, group_id, group,
                max_attempts=max_attempts,
                run_count=run_count,
                tie_break_fn=tie_break_fn,
            )
            if error is not None:
                errors.append(error)
                progress.set_postfix(errors=len(errors))
            else:
                scores.loc[group_scores.index] = group_scores
        return scores, errors

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _score_one_group, scorer, group_id, group,
                max_attempts=max_attempts,
                run_count=run_count,
                tie_break_fn=tie_break_fn,
            )
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
    run_count: int = 1,
    tie_break_fn: Callable[[list[float]], float] = majority_vote,
) -> tuple[pd.Series, dict[str, object] | None]:
    """Score one group, repeating up to run_count times and aggregating with tie_break_fn.

    Inner loop (max_attempts) retries on exceptions (network/parse failures).
    Outer loop (run_count) is deliberate repetition for noise reduction.
    If some but not all runs fail, the successful runs are aggregated.
    The group returns NaN only when every run exhausts all attempts.
    """
    per_run_results: list[pd.Series] = []
    all_errors: list[str] = []

    for _ in range(run_count):
        for attempt in range(max_attempts):
            try:
                per_run_results.append(scorer.score(group))
                break  # this run succeeded
            except Exception as exc:  # noqa: BLE001 — keep long runs alive
                if attempt == max_attempts - 1:
                    all_errors.append(repr(exc))

    if not per_run_results:
        empty = pd.Series(index=group.index, dtype=float, name="score")
        return empty, {
            "candidate_group": str(group_id),
            "attempts": run_count * max_attempts,
            "errors": all_errors,
        }

    if len(per_run_results) == 1:
        return per_run_results[0], None

    # Aggregate scores across all successful runs using tie_break_fn.
    aggregated = pd.Series(index=group.index, dtype=float, name="score")
    for idx in group.index:
        aggregated.loc[idx] = tie_break_fn([r.loc[idx] for r in per_run_results])
    return aggregated, None


def _write_errors(path: Path, errors: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for error in errors:
            f.write(json.dumps(error, sort_keys=True) + "\n")


def _record(stage_times: dict[str, float], stage: str, t0: float) -> None:
    from time import perf_counter
    stage_times[stage] = perf_counter() - t0
