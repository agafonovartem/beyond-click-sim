from __future__ import annotations

from pathlib import Path

from beyond_click_sim.evaluation import (
    binary_classification_metrics,
    grouped_binary_classification_metrics,
    user_grouped_binary_classification_metrics,
)
from beyond_click_sim.llm_clients import make_llm_client
from beyond_click_sim.scorers import Agent4RecProfileGenerator, Agent4RecYesNoScorer
from beyond_click_sim.tasks import PREFIXED_ITEM_RATING_MEAN_COLUMN, Task
from runners.in_distribution.interaction_prediction.methods.common import (
    candidate_group_summary,
    current_git_commit,
    limit_candidate_groups,
    prediction_frame,
    ranking_metrics_for_split,
    task_xy,
    write_json,
)
from runners.in_distribution.interaction_prediction.methods.llm_yes_no import (
    _score_groups,
    _write_errors,
)
from runners.in_distribution.interaction_prediction.metrics import (
    POINTWISE_MAIN_METRIC,
    POINTWISE_METRICS_FILENAME,
    RANKING_KS,
    RANKING_MAIN_METRIC,
    RANKING_METRICS_FILENAME,
    RANKING_TIE_POLICY,
)
from runners.in_distribution.interaction_prediction.task_builders import repo_root


OLLAMA_LLAMA31_8B_METHOD_NAME = "agent4rec_yes_no_ollama_llama31_8b"
OLLAMA_LLAMA31_8B_CLIENT = "ollama_local"
OLLAMA_LLAMA31_8B_MODEL = "llama3.1:8b"
VLLM_LLAMA33_70B_METHOD_NAME = "agent4rec_yes_no_vllm_llama33_70b"
VLLM_LLAMA33_70B_CLIENT = "vllm_local"
VLLM_LLAMA33_70B_MODEL = "llama-3.3-70b-instruct"
TEMPERATURE = 0.2
MAX_TOKENS = 1000
MAX_LLM_ATTEMPTS = 5
OLLAMA_MAX_WORKERS = 1
VLLM_MAX_WORKERS = 32

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


def run_llama31_8b_smoke(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OLLAMA_LLAMA31_8B_METHOD_NAME}_smoke",
        client_name=OLLAMA_LLAMA31_8B_CLIENT,
        model=OLLAMA_LLAMA31_8B_MODEL,
        max_candidate_groups=25,
        max_workers=OLLAMA_MAX_WORKERS,
    )


def run_llama31_8b_full(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{OLLAMA_LLAMA31_8B_METHOD_NAME}_full",
        client_name=OLLAMA_LLAMA31_8B_CLIENT,
        model=OLLAMA_LLAMA31_8B_MODEL,
        max_candidate_groups=None,
        max_workers=OLLAMA_MAX_WORKERS,
    )


def run_llama33_70b_smoke(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_LLAMA33_70B_METHOD_NAME}_smoke",
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        max_candidate_groups=25,
        max_workers=VLLM_MAX_WORKERS,
    )


def run_llama33_70b_full(task: Task, output_dir: Path) -> dict[str, object]:
    return run_method(
        task,
        output_dir,
        method_name=f"{VLLM_LLAMA33_70B_METHOD_NAME}_full",
        client_name=VLLM_LLAMA33_70B_CLIENT,
        model=VLLM_LLAMA33_70B_MODEL,
        max_candidate_groups=None,
        max_workers=VLLM_MAX_WORKERS,
    )


def run_method(
    task: Task,
    output_dir: Path,
    *,
    method_name: str,
    client_name: str,
    model: str,
    max_candidate_groups: int | None,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    max_llm_attempts: int = MAX_LLM_ATTEMPTS,
    max_workers: int = 1,
) -> dict[str, object]:
    """Run the Agent4Rec profile-based yes/no scorer."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = str(task.manifest["dataset"])
    if dataset_name not in DATASET_CANDIDATE_COLUMNS:
        raise ValueError(
            "Agent4Rec yes/no v1 supports only datasets with MovieLens-style "
            f"rating and genre fields. Got dataset: {dataset_name!r}"
        )
    candidate_group_column = task.schema.candidate_group_column
    if candidate_group_column is None:
        raise ValueError("Agent4Rec yes/no method requires candidate_group_column")

    xy = task_xy(task)
    X_train, y_train = xy["train"]
    X_test, y_test = limit_candidate_groups(
        *xy["test"],
        candidate_group_column=candidate_group_column,
        max_candidate_groups=max_candidate_groups,
    )
    _require_columns(X_test, list(DATASET_CANDIDATE_COLUMNS[dataset_name]))

    profile_generator = Agent4RecProfileGenerator(profile_components=("traits",))
    scorer = Agent4RecYesNoScorer(
        client=make_llm_client(client_name),
        model=model,
        profile_generator=profile_generator,
        candidate_description_columns=DATASET_CANDIDATE_COLUMNS[dataset_name],
        column_labels=DATASET_COLUMN_LABELS[dataset_name],
        temperature=temperature,
        max_tokens=max_tokens,
    ).fit(X_train, y_train)

    scores, errors = _score_groups(
        scorer,
        X_test,
        candidate_group_column=candidate_group_column,
        max_attempts=max_llm_attempts,
        max_workers=max_workers,
    )
    valid = scores.notna()
    predictions = scores.astype("boolean").rename("prediction")
    prediction_frame(
        split="test",
        X=X_test,
        y=y_test,
        scores=scores,
        predictions=predictions,
    ).to_parquet(output_dir / "predictions.parquet", index=False)
    _write_errors(output_dir / "llm_errors.jsonl", errors)

    if not valid.any():
        raise RuntimeError("Agent4Rec scorer did not produce any valid scores")

    valid_scores = scores.loc[valid]
    valid_X = X_test.loc[valid].copy()
    valid_y = y_test.loc[valid].copy()
    valid_predictions = valid_scores.astype(bool).rename("prediction")
    macro_metrics = grouped_binary_classification_metrics(
        valid_y,
        valid_predictions,
        valid_X[candidate_group_column],
    )
    user_group_metrics = user_grouped_binary_classification_metrics(
        valid_y,
        valid_predictions,
        valid_X[candidate_group_column],
        valid_X["user_id"],
    )
    micro_metrics = binary_classification_metrics(valid_y, valid_predictions)
    ranking_metrics = ranking_metrics_for_split(
        X=valid_X,
        y=valid_y,
        scores=valid_scores,
        candidate_group_column=candidate_group_column,
    )
    requested_candidate_groups = candidate_group_summary(
        X_test,
        y_test,
        candidate_group_column=candidate_group_column,
        sampled_column=task.schema.sampled_column,
    )
    scored_candidate_groups = candidate_group_summary(
        valid_X,
        valid_y,
        candidate_group_column=candidate_group_column,
        sampled_column=task.schema.sampled_column,
    )

    root = repo_root()
    manifest = {
        "method": method_name,
        "scorer": {
            "class": "Agent4RecYesNoScorer",
            "client_name": client_name,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "candidate_description_columns": list(
                DATASET_CANDIDATE_COLUMNS[dataset_name]
            ),
            "column_labels": DATASET_COLUMN_LABELS[dataset_name],
            "profile_generator": scorer.profile_generator.manifest(),
        },
        "decision_rule": {
            "kind": "hard_binary_yes_no_parser",
            "parser_contract": "agent4rec_labeled_id_movie_watch_reason",
            "threshold": None,
        },
        "limits": {
            "max_candidate_groups": max_candidate_groups,
            "max_llm_attempts": max_llm_attempts,
            "max_workers": max_workers,
        },
        "llm_errors": len(errors),
        "candidate_groups": {
            "requested": requested_candidate_groups,
            "scored": scored_candidate_groups,
        },
        "task": {
            "name": task.name,
            "manifest": task.manifest,
        },
        "git_commit": current_git_commit(root),
    }
    result = {
        "method": method_name,
        "task": task.name,
        "main_metric": POINTWISE_MAIN_METRIC,
        "test": {
            "macro_by_group": macro_metrics,
            "macro_by_user_group_mean": user_group_metrics,
            "micro": micro_metrics,
        },
        "llm_errors": len(errors),
        "scored_rows": int(valid.sum()),
        "requested_rows": int(len(X_test)),
        "max_candidate_groups": max_candidate_groups,
        "max_workers": max_workers,
        "candidate_groups": {
            "requested": requested_candidate_groups,
            "scored": scored_candidate_groups,
        },
    }
    ranking_result = {
        "method": method_name,
        "task": task.name,
        "protocol": "ranking",
        "main_metric": RANKING_MAIN_METRIC,
        "ranking_evaluation": {
            "ks": list(RANKING_KS),
            "tie_policy": RANKING_TIE_POLICY,
        },
        "test": ranking_metrics,
        "llm_errors": len(errors),
        "scored_rows": int(valid.sum()),
        "requested_rows": int(len(X_test)),
        "max_candidate_groups": max_candidate_groups,
        "max_workers": max_workers,
        "candidate_groups": {
            "requested": requested_candidate_groups,
            "scored": scored_candidate_groups,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / POINTWISE_METRICS_FILENAME, result)
    write_json(output_dir / RANKING_METRICS_FILENAME, ranking_result)
    return result


def _require_columns(frame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            "Agent4Rec yes/no requires item-stats task columns for the candidate "
            f"prompt. Missing columns: {missing}"
        )
