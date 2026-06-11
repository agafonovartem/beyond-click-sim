from __future__ import annotations

"""Evaluate saved LLM/fixed-prediction interaction runs.

This script is safe only when `predictions.parquet` already contains fixed row-level
scores and, for pointwise metrics, final binary decisions from a fixed decision rule such
as the LLM yes/no parser. Do not use it for score-based methods whose predictions depend
on validation tuning; rerun those methods so their threshold or hyperparameters are
selected under the current protocol.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Literal

import pandas as pd

from beyond_click_sim.evaluation import (
    binary_classification_metrics,
    grouped_binary_classification_metrics,
    grouped_ranking_metrics,
    user_grouped_binary_classification_metrics,
    user_grouped_ranking_metrics,
)
from runners.in_distribution.interaction_prediction.metrics import (
    POINTWISE_MAIN_METRIC,
    POINTWISE_METRICS_FILENAME,
    RANKING_KS,
    RANKING_MAIN_METRIC,
    RANKING_METRICS_FILENAME,
    RANKING_TIE_POLICY,
)

LEGACY_METRICS_FILENAME = "metrics.legacy_macro_by_group.json"
MAIN_METRIC = POINTWISE_MAIN_METRIC
FIXED_PREDICTION_DECISION_RULES = {"hard_binary_yes_no_parser"}
EvaluationProtocol = Literal["pointwise", "ranking"]
DEFAULT_PROTOCOLS: tuple[EvaluationProtocol, ...] = ("pointwise", "ranking")
BASE_REQUIRED_PREDICTION_COLUMNS = {
    "split",
    "user_id",
    "candidate_group",
    "target",
}
POINTWISE_REQUIRED_PREDICTION_COLUMNS = {
    *BASE_REQUIRED_PREDICTION_COLUMNS,
    "prediction",
}
RANKING_REQUIRED_PREDICTION_COLUMNS = {
    *BASE_REQUIRED_PREDICTION_COLUMNS,
    "score",
}


@dataclass(frozen=True)
class EvaluationResult:
    run_dir: Path
    status: str
    message: str


def evaluate_run_predictions(
    run_dir: Path,
    *,
    protocols: tuple[EvaluationProtocol, ...] = DEFAULT_PROTOCOLS,
    dry_run: bool = False,
    force: bool = False,
) -> EvaluationResult:
    """Evaluate saved fixed-prediction LLM outputs for one run directory."""

    run_dir = run_dir.resolve()
    predictions_path = run_dir / "predictions.parquet"
    metrics_path = run_dir / POINTWISE_METRICS_FILENAME
    ranking_metrics_path = run_dir / RANKING_METRICS_FILENAME
    manifest_path = run_dir / "manifest.json"
    metrics_backup_path = run_dir / LEGACY_METRICS_FILENAME

    _require_supported_protocols(protocols)
    _require_files(predictions_path, metrics_path, manifest_path)
    manifest = _read_json(manifest_path)
    decision_rule = manifest.get("decision_rule", {})
    decision_rule_kind = decision_rule.get("kind")
    if decision_rule_kind not in FIXED_PREDICTION_DECISION_RULES:
        return EvaluationResult(
            run_dir=run_dir,
            status="skipped",
            message=(
                f"decision_rule.kind={decision_rule_kind!r} is not a known "
                "fixed-prediction rule; rerun method instead of metrics-only migration"
            ),
        )

    predictions = pd.read_parquet(predictions_path)
    pointwise_metrics: dict[str, Any] | None = None
    ranking_metrics: dict[str, Any] | None = None
    messages: list[str] = []

    if "pointwise" in protocols:
        _require_prediction_columns(predictions, POINTWISE_REQUIRED_PREDICTION_COLUMNS)
        if metrics_backup_path.exists() and not force:
            messages.append("skipped pointwise: legacy backup already exists")
        else:
            base_metrics_path = metrics_backup_path if metrics_backup_path.exists() else metrics_path
            pointwise_metrics = evaluate_metrics_payload(
                predictions,
                _read_json(base_metrics_path),
                protocols=("pointwise",),
            )
            messages.append(f"wrote {metrics_path.name}")

    if "ranking" in protocols:
        _require_prediction_columns(predictions, RANKING_REQUIRED_PREDICTION_COLUMNS)
        if ranking_metrics_path.exists() and not force:
            messages.append(f"skipped ranking: {ranking_metrics_path.name} already exists")
        else:
            ranking_metrics = evaluate_ranking_metrics_payload(
                predictions,
                pointwise_metrics if pointwise_metrics is not None else _read_json(metrics_path),
            )
            messages.append(f"wrote {ranking_metrics_path.name}")

    if dry_run:
        return EvaluationResult(
            run_dir=run_dir,
            status="dry-run",
            message=f"would process protocols={protocols}: {'; '.join(messages)}",
        )

    if pointwise_metrics is not None:
        _backup_once(metrics_path, metrics_backup_path)
        _write_json(metrics_path, pointwise_metrics)
    if ranking_metrics is not None:
        _write_json(ranking_metrics_path, ranking_metrics)

    wrote_anything = pointwise_metrics is not None or ranking_metrics is not None
    return EvaluationResult(
        run_dir=run_dir,
        status="updated" if wrote_anything else "skipped",
        message="; ".join(messages),
    )


def evaluate_metrics_payload(
    predictions: pd.DataFrame,
    base_metrics: dict[str, Any],
    *,
    protocols: tuple[EvaluationProtocol, ...] = ("pointwise",),
) -> dict[str, Any]:
    """Build an updated metrics payload from row-level fixed predictions."""

    _require_prediction_columns(predictions, POINTWISE_REQUIRED_PREDICTION_COLUMNS)
    if predictions.empty:
        raise ValueError("predictions.parquet is empty")
    _require_supported_protocols(protocols)

    updated = dict(base_metrics)
    if "pointwise" in protocols:
        updated["main_metric"] = MAIN_METRIC
        _evaluate_pointwise_predictions(predictions, updated)

    return updated


def evaluate_ranking_metrics_payload(
    predictions: pd.DataFrame,
    base_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build a ranking metrics payload from row-level scores."""

    _require_prediction_columns(predictions, RANKING_REQUIRED_PREDICTION_COLUMNS)
    if predictions.empty:
        raise ValueError("predictions.parquet is empty")

    updated = _ranking_base_payload(base_metrics)
    for split in _split_order(predictions["split"]):
        split_frame = predictions[predictions["split"].eq(split)]
        y_true = split_frame["target"]
        scores = split_frame["score"]
        updated[str(split)] = {
            "macro_by_group": grouped_ranking_metrics(
                y_true,
                scores,
                split_frame["candidate_group"],
                ks=RANKING_KS,
                tie_policy=RANKING_TIE_POLICY,
            ),
            "macro_by_user_group_mean": user_grouped_ranking_metrics(
                y_true,
                scores,
                split_frame["candidate_group"],
                split_frame["user_id"],
                ks=RANKING_KS,
                tie_policy=RANKING_TIE_POLICY,
            ),
        }

    return updated


def _evaluate_pointwise_predictions(
    predictions: pd.DataFrame,
    updated: dict[str, Any],
) -> None:
    """Update metrics payload with pointwise binary metrics by split."""

    for split in _split_order(predictions["split"]):
        split_frame = predictions[predictions["split"].eq(split)]
        y_true = split_frame["target"]
        y_pred = split_frame["prediction"]
        updated[str(split)] = {
            "macro_by_group": grouped_binary_classification_metrics(
                y_true,
                y_pred,
                split_frame["candidate_group"],
            ),
            "macro_by_user_group_mean": user_grouped_binary_classification_metrics(
                y_true,
                y_pred,
                split_frame["candidate_group"],
                split_frame["user_id"],
            ),
            "micro": binary_classification_metrics(y_true, y_pred),
        }


def discover_run_dirs(paths: list[Path], *, recursive: bool) -> list[Path]:
    """Return run directories from explicit paths or recursively from parents."""

    discovered: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        candidates: list[Path]
        if recursive:
            candidates = []
            if _is_run_dir(path):
                candidates.append(path)
            candidates.extend(
                candidate.parent
                for candidate in path.rglob("predictions.parquet")
                if _is_run_dir(candidate.parent)
            )
        else:
            candidates = [path]

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(resolved)

    return discovered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved LLM/fixed-prediction interaction outputs. "
            "Skips runs with validation-tuned decision rules."
        ),
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--protocol",
        choices=("pointwise", "ranking"),
        action="append",
        default=None,
        help=(
            "Evaluation protocol to run. Can be passed multiple times. "
            "Default: pointwise and ranking."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Treat inputs as parent directories and evaluate child run directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be evaluated without writing files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Evaluate and overwrite metrics.json even when legacy backups already exist.",
    )
    args = parser.parse_args(argv)

    try:
        protocols = _protocols_from_args(args.protocol)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    failed = False
    for run_dir in discover_run_dirs(args.run_dirs, recursive=args.recursive):
        try:
            result = evaluate_run_predictions(
                run_dir,
                protocols=protocols,
                dry_run=args.dry_run,
                force=args.force,
            )
        except Exception as error:  # pragma: no cover - CLI reporting path
            failed = True
            print(f"ERROR {run_dir}: {error}", file=sys.stderr)
            continue
        print(f"{result.status.upper()} {result.run_dir}: {result.message}")

    return 1 if failed else 0


def _require_files(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required run files: {missing}")


def _require_prediction_columns(
    predictions: pd.DataFrame,
    required_columns: set[str],
) -> None:
    missing = sorted(required_columns - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions.parquet is missing columns: {missing}")


def _require_supported_protocols(protocols: tuple[EvaluationProtocol, ...]) -> None:
    unsupported = sorted(set(protocols) - {"pointwise", "ranking"})
    if unsupported:
        raise ValueError(f"unsupported protocols {unsupported}")


def _protocols_from_args(protocols: list[str] | None) -> tuple[EvaluationProtocol, ...]:
    if protocols is None:
        return DEFAULT_PROTOCOLS

    unique_protocols = tuple(dict.fromkeys(protocols))
    _require_supported_protocols(unique_protocols)  # type: ignore[arg-type]
    return unique_protocols  # type: ignore[return-value]


def _ranking_base_payload(base_metrics: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol": "ranking",
        "main_metric": RANKING_MAIN_METRIC,
        "ranking_evaluation": {
            "ks": list(RANKING_KS),
            "tie_policy": RANKING_TIE_POLICY,
        },
    }
    for key in [
        "method",
        "task",
        "llm_errors",
        "scored_rows",
        "requested_rows",
        "max_candidate_groups",
        "max_workers",
        "candidate_groups",
    ]:
        if key in base_metrics:
            payload[key] = base_metrics[key]
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _backup_once(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    shutil.copy2(source, destination)


def _is_run_dir(path: Path) -> bool:
    return (
        (path / "predictions.parquet").exists()
        and (path / "metrics.json").exists()
        and (path / "manifest.json").exists()
    )


def _split_order(splits: pd.Series) -> list[Any]:
    return list(dict.fromkeys(splits.tolist()))


if __name__ == "__main__":
    raise SystemExit(main())
