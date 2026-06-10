from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import pandas as pd

from beyond_click_sim.evaluation import (
    binary_classification_metrics,
    grouped_binary_classification_metrics,
    user_grouped_binary_classification_metrics,
)


LEGACY_METRICS_FILENAME = "metrics.legacy_macro_by_group.json"
MAIN_METRIC = "test.macro_by_user_group_mean.f1"
REQUIRED_PREDICTION_COLUMNS = {
    "split",
    "user_id",
    "candidate_group",
    "target",
    "prediction",
}


@dataclass(frozen=True)
class RecomputeResult:
    run_dir: Path
    status: str
    message: str


def recompute_run_metrics(
    run_dir: Path,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> RecomputeResult:
    """Recompute fixed-prediction binary metrics for one run directory."""

    run_dir = run_dir.resolve()
    predictions_path = run_dir / "predictions.parquet"
    metrics_path = run_dir / "metrics.json"
    manifest_path = run_dir / "manifest.json"
    metrics_backup_path = run_dir / LEGACY_METRICS_FILENAME

    _require_files(predictions_path, metrics_path, manifest_path)
    manifest = _read_json(manifest_path)
    decision_rule = manifest.get("decision_rule", {})
    if decision_rule.get("kind") == "threshold_on_validation":
        return RecomputeResult(
            run_dir=run_dir,
            status="skipped",
            message="threshold_on_validation run; rerun method instead of metrics-only migration",
        )

    if metrics_backup_path.exists() and not force:
        return RecomputeResult(
            run_dir=run_dir,
            status="skipped",
            message="legacy backup already exists; use --force to recompute metrics.json",
        )

    base_metrics_path = metrics_backup_path if metrics_backup_path.exists() else metrics_path
    base_metrics = _read_json(base_metrics_path)
    predictions = pd.read_parquet(predictions_path)
    _require_prediction_columns(predictions)
    new_metrics = recompute_metrics_payload(predictions, base_metrics)

    if dry_run:
        return RecomputeResult(
            run_dir=run_dir,
            status="dry-run",
            message=f"would write {metrics_path.name} with {MAIN_METRIC}",
        )

    _backup_once(metrics_path, metrics_backup_path)
    _write_json(metrics_path, new_metrics)
    return RecomputeResult(
        run_dir=run_dir,
        status="updated",
        message=f"wrote {metrics_path.name} with {MAIN_METRIC}",
    )


def recompute_metrics_payload(
    predictions: pd.DataFrame,
    base_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build an updated metrics payload from row-level fixed predictions."""

    _require_prediction_columns(predictions)
    if predictions.empty:
        raise ValueError("predictions.parquet is empty")

    updated = dict(base_metrics)
    updated["main_metric"] = MAIN_METRIC

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

    return updated


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
        description="Recompute interaction-prediction metrics for fixed predictions.",
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Treat inputs as parent directories and migrate child run directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be migrated without writing files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute metrics.json even when legacy backups already exist.",
    )
    args = parser.parse_args(argv)

    failed = False
    for run_dir in discover_run_dirs(args.run_dirs, recursive=args.recursive):
        try:
            result = recompute_run_metrics(
                run_dir,
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


def _require_prediction_columns(predictions: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions.parquet is missing columns: {missing}")


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
