from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.in_distribution.interaction_prediction.recompute_metrics import (
    LEGACY_METRICS_FILENAME,
    MAIN_METRIC,
    recompute_run_metrics,
)


def test_recompute_run_metrics_migrates_fixed_prediction_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "llm_run"
    run_dir.mkdir()
    predictions = pd.DataFrame(
        {
            "split": ["test"] * 6,
            "user_id": ["u1", "u1", "u1", "u1", "u2", "u2"],
            "candidate_group": [
                "u1-g1",
                "u1-g1",
                "u1-g2",
                "u1-g2",
                "u2-g1",
                "u2-g1",
            ],
            "target": [1, 0, 1, 0, 1, 0],
            "prediction": [True, True, False, False, True, False],
            "score": [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        }
    )
    original_metrics = {
        "method": "llm_yes_no_test",
        "task": "toy",
        "main_metric": "test.macro_by_group.f1",
        "test": {
            "macro_by_group": {"f1": -1.0},
            "micro": {"f1": -1.0},
        },
        "llm_errors": 0,
    }
    original_manifest = {
        "method": "llm_yes_no_test",
        "decision_rule": {"kind": "hard_binary_yes_no_parser", "threshold": None},
    }
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    _write_json(run_dir / "metrics.json", original_metrics)
    _write_json(run_dir / "manifest.json", original_manifest)

    result = recompute_run_metrics(run_dir)

    assert result.status == "updated"
    assert _read_json(run_dir / LEGACY_METRICS_FILENAME) == original_metrics
    assert _read_json(run_dir / "manifest.json") == original_manifest
    assert not (run_dir / "manifest.legacy_macro_by_group.json").exists()

    migrated_metrics = _read_json(run_dir / "metrics.json")
    assert migrated_metrics["main_metric"] == MAIN_METRIC
    assert migrated_metrics["test"]["macro_by_group"]["f1"] == pytest.approx(
        (2 / 3 + 0.0 + 1.0) / 3
    )
    assert migrated_metrics["test"]["macro_by_user_group_mean"]["f1"] == pytest.approx(
        (((2 / 3) + 0.0) / 2 + 1.0) / 2
    )
    assert migrated_metrics["test"]["macro_by_user_group_mean"]["n_users"] == 2
    assert migrated_metrics["test"]["macro_by_user_group_mean"]["n_groups"] == 3
    assert migrated_metrics["test"]["micro"]["n"] == 6


def test_recompute_run_metrics_skips_threshold_selected_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "popularity_run"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "split": ["val", "test"],
            "user_id": ["u1", "u1"],
            "candidate_group": ["g1", "g2"],
            "target": [1, 0],
            "prediction": [True, False],
        }
    ).to_parquet(run_dir / "predictions.parquet", index=False)
    metrics = {
        "method": "popularity_f1_threshold",
        "main_metric": "test.macro_by_group.f1",
    }
    manifest = {
        "method": "popularity_f1_threshold",
        "decision_rule": {"kind": "threshold_on_validation", "threshold": 1.0},
    }
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "manifest.json", manifest)

    result = recompute_run_metrics(run_dir)

    assert result.status == "skipped"
    assert "threshold_on_validation" in result.message
    assert _read_json(run_dir / "metrics.json") == metrics
    assert not (run_dir / LEGACY_METRICS_FILENAME).exists()
    assert not (run_dir / "manifest.legacy_macro_by_group.json").exists()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
