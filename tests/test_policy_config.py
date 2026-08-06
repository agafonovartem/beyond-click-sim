"""Tests for config-driven policy rosters (`tasks/policy_config.py`)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from beyond_click_sim.tasks.policy_config import (  # noqa: E402
    PolicyConfigError,
    build_config_skeleton,
    load_policy_roster,
)


def _payload(
    roster: list[dict] | None = None,
    *,
    dataset: str = "ml-1m",
    setup: str = "preference",
    **dataset_extra: Any,
) -> dict[str, Any]:
    if roster is None:
        roster = [
            {"name": "random", "class": "RandomPolicy"},
            {
                "name": "als",
                "class": "ALSPolicy",
                "params": {"n_factors": 32},
                "tuned_from": "artifacts/ml-1m_target_like_ge4_seed0.json",
                "comment": "tuned pick",
            },
        ]
    return {
        "schema_version": 1,
        "setup": setup,
        "datasets": {dataset: {"default": roster, **dataset_extra}},
    }


def _write_config(tmp_path: Path, payload: dict, setup: str = "preference") -> Path:
    path = tmp_path / f"{setup}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def _load(config_dir: Path, **overrides: Any):
    kwargs: dict[str, Any] = dict(
        setup="preference",
        dataset_name="ml-1m",
        split_seed=0,
        policy_k=10,
        positive_feedback_column="liked",
        config_dir=config_dir,
    )
    kwargs.update(overrides)
    return load_policy_roster(**kwargs)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_default_roster_loaded(tmp_path):
    config_dir = _write_config(tmp_path, _payload())
    roster = _load(config_dir)

    assert [p.name for p in roster] == ["random", "als"]
    assert roster[1].n_factors == 32
    assert roster[1].k == 10
    assert roster[1].positive_feedback_column == "liked"
    selection = roster[1].hyperparameter_selection
    assert selection["kind"] == "manual_config"
    assert selection["section"] == "datasets.ml-1m.default"
    assert selection["tuned_from"] == "artifacts/ml-1m_target_like_ge4_seed0.json"
    assert selection["comment"] == "tuned pick"


def test_seed_override_wins_over_default(tmp_path):
    payload = _payload(
        seed0=[
            {"name": "als", "class": "ALSPolicy", "params": {"n_factors": 64}},
        ]
    )
    config_dir = _write_config(tmp_path, payload)

    seed0_roster = _load(config_dir, split_seed=0)
    assert [p.name for p in seed0_roster] == ["als"]
    assert seed0_roster[0].n_factors == 64

    seed1_roster = _load(config_dir, split_seed=1)
    assert [p.name for p in seed1_roster] == ["random", "als"]
    assert seed1_roster[1].n_factors == 32


def test_same_class_variants_coexist_via_names(tmp_path):
    payload = _payload(
        roster=[
            {"name": "als_f32", "class": "ALSPolicy", "params": {"n_factors": 32}},
            {"name": "als_f64", "class": "ALSPolicy", "params": {"n_factors": 64}},
        ]
    )
    roster = _load(_write_config(tmp_path, payload))
    assert [p.name for p in roster] == ["als_f32", "als_f64"]
    assert roster[0].n_factors == 32
    assert roster[1].n_factors == 64


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_missing_file_raises_with_path(tmp_path):
    with pytest.raises(RuntimeError, match="preference.json"):
        _load(tmp_path)


def test_setup_mismatch_raises(tmp_path):
    config_dir = _write_config(tmp_path, _payload(setup="interaction"))
    with pytest.raises(PolicyConfigError, match="setup mismatch"):
        _load(config_dir)


def test_missing_dataset_section_raises(tmp_path):
    config_dir = _write_config(tmp_path, _payload(dataset="steam"))
    with pytest.raises(PolicyConfigError, match="ml-1m"):
        _load(config_dir)


def test_missing_default_and_seed_section_raises(tmp_path):
    payload = _payload()
    payload["datasets"]["ml-1m"] = {
        "seed5": [{"name": "random", "class": "RandomPolicy"}]
    }
    config_dir = _write_config(tmp_path, payload)
    with pytest.raises(PolicyConfigError, match="seed0.*default"):
        _load(config_dir, split_seed=0)


def test_unknown_section_key_raises(tmp_path):
    payload = _payload()
    payload["datasets"]["ml-1m"]["seed_3"] = [
        {"name": "random", "class": "RandomPolicy"}
    ]
    config_dir = _write_config(tmp_path, payload)
    with pytest.raises(PolicyConfigError, match="seed_3"):
        _load(config_dir)


def test_duplicate_names_rejected(tmp_path):
    payload = _payload(
        roster=[
            {"name": "als", "class": "ALSPolicy"},
            {"name": "als", "class": "ALSPolicy", "params": {"n_factors": 64}},
        ]
    )
    config_dir = _write_config(tmp_path, payload)
    with pytest.raises(PolicyConfigError, match="duplicate policy name"):
        _load(config_dir)


def test_unknown_class_rejected(tmp_path):
    payload = _payload(roster=[{"name": "x", "class": "XGBoostPolicy"}])
    config_dir = _write_config(tmp_path, payload)
    with pytest.raises(PolicyConfigError, match="unknown class"):
        _load(config_dir)


def test_unknown_entry_key_rejected(tmp_path):
    payload = _payload(
        roster=[{"name": "als", "class": "ALSPolicy", "parms": {"n_factors": 32}}]
    )
    config_dir = _write_config(tmp_path, payload)
    with pytest.raises(PolicyConfigError, match="unknown keys"):
        _load(config_dir)


def test_reserved_param_key_rejected(tmp_path):
    payload = _payload(
        roster=[{"name": "als", "class": "ALSPolicy", "params": {"k": 5}}]
    )
    config_dir = _write_config(tmp_path, payload)
    with pytest.raises(PolicyConfigError, match="injected"):
        _load(config_dir)


def test_bad_param_value_wrapped_with_entry_context(tmp_path):
    payload = _payload(
        roster=[{"name": "als", "class": "ALSPolicy", "params": {"n_factors": 0}}]
    )
    config_dir = _write_config(tmp_path, payload)
    with pytest.raises(PolicyConfigError, match="'als'.*n_factors"):
        _load(config_dir)


def test_unknown_param_name_wrapped_with_entry_context(tmp_path):
    payload = _payload(
        roster=[{"name": "als", "class": "ALSPolicy", "params": {"layers": 3}}]
    )
    config_dir = _write_config(tmp_path, payload)
    with pytest.raises(PolicyConfigError, match="'als'"):
        _load(config_dir)


# ---------------------------------------------------------------------------
# Skeleton emission from tuning artifacts
# ---------------------------------------------------------------------------


def _fake_artifact(params_by_policy: dict[str, dict], *, target: str) -> dict:
    return {
        "schema_version": 1,
        "identity": {"target_source_column": target},
        "policies": {
            "RandomPolicy": {"tuned": False, "selected_params": {}},
            **{
                name: {"tuned": True, "selected_params": params}
                for name, params in params_by_policy.items()
            },
        },
    }


def test_skeleton_uses_lowest_seed_and_marks_baselines():
    artifacts = {
        0: _fake_artifact({"ItemKNNPolicy": {"n_neighbors": 10}}, target="liked"),
        1: _fake_artifact({"ItemKNNPolicy": {"n_neighbors": 50}}, target="liked"),
    }
    skeleton = build_config_skeleton(
        setup="preference", dataset_sections={"ml-1m": artifacts}
    )

    section = skeleton["datasets"]["ml-1m"]
    roster = section["default"]
    knn = next(e for e in roster if e["class"] == "ItemKNNPolicy")
    assert knn["params"] == {"n_neighbors": 10}
    assert knn["tuned_from"] == "artifacts/ml-1m_liked_seed0.json"
    random_entry = next(e for e in roster if e["class"] == "RandomPolicy")
    assert random_entry["params"] == {}
    assert "baseline" in random_entry["comment"]
    # Params diverge across seeds -> the review notes must say so.
    assert any("differ across seeds" in note for note in section["_review_notes"])


def test_skeleton_notes_stable_params_across_seeds():
    artifacts = {
        seed: _fake_artifact({"ItemKNNPolicy": {"n_neighbors": 10}}, target="liked")
        for seed in (0, 1, 2)
    }
    skeleton = build_config_skeleton(
        setup="interaction", dataset_sections={"steam": artifacts}
    )
    notes = skeleton["datasets"]["steam"]["_review_notes"]
    assert any("same params selected on all 3 seeds" in note for note in notes)
