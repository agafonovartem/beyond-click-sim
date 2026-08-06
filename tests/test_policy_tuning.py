"""Tests for the offline Optuna policy tuner (`tasks/policy_tuning.py`)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from beyond_click_sim.tasks import MinUserInteractionsFilter, RandomFractionSplitter  # noqa: E402
from beyond_click_sim.tasks.base import TaskBuilder  # noqa: E402
from beyond_click_sim.tasks.policies import Policy  # noqa: E402
from beyond_click_sim.tasks.policy_tuning import (  # noqa: E402
    INNER_SEED_OFFSET,
    POLICY_ROSTER,
    artifact_filename,
    build_tuned_policy_roster,
    tune_policies_for_split,
)

TINY_SPACES: dict[str, Any] = {
    "ItemKNNPolicy": {
        "n_trials": 2,
        "params": {"n_neighbors": {"kind": "categorical", "choices": [1, 2]}},
    },
    "ALSPolicy": {
        "n_trials": 2,
        "params": {"n_factors": {"kind": "categorical", "choices": [2, 4]}},
    },
    "BPRPolicy": {
        "n_trials": 2,
        "params": {"learning_rate": {"kind": "categorical", "choices": [0.01, 0.05]}},
    },
    "LightGCNTorchPolicy": {
        "n_trials": 2,
        "params": {"n_layers": {"kind": "categorical", "choices": [1, 2]}},
    },
}


def _frames():
    users = pd.DataFrame({"user_id": [f"u{i}" for i in range(8)]})
    items = pd.DataFrame({"item_id": [f"i{i}" for i in range(12)]})
    rows = []
    for u in range(8):
        for offset in range(6):
            rows.append(
                {
                    "user_id": f"u{u}",
                    "item_id": f"i{(u * 3 + offset) % 12}",
                    "liked": int((u + offset) % 3 != 0),
                }
            )
    interactions = pd.DataFrame(rows).drop_duplicates(subset=["user_id", "item_id"])
    return users, items, interactions


def _configs():
    return (
        TaskBuilder._component_manifest(
            MinUserInteractionsFilter(min_interactions=3)
        ),
        TaskBuilder._component_manifest(
            RandomFractionSplitter(
                train_fraction=0.8, val_fraction=0.0, test_fraction=0.2, seed=0
            )
        ),
    )


def _tune(**overrides) -> dict[str, Any]:
    users, items, interactions = _frames()
    outer = RandomFractionSplitter(
        train_fraction=0.8, val_fraction=0.0, test_fraction=0.2, seed=0
    ).split(interactions)
    filter_config, splitter_config = _configs()
    kwargs: dict[str, Any] = dict(
        train_interactions=outer.train,
        users=users,
        items=items,
        dataset_name="toy",
        dataset_version="v1",
        target_source_column="liked",
        positive_feedback_column=None,
        split_seed=0,
        policy_k=3,
        filter_config=filter_config,
        splitter_config=splitter_config,
        search_spaces=TINY_SPACES,
    )
    kwargs.update(overrides)
    return tune_policies_for_split(**kwargs)


class _StubPolicy(Policy):
    """Deterministic stub: recommends the first k catalog items to everyone."""

    def __init__(self, k, seed=0, positive_feedback_column=None, magic=0):
        super().__init__(
            k=k, seed=seed, positive_feedback_column=positive_feedback_column
        )
        self.magic = magic

    def fit(self, train_interactions, *, items, user_column="user_id", item_column="item_id"):
        return self

    def recommend(self, users, *, train_interactions, items, user_column="user_id", item_column="item_id"):
        rows = []
        for user_id in users[user_column]:
            rows.extend(
                self._build_recommendation_rows(
                    user_id,
                    items[item_column].tolist()[: self.k],
                    user_column=user_column,
                    item_column=item_column,
                )
            )
        return pd.DataFrame(rows)


class _SpyPolicy(_StubPolicy):
    """Records the exact (user, item) pairs seen by fit and recommend."""

    instances: list["_SpyPolicy"] = []

    def fit(self, train_interactions, *, items, user_column="user_id", item_column="item_id"):
        self.fit_pairs_ = set(
            zip(train_interactions[user_column], train_interactions[item_column])
        )
        type(self).instances.append(self)
        return super().fit(
            train_interactions, items=items, user_column=user_column, item_column=item_column
        )

    def recommend(self, users, *, train_interactions, items, user_column="user_id", item_column="item_id"):
        self.recommend_pairs_ = set(
            zip(train_interactions[user_column], train_interactions[item_column])
        )
        return super().recommend(
            users,
            train_interactions=train_interactions,
            items=items,
            user_column=user_column,
            item_column=item_column,
        )


class _FlakyPolicy(_StubPolicy):
    """Fails every second fit attempt, deterministically."""

    attempts: int = 0

    def fit(self, train_interactions, *, items, user_column="user_id", item_column="item_id"):
        type(self).attempts += 1
        if type(self).attempts % 2 == 1:
            raise RuntimeError("boom")
        return super().fit(
            train_interactions, items=items, user_column=user_column, item_column=item_column
        )


class _AlwaysFailsPolicy(_StubPolicy):
    def fit(self, train_interactions, *, items, user_column="user_id", item_column="item_id"):
        raise RuntimeError("always")


# ---------------------------------------------------------------------------
# Artifact content
# ---------------------------------------------------------------------------


def test_artifact_contains_all_policies_with_full_trial_logs():
    artifact = _tune()

    assert artifact["schema_version"] == 1
    assert set(artifact["policies"]) == {cls.__name__ for cls in POLICY_ROSTER}

    for name in ("RandomPolicy", "PopularityPolicy"):
        entry = artifact["policies"][name]
        assert entry["tuned"] is False
        assert entry["selected_params"] == {}
        assert len(entry["trials"]) == 1

    for name, spec in TINY_SPACES.items():
        entry = artifact["policies"][name]
        assert entry["tuned"] is True
        assert len(entry["trials"]) == spec["n_trials"]
        for trial in entry["trials"]:
            assert trial["state"] == "complete"
            ranking = trial["metrics"]["ranking"]
            assert "ndcg@10" in ranking["macro_by_user_group_mean"]
            assert "fit_recommend_seconds" in trial["metrics"]
        # Selection must be the argmax over the logged trials.
        best = max(entry["trials"], key=lambda t: t["value"])
        assert entry["selected_params"] == best["params"]
        assert entry["val_objective"] == best["value"]


def test_tune_is_deterministic():
    first = _tune()
    second = _tune()
    for name in first["policies"]:
        a, b = first["policies"][name], second["policies"][name]
        assert a["selected_params"] == b["selected_params"], name
        assert a["val_objective"] == b["val_objective"], name
        # Trial-level params and objective values are reproducible;
        # wall-clock timings in the metrics payload are not.
        assert [t["params"] for t in a["trials"]] == [
            t["params"] for t in b["trials"]
        ], name
        assert [t.get("value") for t in a["trials"]] == [
            t.get("value") for t in b["trials"]
        ], name


def test_artifact_is_json_serializable():
    artifact = _tune()
    json.dumps(artifact)


# ---------------------------------------------------------------------------
# Leakage: the inner split must stay inside the outer train frame
# ---------------------------------------------------------------------------


def test_inner_split_never_leaks_val_into_fit_or_recommend():
    _SpyPolicy.instances = []
    _tune(
        policy_classes=(_SpyPolicy,),
        search_spaces={
            "_SpyPolicy": {
                "n_trials": 2,
                "params": {"magic": {"kind": "categorical", "choices": [0, 1]}},
            }
        },
    )
    assert len(_SpyPolicy.instances) == 2

    _, _, interactions = _frames()
    outer = RandomFractionSplitter(
        train_fraction=0.8, val_fraction=0.0, test_fraction=0.2, seed=0
    ).split(interactions)
    inner = RandomFractionSplitter(
        train_fraction=0.75,
        val_fraction=0.25,
        test_fraction=0.0,
        seed=0 + INNER_SEED_OFFSET,
    ).split(outer.train)
    inner_train_pairs = set(zip(inner.train["user_id"], inner.train["item_id"]))
    inner_val_pairs = set(zip(inner.val["user_id"], inner.val["item_id"]))
    assert inner_val_pairs  # the test is vacuous otherwise

    for instance in _SpyPolicy.instances:
        assert instance.fit_pairs_ == inner_train_pairs
        assert instance.recommend_pairs_ == inner_train_pairs
        assert instance.fit_pairs_.isdisjoint(inner_val_pairs)


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_failed_trials_are_logged_and_search_continues():
    _FlakyPolicy.attempts = 0
    artifact = _tune(
        policy_classes=(_FlakyPolicy,),
        search_spaces={
            "_FlakyPolicy": {
                "n_trials": 2,
                "params": {"magic": {"kind": "categorical", "choices": [0]}},
            }
        },
    )
    trials = artifact["policies"]["_FlakyPolicy"]["trials"]
    assert [t["state"] for t in trials] == ["fail", "complete"]
    assert "boom" in trials[0]["error"]


def test_all_trials_failed_raises():
    with pytest.raises(RuntimeError, match="refusing to write"):
        _tune(
            policy_classes=(_AlwaysFailsPolicy,),
            search_spaces={
                "_AlwaysFailsPolicy": {
                    "n_trials": 2,
                    "params": {"magic": {"kind": "categorical", "choices": [0]}},
                }
            },
        )


# ---------------------------------------------------------------------------
# Artifact loading / roster construction
# ---------------------------------------------------------------------------


def _write_artifact(tmp_path: Path, artifact: dict[str, Any]) -> Path:
    path = tmp_path / artifact_filename("toy", "liked", 0)
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def _roster_kwargs(tmp_path: Path) -> dict[str, Any]:
    filter_config, splitter_config = _configs()
    return dict(
        artifacts_dir=tmp_path,
        dataset_name="toy",
        dataset_version="v1",
        target_source_column="liked",
        positive_feedback_column=None,
        split_seed=0,
        policy_k=3,
        filter_config=filter_config,
        splitter_config=splitter_config,
        search_spaces=TINY_SPACES,
    )


def test_roster_roundtrip_applies_tuned_params(tmp_path):
    artifact = _tune()
    _write_artifact(tmp_path, artifact)

    roster = build_tuned_policy_roster(**_roster_kwargs(tmp_path))

    assert [policy.name for policy in roster] == [
        cls.__name__ for cls in POLICY_ROSTER
    ]
    knn = roster[2]
    assert knn.n_neighbors == (
        artifact["policies"]["ItemKNNPolicy"]["selected_params"]["n_neighbors"]
    )
    for policy in roster:
        assert policy.hyperparameter_selection is not None
        assert policy.hyperparameter_selection["artifact"].endswith(
            "toy_liked_seed0.json"
        )
    assert roster[0].hyperparameter_selection["kind"] == "fixed_baseline"
    assert knn.hyperparameter_selection["kind"] == "optuna_tpe_search"
    assert knn.hyperparameter_selection["val_objective"] == (
        artifact["policies"]["ItemKNNPolicy"]["val_objective"]
    )


def test_missing_artifact_raises_with_instructions(tmp_path):
    with pytest.raises(RuntimeError, match="policy_tuning.run"):
        build_tuned_policy_roster(**_roster_kwargs(tmp_path))


@pytest.mark.parametrize(
    "field",
    [
        "split_seed",
        "target_source_column",
        "positive_feedback_column",
        "filter",
        "search_spaces_sha256",
    ],
)
def test_stale_identity_raises(tmp_path, field):
    artifact = _tune()
    artifact["identity"][field] = "tampered"
    _write_artifact(tmp_path, artifact)

    with pytest.raises(ValueError, match="Stale"):
        build_tuned_policy_roster(**_roster_kwargs(tmp_path))


# ---------------------------------------------------------------------------
# Runner seed resolution
# ---------------------------------------------------------------------------


def test_resolve_seeds_defaults_to_each_setups_registry():
    from runners.in_distribution.policy_ranking_agreement.task_builders import (
        SEEDS as INTERACTION_SEEDS,
    )
    from runners.in_distribution.policy_ranking_agreement_preference.task_builders import (
        SEEDS as PREFERENCE_SEEDS,
    )
    from runners.in_distribution.policy_tuning.run import resolve_seeds

    assert resolve_seeds("interaction", None) == tuple(INTERACTION_SEEDS)
    assert resolve_seeds("preference", None) == tuple(PREFERENCE_SEEDS)
    # Both setups tune the same seed range by default.
    assert tuple(INTERACTION_SEEDS) == tuple(PREFERENCE_SEEDS)


def test_resolve_seeds_explicit_override_applies_to_all_setups():
    from runners.in_distribution.policy_tuning.run import resolve_seeds

    assert resolve_seeds("interaction", [3, 4]) == (3, 4)
    assert resolve_seeds("preference", [3, 4]) == (3, 4)
