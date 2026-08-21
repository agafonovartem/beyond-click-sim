"""Tests for the Coat re-ranking tuner (`tasks/policy_tuning_rerank.py`).

The properties worth pinning here are the ones whose violation would be silent:
a starved objective still returns a number, and a mis-joined label still ranks
policies. See the module docstring for why this protocol exists at all.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from beyond_click_sim.tasks import MinUserInteractionsFilter  # noqa: E402
from beyond_click_sim.tasks.base import TaskBuilder  # noqa: E402
from beyond_click_sim.tasks.policies import PopularityPolicy  # noqa: E402
from beyond_click_sim.tasks.policy_tuning import POLICY_ROSTER  # noqa: E402
from beyond_click_sim.tasks.policy_tuning_rerank import (  # noqa: E402
    OBJECTIVE_METRIC,
    PROTOCOL,
    _slate_frame,
    artifact_filename,
    select_tuning_users,
    tune_policies_for_rerank_split,
)
from beyond_click_sim.tasks.splitters import SourcePartitionSplitter  # noqa: E402

N_USERS = 12
POOL = 8
K = 3

TINY_SPACES: dict[str, Any] = {
    "ItemKNNPolicy": {
        "n_trials": 2,
        "params": {"n_neighbors": {"kind": "categorical", "choices": [1, 2]}},
    },
}


def _frames():
    """A miniature of Coat: a self-selected half and a disjoint uniform pool."""

    users = pd.DataFrame({"user_id": [f"u{i}" for i in range(N_USERS)]})
    items = pd.DataFrame({"item_id": [f"i{i}" for i in range(24)]})
    rows = []
    for u in range(N_USERS):
        for offset in range(6):
            rows.append({
                "user_id": f"u{u}",
                "item_id": f"i{(u * 2 + offset) % 24}",
                "source": "self_selected",
                "liked": int((u + offset) % 3 != 0),
            })
        for slot in range(POOL):
            rows.append({
                "user_id": f"u{u}",
                "item_id": f"i{(u * 5 + slot * 3) % 24}",
                "source": "random",
                "liked": int((u + slot) % 4 == 0),
            })
    interactions = pd.DataFrame(rows).drop_duplicates(subset=["user_id", "item_id", "source"])
    return users, items, interactions


def _tune(**overrides) -> dict[str, Any]:
    users, items, interactions = _frames()
    dataset_filter = MinUserInteractionsFilter(min_interactions=2)
    splitter = SourcePartitionSplitter()
    split = splitter.split(interactions)
    kwargs: dict[str, Any] = dict(
        train_interactions=split.train,
        val_interactions=split.test,
        users=users,
        items=items,
        dataset_name="toy",
        dataset_version="v1",
        target_source_column="liked",
        positive_feedback_column="liked",
        split_seed=0,
        policy_k=K,
        filter_config=TaskBuilder._component_manifest(dataset_filter),
        splitter_config=TaskBuilder._component_manifest(splitter),
        search_spaces=TINY_SPACES,
        policy_classes=None,
    )
    kwargs.update(overrides)
    return tune_policies_for_rerank_split(**kwargs)


# --------------------------------------------------------------------------
# The properties that would otherwise fail silently
# --------------------------------------------------------------------------


def test_objective_equals_the_runners_per_user_utility():
    """`mean_hit_rate` must equal the macro-by-user mean the runner reports.

    The two coincide only while every user contributes exactly k rows. If that
    ever stops holding, tuning silently optimizes a different quantity than the
    experiment ranks policies by.
    """

    users, items, interactions = _frames()
    split = SourcePartitionSplitter().split(interactions)
    pool_by_user = split.test.groupby("user_id")["item_id"].agg(list).to_dict()
    val_targets = {
        (r.user_id, r.item_id): r.liked for r in split.test.itertuples(index=False)
    }
    policy = PopularityPolicy(k=K, seed=0, positive_feedback_column="liked")
    policy.fit(split.train, items=items)

    recs, _ = _slate_frame(
        policy,
        eval_users=sorted(pool_by_user),
        pool_by_user=pool_by_user,
        val_targets=val_targets,
        fit_interactions=split.train,
        user_column="user_id",
        item_column="item_id",
    )
    pooled = recs["target"].mean()
    macro_by_user = recs.groupby("user_id")["target"].mean().mean()
    assert pooled == pytest.approx(macro_by_user)
    assert (recs.groupby("user_id").size() == K).all()


def test_missing_label_raises_instead_of_scoring_zero():
    """The failure that `policy_tuning._evaluate_once` cannot detect.

    Its `val_targets.get(key, 0)` turns an unlabelled candidate into a negative.
    Under free retrieval on Coat that fabricates ~98% of the objective, so this
    protocol must refuse the join rather than fill it.
    """

    users, items, interactions = _frames()
    split = SourcePartitionSplitter().split(interactions)
    pool_by_user = split.test.groupby("user_id")["item_id"].agg(list).to_dict()
    val_targets = {
        (r.user_id, r.item_id): r.liked for r in split.test.itertuples(index=False)
    }
    dropped = next(iter(val_targets))
    del val_targets[dropped]

    policy = PopularityPolicy(k=K, seed=0, positive_feedback_column="liked")
    policy.fit(split.train, items=items)
    with pytest.raises(ValueError, match="fully labelled"):
        _slate_frame(
            policy,
            eval_users=sorted(pool_by_user),
            pool_by_user=pool_by_user,
            val_targets=val_targets,
            fit_interactions=split.train,
            user_column="user_id",
            item_column="item_id",
        )


def test_non_constant_slate_size_raises():
    """Guards the identity asserted in `test_objective_equals_...`."""

    users, items, interactions = _frames()
    split = SourcePartitionSplitter().split(interactions)
    pool_by_user = split.test.groupby("user_id")["item_id"].agg(list).to_dict()
    starved = sorted(pool_by_user)[0]
    pool_by_user[starved] = pool_by_user[starved][: K - 1]
    val_targets = {
        (r.user_id, r.item_id): r.liked for r in split.test.itertuples(index=False)
    }
    policy = PopularityPolicy(k=K, seed=0, positive_feedback_column="liked")
    policy.fit(split.train, items=items)
    with pytest.raises(ValueError, match="slate sizes are not constant"):
        _slate_frame(
            policy,
            eval_users=sorted(pool_by_user),
            pool_by_user=pool_by_user,
            val_targets=val_targets,
            fit_interactions=split.train,
            user_column="user_id",
            item_column="item_id",
        )


# --------------------------------------------------------------------------
# Artifact and leakage bookkeeping
# --------------------------------------------------------------------------


def test_artifact_records_the_protocol_and_the_leakage():
    artifact = _tune()
    identity = artifact["identity"]
    assert identity["protocol"] == PROTOCOL
    assert identity["objective"] == OBJECTIVE_METRIC
    assert identity["policy_k"] == K
    # The caveat has to be machine-readable: `load_policy_roster` never
    # validates `tuned_from`, so a config entry can outlive its artifact.
    assert identity["leakage"]["overlaps_eval_users"] is True
    assert identity["leakage"]["validation_source"] == "mcar_eval_pool"
    assert identity["leakage"]["n_tuning_users"] == N_USERS
    assert identity["fit_frame"]["n_rows"] == len(
        SourcePartitionSplitter().split(_frames()[2]).train
    )
    assert set(artifact["policies"]) == {cls.__name__ for cls in POLICY_ROSTER}


def test_full_fraction_tunes_on_every_user_and_skips_the_diagnostic():
    artifact = _tune()
    assert artifact["curse_summary"] is None
    assert artifact["identity"]["leakage"]["n_holdout_users"] == 0
    assert all("curse_diagnostic" not in e for e in artifact["policies"].values())


def test_untunable_policies_report_exactly_zero_curse():
    """A one-point grid selects nothing, so it cannot be inflated.

    This is the self-consistency check on the estimator: the control is the
    policy's own grid mean, which for a single point is the point itself.
    """

    artifact = _tune(tuning_user_fraction=0.5)
    summary = artifact["curse_summary"]
    assert summary is not None
    for name in ("RandomPolicy", "PopularityPolicy"):
        diagnostic = artifact["policies"][name]["curse_diagnostic"]
        assert diagnostic["n_grid_points"] == 1
        assert diagnostic["excess_over_grid"] == pytest.approx(0.0, abs=1e-12)
        assert diagnostic["excess_ci95"] == pytest.approx([0.0, 0.0], abs=1e-12)


def test_tuning_users_are_seeded_disjoint_and_reproducible():
    everyone = [f"u{i}" for i in range(N_USERS)]
    first = select_tuning_users(everyone, fraction=0.5, split_seed=0)
    again = select_tuning_users(everyone, fraction=0.5, split_seed=0)
    other_seed = select_tuning_users(everyone, fraction=0.5, split_seed=7)
    assert first == again
    assert len(first) == N_USERS // 2
    assert set(first) < set(everyone)
    assert first != other_seed
    assert select_tuning_users(everyone, fraction=1.0, split_seed=0) == sorted(everyone)
    with pytest.raises(ValueError, match="tuning_user_fraction"):
        select_tuning_users(everyone, fraction=0.0, split_seed=0)


def test_artifact_filename_separates_production_from_diagnostic():
    production = artifact_filename("coat", "target_like_ge4", 290, 0)
    diagnostic = artifact_filename("coat", "target_like_ge4", 145, 0)
    assert production != diagnostic
    assert production == "coat_target_like_ge4_rerank_users290_seed0.json"


# --------------------------------------------------------------------------
# Real Coat data. The requirement is "the same train split as the runner", so
# assert it rather than trusting two constants to agree.
# --------------------------------------------------------------------------

COAT_ROOT = REPO_ROOT / "data" / "canonical" / "coat" / "v1"


@pytest.mark.skipif(not COAT_ROOT.exists(), reason="local Coat parquet required")
def test_fit_frame_is_the_rerank_runners_train_split():
    from runners.in_distribution.policy_ranking_agreement_rerank import task_builders

    dataset = task_builders.load_canonical_dataset("coat")
    dataset_filter = MinUserInteractionsFilter(
        min_interactions=task_builders.MIN_INTERACTIONS
    )
    _, _, interactions = dataset_filter.filter(
        users=dataset.load_users(),
        items=dataset.load_items(),
        interactions=dataset.load_interactions(),
    )
    splitter = SourcePartitionSplitter()
    split = splitter.split(interactions)

    assert len(split.train) == 6594
    assert split.train["user_id"].nunique() == 290
    assert splitter.dropped_overlap_rows_ == 366
    # No pool cell may reach the fit frame under any name.
    train_pairs = set(zip(split.train["user_id"], split.train["item_id"], strict=True))
    test_pairs = set(zip(split.test["user_id"], split.test["item_id"], strict=True))
    assert train_pairs.isdisjoint(test_pairs)
    # Every pool is the same size, which is what makes the objective equal the
    # runner's utility.
    assert set(split.test.groupby("user_id").size().unique()) == {16}
    assert task_builders.POLICY_K == 5
