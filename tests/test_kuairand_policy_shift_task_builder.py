from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks import (
    KuaiRandPolicyShiftTaskBuilder,
)


def _canonical_fixture(tmp_path: Path) -> CanonicalDataset:
    root = tmp_path / "kuairand" / "v1"
    root.mkdir(parents=True)

    users = pd.DataFrame({"user_id": ["u1", "u2"]})
    item_ids = [
        "p_u1",
        "n_u1",
        "n_u2",
        *[f"cand_{index:02d}" for index in range(1, 16)],
        "cold_item",
        "no_text",
    ]
    items = pd.DataFrame(
        {
            "item_id": item_ids,
            "description": [
                "" if item_id == "no_text" else f"Description for {item_id}"
                for item_id in item_ids
            ],
            "category_path": [
                "root > test" for _ in item_ids
            ],
        }
    )

    rows: list[dict[str, object]] = []
    sequence = 0

    def add(
        *,
        user: str,
        item: str,
        policy: str,
        date: int,
        click: int,
        tab: int = 1,
        timestamp_offset: int = 0,
    ) -> None:
        nonlocal sequence
        sequence += 1
        rows.append(
            {
                "interaction_id": f"e{sequence:04d}",
                "user_id": user,
                "item_id": item,
                "exposure_policy": policy,
                "tab": tab,
                "date": date,
                "timestamp_ms": date * 100_000 + sequence + timestamp_offset,
                "target_click": click,
            }
        )

    # Balanced tab-1 source histories. u2 makes every candidate item warm and
    # positive under both source policies; both users also have a non-click.
    for policy in ("standard", "random"):
        add(
            user="u1",
            item="p_u1",
            policy=policy,
            date=20220422,
            click=1,
        )
        add(
            user="u1",
            item="n_u1",
            policy=policy,
            date=20220422,
            click=0,
        )
        for index in range(1, 16):
            add(
                user="u2",
                item=f"cand_{index:02d}",
                policy=policy,
                date=20220423,
                click=1,
            )
        add(
            user="u2",
            item="n_u2",
            policy=policy,
            date=20220423,
            click=0,
        )
        # These rows cannot enter the semantic source because one item has no
        # description and the other is not made common-positive.
        add(
            user="u2",
            item="no_text",
            policy=policy,
            date=20220424,
            click=1,
        )

    # Strict novelty is across every tab. This makes u1/cand_02 invalid for
    # validation even though its previous event was not in tab 1.
    add(
        user="u1",
        item="cand_02",
        policy="standard",
        date=20220425,
        click=0,
        tab=2,
    )

    # Validation standard: the later cand_01 duplicate must be ignored and the
    # earlier other-tab cand_02 pair must be excluded.
    add(
        user="u1",
        item="cand_01",
        policy="standard",
        date=20220501,
        click=1,
    )
    add(
        user="u1",
        item="cand_01",
        policy="standard",
        date=20220501,
        click=0,
        timestamp_offset=1_000,
    )
    add(
        user="u1",
        item="cand_02",
        policy="standard",
        date=20220501,
        click=0,
    )
    add(
        user="u1",
        item="cand_03",
        policy="standard",
        date=20220502,
        click=0,
    )
    add(
        user="u1",
        item="cand_13",
        policy="standard",
        date=20220503,
        click=0,
    )
    add(
        user="u1",
        item="cold_item",
        policy="standard",
        date=20220503,
        click=0,
    )

    for item, click in (
        ("cand_04", 1),
        ("cand_05", 0),
        ("cand_06", 0),
        ("cand_14", 0),
        ("no_text", 0),
    ):
        add(
            user="u1",
            item=item,
            policy="random",
            date=20220502,
            click=click,
        )

    # Test standard contains one pair already observed in validation. Pair
    # novelty against every earlier policy must exclude cand_04.
    for item, click in (
        ("cand_07", 1),
        ("cand_08", 0),
        ("cand_09", 0),
        ("cand_15", 0),
        ("cand_04", 0),
    ):
        add(
            user="u1",
            item=item,
            policy="standard",
            date=20220505,
            click=click,
        )

    for item, click in (
        ("cand_10", 1),
        ("cand_11", 0),
        ("cand_12", 0),
        ("cand_13", 0),
    ):
        add(
            user="u1",
            item=item,
            policy="random",
            date=20220506,
            click=click,
        )

    interactions = pd.DataFrame.from_records(rows)
    users_path = root / "users.parquet"
    items_path = root / "items.parquet"
    interactions_path = root / "interactions.parquet"
    manifest_path = root / "manifest.json"
    users.to_parquet(users_path, index=False)
    items.to_parquet(items_path, index=False)
    interactions.to_parquet(interactions_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "kuairand",
                "version": "v1",
                "raw_sources": [
                    {
                        "path": "standard.csv",
                        "sha256": "standard-fixture-sha256",
                    },
                    {
                        "path": "random.csv",
                        "sha256": "random-fixture-sha256",
                    },
                ],
                "semantic_supplement": {
                    "captions_sha256": "captions-fixture-sha256",
                    "categories_sha256": "categories-fixture-sha256",
                },
            }
        ),
        encoding="utf-8",
    )
    return CanonicalDataset(
        name="kuairand",
        version="v1",
        root=root,
        users_path=users_path,
        items_path=items_path,
        interactions_path=interactions_path,
        manifest_path=manifest_path,
    )


def test_builds_locked_full_square_with_observed_c3_targets(
    tmp_path: Path,
) -> None:
    dataset = _canonical_fixture(tmp_path)
    task = KuaiRandPolicyShiftTaskBuilder(seed=42).build(dataset)

    assert task.manifest["scenario"] == {"column": "tab", "value": 1}
    assert task.manifest["windows"] == {
        "source": ["2022-04-22", "2022-04-30"],
        "val": ["2022-05-01", "2022-05-03"],
        "test": ["2022-05-04", "2022-05-08"],
    }
    assert task.manifest["candidate_construction"][
        "kind"
    ] == "observed_c3_pseudo_list"
    assert task.manifest["candidate_construction"][
        "missing_as_negative"
    ] is False
    assert task.manifest["pair_novelty"]["required"] is True
    assert (
        task.manifest["support"]["candidate_common_positive_items"] == 16
    )
    assert task.manifest["support"]["balanced_source_users"] == 2
    assert task.manifest["support"]["target_eligible_users"] == 2
    assert task.manifest["history"]["target_user_filter_applied"] is False
    for policy in ("standard", "random"):
        degree_audit = task.manifest["history"]["source_degree_audit"][policy]
        assert degree_audit["source_exposures"] == len(
            getattr(task, f"{policy}_history")
        )
        assert degree_audit["items_with_positive_support"] == 16
        permutation_audit = task.manifest["history"]["permuted_control"][
            "assignment_change_audit"
        ][policy]
        assert permutation_audit["rows"] == len(
            getattr(task, f"{policy}_history")
        )
        assert permutation_audit["mixed_label_users"] == 2
        assert permutation_audit["changed_rows"] > 0
        assert 0.0 < permutation_audit["changed_fraction"] <= 1.0
        assert (
            permutation_audit["changed_mixed_label_rows"]
            == permutation_audit["changed_rows"]
        )
    assert task.manifest["source_seed"] == 42
    assert task.manifest["candidate_seed"] == 42
    assert task.manifest["min_positive_source_users_per_policy"] == 1
    assert (
        task.manifest["support"][
            "candidate_min_positive_user_degree_per_policy"
        ]
        == 1
    )
    assert task.manifest["history"]["max_items_per_user_per_policy"] == 20
    assert (
        task.manifest["history"]["balance"][
            "max_history_items_per_policy"
        ]
        == 20
    )
    assert "hcap20" in task.name
    assert task.manifest["metadata_visibility"]["hidden_from_llm"] == [
        "user_id",
        "item_id",
        "candidate_group",
        "history_policy",
        "target_policy",
        "event_date",
        "event_timestamp_ms",
        "interaction_id",
        "target",
    ]

    standard_counts = (
        task.standard_history.groupby("user_id").size().sort_index()
    )
    random_counts = task.random_history.groupby("user_id").size().sort_index()
    pd.testing.assert_series_equal(standard_counts, random_counts)
    assert task.standard_history["history_click"].notna().all()
    assert task.random_history["history_click"].notna().all()
    assert task.standard_history["item_description"].str.len().gt(0).all()
    assert task.standard_history["item_category_path"].str.len().gt(0).all()

    for split, standard, random in (
        ("val", task.val_standard_targets, task.val_random_targets),
        ("test", task.test_standard_targets, task.test_random_targets),
    ):
        assert set(standard["user_id"]) == {"u1"}
        assert set(random["user_id"]) == {"u1"}
        for policy, frame in (("standard", standard), ("random", random)):
            assert len(frame) == 3
            assert frame["candidate_group"].nunique() == 1
            assert int(frame["target"].sum()) == 1
            assert frame["history_click"].isna().all()
            assert frame["target_policy"].eq(policy).all()
            assert frame["item_description"].str.len().gt(0).all()
            assert frame["item_category_path"].str.len().gt(0).all()
        assert (
            task.manifest["shared_target_users"][split]["count"] == 1
        )

    # Pair novelty checks all prior tabs and both policies.
    assert "cand_02" not in set(task.val_standard_targets["item_id"])
    assert "cand_04" not in set(task.test_standard_targets["item_id"])
    # Semantically incomplete and non-warm candidates cannot enter targets.
    for frame in (
        task.val_standard_targets,
        task.val_random_targets,
        task.test_standard_targets,
        task.test_random_targets,
    ):
        assert "no_text" not in set(frame["item_id"])
        assert "cold_item" not in set(frame["item_id"])


def test_cell_and_generic_task_views_preserve_fixed_targets_and_controls(
    tmp_path: Path,
) -> None:
    task = KuaiRandPolicyShiftTaskBuilder(seed=42).build(
        _canonical_fixture(tmp_path)
    )

    matched = task.cell(
        history_policy="standard",
        target_policy="standard",
        split="test",
    )
    transferred = task.cell(
        history_policy="random",
        target_policy="standard",
        split="test",
    )
    no_history = task.cell(
        history_policy="standard",
        target_policy="standard",
        split="test",
        history_control="none",
    )
    permuted = task.cell(
        history_policy="standard",
        target_policy="standard",
        split="test",
        history_control="permuted",
    )

    assert matched.manifest["matched"] is True
    assert transferred.manifest["matched"] is False
    pd.testing.assert_frame_equal(matched.targets, transferred.targets)
    assert no_history.history.empty
    assert list(no_history.history.columns) == list(
        task.standard_history.columns
    )

    original = task.standard_history
    permuted_history = permuted.history
    assert original[["user_id", "item_id"]].equals(
        permuted_history[["user_id", "item_id"]]
    )
    assert original.groupby("user_id")["history_click"].sum().to_dict() == (
        permuted_history.groupby("user_id")["history_click"].sum().to_dict()
    )
    assert not original["history_click"].equals(
        permuted_history["history_click"]
    )
    assert permuted_history["history_click"].equals(
        permuted_history["target"]
    )

    generic = task.to_task(
        history_policy="random",
        target_policy="standard",
        history_control="permuted",
    )
    pd.testing.assert_frame_equal(
        generic.train,
        task.random_history_permuted,
    )
    pd.testing.assert_frame_equal(
        generic.val,
        task.val_standard_targets,
    )
    pd.testing.assert_frame_equal(
        generic.test,
        task.test_standard_targets,
    )
    assert generic.schema.history_context_columns == ("history_click",)
    assert generic.schema.feature_columns == (
        "item_description",
        "item_category_path",
    )
    assert generic.manifest["view"] == {
        "history_policy": "random",
        "target_policy": "standard",
        "history_control": "permuted",
        "matched": False,
        "history_sha256": generic.manifest["hashes"]["frames"][
            "random_history_permuted"
        ],
        "val_targets_sha256": generic.manifest["hashes"]["frames"][
            "val_standard_targets"
        ],
        "test_targets_sha256": generic.manifest["hashes"]["frames"][
            "test_standard_targets"
        ],
    }


def test_build_is_deterministic_and_records_provenance_hashes(
    tmp_path: Path,
) -> None:
    dataset = _canonical_fixture(tmp_path)
    first = KuaiRandPolicyShiftTaskBuilder(seed=42).build(dataset)
    second = KuaiRandPolicyShiftTaskBuilder(seed=42).build(dataset)

    for attribute in (
        "standard_history",
        "random_history",
        "standard_history_permuted",
        "random_history_permuted",
        "val_standard_targets",
        "val_random_targets",
        "test_standard_targets",
        "test_random_targets",
    ):
        pd.testing.assert_frame_equal(
            getattr(first, attribute),
            getattr(second, attribute),
        )
    assert first.manifest["hashes"] == second.manifest["hashes"]
    assert len(first.manifest["hashes"]["task_identity_sha256"]) == 64
    assert all(
        len(value) == 64
        for value in first.manifest["hashes"]["frames"].values()
    )
    assert first.manifest["canonical_provenance"] == {
        "raw_sources": [
            {
                "path": "standard.csv",
                "sha256": "standard-fixture-sha256",
            },
            {
                "path": "random.csv",
                "sha256": "random-fixture-sha256",
            },
        ],
        "semantic_supplement": {
            "captions_sha256": "captions-fixture-sha256",
            "categories_sha256": "categories-fixture-sha256",
        },
    }


def test_source_history_is_locked_across_candidate_seeds(
    tmp_path: Path,
) -> None:
    dataset = _canonical_fixture(tmp_path)
    first = KuaiRandPolicyShiftTaskBuilder(
        source_seed=42,
        candidate_seed=42,
    ).build(dataset)
    second = KuaiRandPolicyShiftTaskBuilder(
        source_seed=42,
        candidate_seed=43,
    ).build(dataset)

    for attribute in (
        "standard_history",
        "random_history",
        "standard_history_permuted",
        "random_history_permuted",
    ):
        pd.testing.assert_frame_equal(
            getattr(first, attribute),
            getattr(second, attribute),
        )
    assert (
        first.manifest["support"]["target_eligible_user_ids_sha256"]
        == second.manifest["support"]["target_eligible_user_ids_sha256"]
    )
    assert (
        first.manifest["support"][
            "candidate_common_positive_item_ids_sha256"
        ]
        == second.manifest["support"][
            "candidate_common_positive_item_ids_sha256"
        ]
    )
    assert (
        first.manifest["shared_target_users"]
        == second.manifest["shared_target_users"]
    )
    assert first.manifest["source_seed"] == second.manifest["source_seed"] == 42
    assert first.manifest["candidate_seed"] == 42
    assert second.manifest["candidate_seed"] == 43
    target_names = (
        "val_standard_targets",
        "val_random_targets",
        "test_standard_targets",
        "test_random_targets",
    )
    assert any(
        first.manifest["hashes"]["frames"][name]
        != second.manifest["hashes"]["frames"][name]
        for name in target_names
    )


def test_task_identity_ignores_manifest_time_and_absolute_paths(
    tmp_path: Path,
) -> None:
    dataset = _canonical_fixture(tmp_path)
    first = KuaiRandPolicyShiftTaskBuilder().build(dataset)

    manifest = json.loads(dataset.manifest_path.read_text(encoding="utf-8"))
    manifest["created_at"] = "2099-01-01T00:00:00Z"
    manifest["raw_sources"][0]["path"] = "/different/machine/standard.csv"
    dataset.manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    rebuilt = KuaiRandPolicyShiftTaskBuilder().build(dataset)

    assert (
        first.manifest["hashes"]["canonical_manifest_sha256"]
        != rebuilt.manifest["hashes"]["canonical_manifest_sha256"]
    )
    assert (
        first.manifest["hashes"]["canonical_content_sha256"]
        == rebuilt.manifest["hashes"]["canonical_content_sha256"]
    )
    assert (
        first.manifest["hashes"]["task_identity_sha256"]
        == rebuilt.manifest["hashes"]["task_identity_sha256"]
    )


def test_source_balancing_applies_outcome_blind_per_policy_cap() -> None:
    builder = KuaiRandPolicyShiftTaskBuilder(
        source_seed=42,
        candidate_seed=99,
        max_history_items_per_policy=20,
    )

    def policy_rows(policy: str, *, flip_labels: bool) -> pd.DataFrame:
        labels = [index % 2 for index in range(30)]
        if flip_labels:
            labels = [1 - label for label in labels]
        return pd.DataFrame(
            {
                "user_id": ["u1"] * 30,
                "item_id": [f"{policy}_{index:02d}" for index in range(30)],
                # Deliberately opposite to lexical item order: row sampling
                # may use stable item order, but the visible history must be
                # restored to chronological order afterwards.
                "timestamp_ms": list(reversed(range(30))),
                "interaction_id": [
                    f"{policy}_event_{index:02d}" for index in range(30)
                ],
                "target_click": labels,
            }
        )

    original = {
        policy: policy_rows(policy, flip_labels=False)
        for policy in ("standard", "random")
    }
    relabeled = {
        policy: policy_rows(policy, flip_labels=True)
        for policy in ("standard", "random")
    }
    balanced, audit = builder._balance_source(original)
    balanced_relabeled, _ = builder._balance_source(relabeled)

    for policy in ("standard", "random"):
        assert len(balanced[policy]) == 20
        assert set(balanced[policy]["item_id"]) == set(
            balanced_relabeled[policy]["item_id"]
        )
        assert balanced[policy]["timestamp_ms"].tolist() == sorted(
            balanced[policy]["timestamp_ms"].tolist()
        )
    assert audit["per_user_count_max"] == 20
    assert audit["max_history_items_per_policy"] == 20


def test_full_balanced_source_is_not_filtered_to_target_eligible_users(
    tmp_path: Path,
) -> None:
    dataset = _canonical_fixture(tmp_path)
    interactions = pd.read_parquet(dataset.interactions_path)
    # u2 now has only positive source labels under each policy. It therefore
    # cannot be an LLM target user, but its candidate-item observations remain
    # valid outcome-blind evidence for aggregate/CF source training.
    interactions.loc[
        interactions["item_id"].eq("n_u2")
        & interactions["date"].between(20220422, 20220430),
        "target_click",
    ] = 1
    interactions.to_parquet(dataset.interactions_path, index=False)

    task = KuaiRandPolicyShiftTaskBuilder(seed=42).build(dataset)

    assert set(task.standard_history["user_id"]) == {"u1", "u2"}
    assert set(task.random_history["user_id"]) == {"u1", "u2"}
    assert task.manifest["support"]["balanced_source_users"] == 2
    assert task.manifest["support"]["target_eligible_users"] == 1
    assert set(task.test_standard_targets["user_id"]) == {"u1"}
    assert set(task.test_random_targets["user_id"]) == {"u1"}
    assert "cand_01" in set(task.standard_history["item_id"])
    assert "cand_01" in set(task.random_history["item_id"])
    assert task.manifest["support"]["candidate_common_positive_items"] == 17


def test_min_two_positive_source_users_is_predeclared_robustness(
    tmp_path: Path,
) -> None:
    dataset = _canonical_fixture(tmp_path)
    users = pd.read_parquet(dataset.users_path)
    users = pd.concat(
        [users, pd.DataFrame({"user_id": ["u3"]})],
        ignore_index=True,
    )
    users.to_parquet(dataset.users_path, index=False)

    interactions = pd.read_parquet(dataset.interactions_path)
    support_rows: list[dict[str, object]] = []
    sequence = 0
    for policy in ("standard", "random"):
        for index in range(1, 16):
            sequence += 1
            support_rows.append(
                {
                    "interaction_id": f"u3_support_{sequence:03d}",
                    "user_id": "u3",
                    "item_id": f"cand_{index:02d}",
                    "exposure_policy": policy,
                    "tab": 1,
                    "date": 20220424,
                    "timestamp_ms": 20220424 * 100_000 + sequence,
                    "target_click": 1,
                }
            )
    interactions = pd.concat(
        [interactions, pd.DataFrame.from_records(support_rows)],
        ignore_index=True,
    )
    interactions.to_parquet(dataset.interactions_path, index=False)

    primary = KuaiRandPolicyShiftTaskBuilder(
        source_seed=42,
        candidate_seed=42,
        min_positive_source_users_per_policy=1,
    ).build(dataset)
    robust = KuaiRandPolicyShiftTaskBuilder(
        source_seed=42,
        candidate_seed=42,
        min_positive_source_users_per_policy=2,
    ).build(dataset)

    assert "posdeg2" in robust.name
    assert robust.manifest["min_positive_source_users_per_policy"] == 2
    assert (
        robust.manifest["support"][
            "candidate_min_positive_user_degree_per_policy"
        ]
        == 2
    )
    assert robust.manifest["support"]["candidate_common_positive_items"] == 15
    assert set(robust.test_standard_targets["user_id"]) == {"u1"}
    assert set(robust.test_random_targets["user_id"]) == {"u1"}
    # Raising target-item support must not silently change the source graph.
    pd.testing.assert_frame_equal(
        primary.standard_history,
        robust.standard_history,
    )
    pd.testing.assert_frame_equal(
        primary.random_history,
        robust.random_history,
    )
    assert (
        primary.manifest["hashes"]["task_identity_sha256"]
        != robust.manifest["hashes"]["task_identity_sha256"]
    )


def test_cell_rejects_unknown_conditions(tmp_path: Path) -> None:
    task = KuaiRandPolicyShiftTaskBuilder().build(
        _canonical_fixture(tmp_path)
    )

    with pytest.raises(ValueError, match="history_policy"):
        task.cell(
            history_policy="other",  # type: ignore[arg-type]
            target_policy="standard",
            split="test",
        )
    with pytest.raises(ValueError, match="history_control"):
        task.cell(
            history_policy="standard",
            target_policy="standard",
            split="test",
            history_control="shuffled",  # type: ignore[arg-type]
        )
