from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import beyond_click_sim.data.kuairand_semantics as semantics_module
from beyond_click_sim.data.canonical import CanonicalDataset, sha256_file
from beyond_click_sim.tasks import KuaiRandStandardToRandomTaskBuilder


def test_builds_standard_source_task_with_strict_novelty_and_natural_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, translation_dir = _dataset(tmp_path)
    _pin_fixture_output(monkeypatch, translation_dir)
    task = _builder(translation_dir).build(dataset)

    assert task.manifest["windows"] == {
        "source": ["2022-04-08", "2022-04-21"],
        "buffer": ["2022-04-22", "2022-04-22"],
        "val": ["2022-04-23", "2022-04-30"],
        "test": ["2022-05-01", "2022-05-08"],
    }
    assert task.manifest["protocol"] == "kuairand-standard-to-random-v2"
    assert task.manifest["candidate_construction"]["labels_balanced"] is False
    assert task.manifest["cohorts"]["selection_uses_target_labels"] is False
    assert task.training_source["target_policy"].isna().all()
    assert len(task.training_source) > len(
        task.training_source.drop_duplicates(["user_id", "item_id"])
    )
    assert task.visible_history.groupby("user_id").size().to_dict() == {
        "test": 2,
        "val": 2,
    }
    latest_val_h1 = task.visible_history.loc[
        task.visible_history["user_id"].eq("val")
        & task.visible_history["item_id"].eq("h1")
    ].iloc[0]
    assert latest_val_h1["interaction_id"] == "source-val-h1-latest"

    val_users = set(task.val_standard_targets["user_id"])
    test_users = set(task.test_standard_targets["user_id"])
    assert val_users == {"val"}
    assert test_users == {"test"}
    assert not val_users & test_users
    for frame in (
        task.val_standard_targets,
        task.val_random_targets,
        task.test_standard_targets,
        task.test_random_targets,
    ):
        assert frame.groupby("candidate_group").size().eq(2).all()
        assert frame["history_click"].isna().all()
        assert frame["source_exposure_count"].ge(2).all()
        assert frame["source_distinct_user_count"].ge(2).all()
        assert frame["item_caption_en"].str.startswith("English").all()

    # April 22 in another tab invalidates a validation edge.
    assert "v_s3" not in set(task.val_standard_targets["item_id"])
    # A validation-window exposure invalidates a later test edge.
    assert "t_r3" not in set(task.test_random_targets["item_id"])
    # The same-window standard exposure invalidates the later random exposure.
    assert "cross" not in set(task.test_random_targets["item_id"])

    # Degenerate groups are materialized and left for metric-time filtering.
    assert task.manifest["targets"]["val_random"]["all_zero_groups"] == 1
    assert task.manifest["targets"]["test_standard"]["all_positive_groups"] == 1

    observed = task.cell(split="test", target_policy="random")
    no_history = task.cell(
        split="test",
        target_policy="random",
        history_control="none",
    )
    assert set(observed.history["user_id"]) == {"test"}
    assert no_history.history.empty
    pd.testing.assert_frame_equal(
        observed.training_source,
        no_history.training_source,
    )


def test_structural_warm_gate_keeps_zero_positive_item_and_full_pool_is_cold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, translation_dir = _dataset(tmp_path)
    _pin_fixture_output(monkeypatch, translation_dir)
    warm = _builder(translation_dir, target_pool="common_warm").build(dataset)
    full = _builder(translation_dir, target_pool="full_pool").build(dataset)

    assert warm.manifest["support"]["positive_source_user_degree_is_audit_only"]
    assert warm.manifest["support"]["common_warm_items"] == 13
    zero_positive = warm.val_standard_targets.loc[
        warm.val_standard_targets["item_id"].eq("v_s1")
    ]
    assert len(zero_positive) == 1
    assert zero_positive["source_positive_user_count"].eq(0).all()
    assert zero_positive["source_positive_graph_cold"].all()

    assert (
        warm.manifest["cohorts"]["validation"]["user_ids_sha256"]
        == full.manifest["cohorts"]["validation"]["user_ids_sha256"]
    )
    assert (
        warm.manifest["cohorts"]["test"]["user_ids_sha256"]
        == full.manifest["cohorts"]["test"]["user_ids_sha256"]
    )
    full_rows = pd.concat(
        [
            full.val_standard_targets,
            full.val_random_targets,
            full.test_standard_targets,
            full.test_random_targets,
        ]
    )
    assert full_rows["source_exposure_cold"].any()
    assert "0" in set(full_rows["source_exposure_degree_stratum"])


def test_candidate_and_cohort_selection_are_label_blind_and_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, translation_dir = _dataset(tmp_path)
    _pin_fixture_output(monkeypatch, translation_dir)
    builder = _builder(translation_dir, candidate_seed=0)
    first = builder.build(dataset)
    second = builder.build(dataset)
    for attribute in _target_attributes():
        pd.testing.assert_frame_equal(
            getattr(first, attribute),
            getattr(second, attribute),
        )

    interactions = pd.read_parquet(dataset.interactions_path)
    target = interactions["date"].ge(20220423)
    interactions.loc[target, "target_click"] = (
        1 - interactions.loc[target, "target_click"]
    )
    interactions.to_parquet(dataset.interactions_path, index=False)
    flipped = builder.build(dataset)
    identity = ["user_id", "item_id", "interaction_id", "candidate_group"]
    for attribute in _target_attributes():
        assert getattr(first, attribute)[identity].equals(
            getattr(flipped, attribute)[identity]
        )
    assert (
        first.manifest["cohorts"]["validation"]["user_ids_sha256"]
        == flipped.manifest["cohorts"]["validation"]["user_ids_sha256"]
    )
    assert (
        first.manifest["cohorts"]["test"]["user_ids_sha256"]
        == flipped.manifest["cohorts"]["test"]["user_ids_sha256"]
    )


def _target_attributes() -> tuple[str, ...]:
    return (
        "val_standard_targets",
        "val_random_targets",
        "test_standard_targets",
        "test_random_targets",
    )


def _pin_fixture_output(
    monkeypatch: pytest.MonkeyPatch,
    translation_dir: Path,
) -> None:
    monkeypatch.setattr(
        semantics_module,
        "PRIMARY_TRANSLATED_ITEMS_SHA256",
        sha256_file(translation_dir / "kuairand_items_en.csv"),
    )


def _builder(
    translation_dir: Path,
    *,
    target_pool: str = "common_warm",
    candidate_seed: int = 0,
) -> KuaiRandStandardToRandomTaskBuilder:
    return KuaiRandStandardToRandomTaskBuilder(
        translation_dir=translation_dir,
        target_pool=target_pool,  # type: ignore[arg-type]
        candidate_seed=candidate_seed,
        cohort_seed=0,
        history_size=2,
        candidate_size=2,
        validation_users=1,
        test_users=1,
        min_source_exposures=2,
        min_source_users=2,
    )


def _dataset(tmp_path: Path) -> tuple[CanonicalDataset, Path]:
    root = tmp_path / "canonical" / "kuairand-pure" / "v1"
    root.mkdir(parents=True)
    users = pd.DataFrame(
        {"user_id": ["val", "test", "trainer-a", "trainer-b"]}
    )
    warm_items = [
        "h1",
        "h2",
        "v_s1",
        "v_s2",
        "v_s3",
        "v_r1",
        "v_r2",
        "t_s1",
        "t_s2",
        "t_r1",
        "t_r2",
        "t_r3",
        "cross",
        "low_users",
        "low_exposures",
    ]
    cold_items = [f"cold-{index}" for index in range(8)]
    item_ids = warm_items + cold_items
    items = pd.DataFrame(
        {
            "item_id": item_ids,
            "caption": [f"中文 {item_id}" for item_id in item_ids],
            "show_cover_text": [f"封面 {item_id}" for item_id in item_ids],
            "category_path": ["视频 > 测试"] * len(item_ids),
        }
    )

    rows: list[dict[str, object]] = []
    sequence = 0

    def add(
        interaction_id: str,
        user_id: str,
        item_id: str,
        policy: str,
        date: int,
        click: int,
        *,
        tab: int = 1,
        timestamp_offset: int = 0,
    ) -> None:
        nonlocal sequence
        sequence += 1
        rows.append(
            {
                "interaction_id": interaction_id,
                "user_id": user_id,
                "item_id": item_id,
                "exposure_policy": policy,
                "tab": tab,
                "date": date,
                "timestamp_ms": date * 100_000 + sequence + timestamp_offset,
                "target_click": click,
            }
        )

    # Every item except the explicitly thin/cold items is structurally warm.
    for item_id in warm_items:
        if item_id in {"low_users", "low_exposures"}:
            continue
        add(f"source-a-{item_id}", "trainer-a", item_id, "standard", 20220410, 0)
        positive = 0 if item_id == "v_s1" else 1
        add(
            f"source-b-{item_id}",
            "trainer-b",
            item_id,
            "standard",
            20220411,
            positive,
        )
    add("source-thin-a", "trainer-a", "low_users", "standard", 20220410, 1)
    add(
        "source-thin-a-repeat",
        "trainer-a",
        "low_users",
        "standard",
        20220411,
        0,
    )
    add(
        "source-one-exposure",
        "trainer-a",
        "low_exposures",
        "standard",
        20220410,
        1,
    )

    for user_id in ("val", "test"):
        add(f"source-{user_id}-h1", user_id, "h1", "standard", 20220408, 0)
        add(f"source-{user_id}-h2", user_id, "h2", "standard", 20220409, 1)
    add(
        "source-val-h1-latest",
        "val",
        "h1",
        "standard",
        20220420,
        1,
    )

    # Buffer/other-tab exposure must invalidate the later target pair.
    add("buffer-v-s3", "val", "v_s3", "random", 20220422, 0, tab=2)

    for item_id, click in (("v_s1", 1), ("v_s2", 0), ("v_s3", 0)):
        add(f"val-standard-{item_id}", "val", item_id, "standard", 20220424, click)
    for item_id in ("v_r1", "v_r2"):
        add(f"val-random-{item_id}", "val", item_id, "random", 20220425, 0)
    for item_id in cold_items:
        add(f"val-random-{item_id}", "val", item_id, "random", 20220426, 0)

    # This validation-window exposure invalidates t_r3 in test.
    add("val-prior-t-r3", "test", "t_r3", "standard", 20220427, 0, tab=2)

    for item_id in ("t_s1", "t_s2", "cross"):
        add(f"test-standard-{item_id}", "test", item_id, "standard", 20220502, 1)
    for item_id, click in (("t_r1", 1), ("t_r2", 0), ("t_r3", 0)):
        add(f"test-random-{item_id}", "test", item_id, "random", 20220503, click)
    add(
        "test-random-cross",
        "test",
        "cross",
        "random",
        20220504,
        0,
    )
    for item_id in cold_items:
        add(f"test-random-{item_id}", "test", item_id, "random", 20220505, 0)

    paths = {
        "users": root / "users.parquet",
        "items": root / "items.parquet",
        "interactions": root / "interactions.parquet",
        "manifest": root / "manifest.json",
    }
    users.to_parquet(paths["users"], index=False)
    items.to_parquet(paths["items"], index=False)
    pd.DataFrame(rows).to_parquet(paths["interactions"], index=False)
    paths["manifest"].write_text(
        json.dumps(
            {
                "dataset": "kuairand-pure",
                "version": "v1",
                "schema_version": "fixture",
                "raw_sources": [{"sha256": "fixture"}],
            }
        ),
        encoding="utf-8",
    )
    translation_dir = tmp_path / "translation"
    translation_dir.mkdir()
    output = translation_dir / "kuairand_items_en.csv"
    pd.DataFrame(
        {
            "item_id": item_ids,
            "caption_zh": items["caption"],
            "cover_text_zh": items["show_cover_text"],
            "category_path_zh": items["category_path"],
            "caption_en": [f"English caption {item_id}" for item_id in item_ids],
            "cover_text_en": [f"English cover {item_id}" for item_id in item_ids],
            "category_path_en": ["Video > Test"] * len(item_ids),
        }
    ).to_csv(output, index=False)
    (translation_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "kuairand-pure",
                "model": "gpt-5.5",
                "selection": {"items": len(item_ids), "partial": False},
                "source": {
                    "items_parquet_sha256": sha256_file(paths["items"])
                },
                "outputs": {
                    "items": output.name,
                    "items_sha256": sha256_file(output),
                },
            }
        ),
        encoding="utf-8",
    )
    dataset = CanonicalDataset(
        name="kuairand-pure",
        version="v1",
        root=root,
        users_path=paths["users"],
        items_path=paths["items"],
        interactions_path=paths["interactions"],
        manifest_path=paths["manifest"],
    )
    return dataset, translation_dir
