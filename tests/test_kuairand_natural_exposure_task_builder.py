from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks import KuaiRandNaturalExposureTaskBuilder
from runners.ood.kuairand_policy_shift.methods import METHOD_REGISTRY
from runners.ood.kuairand_policy_shift.metrics import evaluate_prediction_cells
from runners.ood.kuairand_policy_shift.run import _history_for_method
from runners.ood.kuairand_policy_shift.run import run_method_suite


def test_builds_label_blind_natural_full_square(tmp_path: Path) -> None:
    task = KuaiRandNaturalExposureTaskBuilder(
        history_size=2,
        candidate_size=2,
        min_exposure_users_per_policy=1,
        candidate_seed=0,
    ).build(_dataset(tmp_path))

    assert task.manifest["protocol"] == "natural-exposure-policy-shift-v1"
    construction = task.manifest["candidate_construction"]
    assert construction["kind"] == "label_blind_natural_exposure_group"
    assert construction["target_user_selection_uses_labels"] is False
    assert construction["candidate_selection_uses_labels"] is False
    assert task.manifest["support"]["uses_click_labels"] is False

    for policy in ("standard", "random"):
        visible = getattr(task, f"{policy}_history")
        assert visible.groupby("user_id").size().eq(2).all()
        assert set(visible["user_id"]) == {"u1", "u2"}

        training = getattr(task, f"{policy}_training_history")
        evaluation_rows = training.loc[
            training["user_id"].isin({"u1", "u2"})
        ]
        assert evaluation_rows.groupby("user_id").size().eq(2).all()
        assert "trainer" in set(training["user_id"])

    generic = task.to_task(
        history_policy="standard",
        target_policy="random",
    )
    assert "trainer" in set(generic.train["user_id"])
    assert generic.train.loc[
        generic.train["user_id"].isin({"u1", "u2"})
    ].groupby("user_id").size().eq(2).all()

    for split, standard, random in (
        ("val", task.val_standard_targets, task.val_random_targets),
        ("test", task.test_standard_targets, task.test_random_targets),
    ):
        for frame in (standard, random):
            assert set(frame["user_id"]) == {"u1", "u2"}
            assert frame.groupby("candidate_group").size().eq(2).all()
            assert frame["minimum_source_exposure_user_degree"].ge(1).all()
            assert frame["minimum_training_exposure_user_degree"].ge(1).all()
            assert set(frame["source_positive_support_stratum"]).issubset(
                {"zero", "one", "two_to_four", "five_or_more"}
            )
        matched = task.cell(
            history_policy="standard",
            target_policy="standard",
            split=split,
        )
        transferred = task.cell(
            history_policy="random",
            target_policy="standard",
            split=split,
        )
        pd.testing.assert_frame_equal(matched.targets, transferred.targets)

    # Earlier exposure from another tab invalidates this exact edge.
    assert not (
        task.val_standard_targets["user_id"].eq("u1")
        & task.val_standard_targets["item_id"].eq("val_s_1")
    ).any()
    # An all-zero natural group remains in the task.
    u2_random = task.test_random_targets.loc[
        task.test_random_targets["user_id"].eq("u2")
    ]
    assert int(u2_random["target"].sum()) == 0


def test_natural_training_control_replaces_only_evaluation_rows(
    tmp_path: Path,
) -> None:
    task = KuaiRandNaturalExposureTaskBuilder(
        history_size=2,
        candidate_size=2,
        min_exposure_users_per_policy=1,
    ).build(_dataset(tmp_path))

    observed = task.cell(
        history_policy="standard",
        target_policy="random",
        split="test",
        history_control="observed",
    )
    permuted = task.cell(
        history_policy="standard",
        target_policy="random",
        split="test",
        history_control="permuted",
    )
    observed_trainer = observed.training_history.loc[
        observed.training_history["user_id"].eq("trainer")
    ]
    permuted_trainer = permuted.training_history.loc[
        permuted.training_history["user_id"].eq("trainer")
    ]
    pd.testing.assert_frame_equal(
        observed_trainer.reset_index(drop=True),
        permuted_trainer.reset_index(drop=True),
    )
    observed_eval = observed.training_history.loc[
        observed.training_history["user_id"].isin({"u1", "u2"})
    ]
    permuted_eval = permuted.training_history.loc[
        permuted.training_history["user_id"].isin({"u1", "u2"})
    ]
    assert observed_eval[["user_id", "item_id"]].reset_index(drop=True).equals(
        permuted_eval[["user_id", "item_id"]].reset_index(drop=True)
    )
    assert observed_eval.groupby("user_id")["history_click"].sum().to_dict() == (
        permuted_eval.groupby("user_id")["history_click"].sum().to_dict()
    )

    llm_history = _history_for_method(
        observed.history,
        METHOD_REGISTRY["llm_openai_vk_gpt54_mini_caption_categories"],
        ["u1", "u2"],
        training_history=observed.training_history,
    )
    item_history = _history_for_method(
        observed.history,
        METHOD_REGISTRY["item_ctr"],
        ["u1", "u2"],
        training_history=observed.training_history,
    )
    assert set(llm_history["user_id"]) == {"u1", "u2"}
    assert "trainer" in set(item_history["user_id"])


def test_target_membership_does_not_change_when_clicks_flip(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    builder = KuaiRandNaturalExposureTaskBuilder(
        history_size=2,
        candidate_size=2,
        min_exposure_users_per_policy=1,
        candidate_seed=2,
    )
    original = builder.build(dataset)
    interactions = pd.read_parquet(dataset.interactions_path)
    target_mask = interactions["date"].ge(20220501)
    interactions.loc[target_mask, "target_click"] = (
        1 - interactions.loc[target_mask, "target_click"]
    )
    interactions.to_parquet(dataset.interactions_path, index=False)
    flipped = builder.build(dataset)

    identity = ["user_id", "item_id", "interaction_id", "candidate_group"]
    for attribute in (
        "val_standard_targets",
        "val_random_targets",
        "test_standard_targets",
        "test_random_targets",
    ):
        assert getattr(original, attribute)[identity].equals(
            getattr(flipped, attribute)[identity]
        )


def test_natural_metrics_keep_all_groups_but_rank_only_mixed_groups() -> None:
    rows = []
    for user_id, labels in (
        ("mixed", (1, 0, 0, 1, 0)),
        ("all-zero", (0, 0, 0, 0, 0)),
    ):
        for position, label in enumerate(labels):
            rows.append(
                {
                    "task_seed": 0,
                    "user_id": user_id,
                    "candidate_group": f"test-standard-{user_id}",
                    "history_policy": "standard",
                    "target_policy": "standard",
                    "history_control": "observed",
                    "metadata_condition": "not_applicable",
                    "matched": True,
                    "target": label,
                    "score": float(len(labels) - position),
                }
            )
    cells, per_user = evaluate_prediction_cells(
        pd.DataFrame(rows),
        ranking_k=5,
        require_mixed_labels=True,
    )

    assert len(cells) == 1
    assert cells[0]["n_users"] == 2
    assert cells[0]["n_ranking_eligible_users"] == 1
    assert cells[0]["ranking_eligible_rate"] == 0.5
    assert cells[0]["n_scored_users"] == 1
    assert "ndcg@5" in cells[0]
    all_zero = per_user.loc[per_user["user_id"].eq("all-zero")].iloc[0]
    assert bool(all_zero["ranking_eligible"]) is False
    assert pd.isna(all_zero["ndcg@5"])


def test_natural_runner_reports_ndcg5_and_mixed_coverage(
    tmp_path: Path,
) -> None:
    task = KuaiRandNaturalExposureTaskBuilder(
        history_size=2,
        candidate_size=2,
        min_exposure_users_per_policy=1,
        candidate_seed=0,
    ).build(_dataset(tmp_path / "data"))
    result = run_method_suite(
        {0: task},
        method_name="global_mean",
        output_dir=tmp_path / "run",
        user_ids=["u1", "u2"],
        controls=["observed"],
    )

    assert len(result["cells"]) == 4
    assert all("ndcg@2" in cell for cell in result["cells"])
    assert all("ranking_eligible_rate" in cell for cell in result["cells"])
    manifest = json.loads(
        (tmp_path / "run" / "manifest.json").read_text(encoding="utf-8")
    )
    assert (
        manifest["protocol"]
        == "test_only_direct_natural_m2_response_prediction"
    )
    assert manifest["metrics"]["primary"] == ["log_loss", "brier"]
    assert all("log_loss" in cell for cell in result["cells"])


def _dataset(tmp_path: Path) -> CanonicalDataset:
    root = tmp_path / "kuairand-natural" / "v1"
    root.mkdir(parents=True)
    users = pd.DataFrame({"user_id": ["u1", "u2", "trainer"]})
    source_items = ["h1", "h2", "h3"]
    val_items = [
        f"val_{policy}_{index}"
        for policy in ("s", "r")
        for index in range(1, 5)
    ]
    test_items = [
        f"test_{policy}_{index}"
        for policy in ("s", "r")
        for index in range(1, 5)
    ]
    item_ids = source_items + val_items + test_items
    items = pd.DataFrame(
        {
            "item_id": item_ids,
            "description": [f"Caption {item_id}" for item_id in item_ids],
            "category_path": ["root > test"] * len(item_ids),
        }
    )

    rows: list[dict[str, object]] = []
    sequence = 0

    def add(
        user: str,
        item: str,
        policy: str,
        date: int,
        click: int,
        *,
        tab: int = 1,
    ) -> None:
        nonlocal sequence
        sequence += 1
        rows.append(
            {
                "interaction_id": f"e{sequence:05d}",
                "user_id": user,
                "item_id": item,
                "exposure_policy": policy,
                "tab": tab,
                "date": date,
                "timestamp_ms": date * 100_000 + sequence,
                "target_click": click,
            }
        )

    for policy in ("standard", "random"):
        # Evaluation users have three source rows; only the latest two may be
        # visible or retained for them in the collaborative training graph.
        for user, labels in (("u1", (0, 1, 0)), ("u2", (0, 0, 0))):
            for item, click in zip(source_items, labels, strict=True):
                add(user, item, policy, 20220425, click)
        # The non-evaluation user makes every future candidate source-warm.
        for item in item_ids:
            add("trainer", item, policy, 20220426, int(item.endswith("2")))

    # Novelty must inspect all tabs, not only target tab 1.
    add("u1", "val_s_1", "random", 20220430, 0, tab=0)

    for user in ("u1", "u2"):
        for policy, short in (("standard", "s"), ("random", "r")):
            for index in range(1, 5):
                add(
                    user,
                    f"val_{short}_{index}",
                    policy,
                    20220502,
                    int(index == 2),
                )
            for index in range(1, 5):
                click = int(index in (1, 3))
                if user == "u2" and policy == "random":
                    click = 0
                add(
                    user,
                    f"test_{short}_{index}",
                    policy,
                    20220505,
                    click,
                )

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
                "dataset": "kuairand",
                "version": "v1",
                "raw_sources": [{"sha256": "fixture"}],
                "semantic_supplement": {"sha256": "semantic-fixture"},
            }
        ),
        encoding="utf-8",
    )
    return CanonicalDataset(
        name="kuairand",
        version="v1",
        root=root,
        users_path=paths["users"],
        items_path=paths["items"],
        interactions_path=paths["interactions"],
        manifest_path=paths["manifest"],
    )
