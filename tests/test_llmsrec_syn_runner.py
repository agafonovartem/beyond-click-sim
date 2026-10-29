from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import random

import pandas as pd
import pytest

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks import (
    AlignmentInteractionTaskBuilder,
    CappedUserInteractionCandidateSampler,
    MinUserInteractionsFilter,
    RandomFractionSplitter,
    TrainItemRatingStatistics,
)
from runners.in_distribution.interaction_prediction.methods import (
    METHOD_RUNNERS,
    llmsrec_syn,
)
from runners.in_distribution.interaction_prediction.methods._llmsrec_syn_data import (
    prepare_training,
)
from test_llmsrec_syn_scorer import EmbeddingClient, RankingClient


ROOT = Path(__file__).resolve().parents[1]


def build_task(tmp_path, *, dataset_name="ml-1m", event_ids=True, repeated=False):
    root = tmp_path / "canonical"
    root.mkdir(parents=True)
    context = "rating" if dataset_name == "ml-1m" else "playtime_forever"
    rows = [
        {
            "interaction_id": f"{user}:event:{i}",
            "user_id": user,
            "item_id": f"{user}:item:{i if not repeated or i != 5 else 4}",
            "timestamp": i,
            context: i % 5 + 1,
            "target_interact": 1,
        }
        for user in [f"u{i}" for i in range(8)]
        for i in reversed(range(15))
    ]
    items = list(dict.fromkeys(r["item_id"] for r in rows))
    pd.DataFrame({"user_id": sorted({r["user_id"] for r in rows})}).to_parquet(
        root / "users.parquet"
    )
    pd.DataFrame(
        {
            "item_id": items,
            "title": [f"Title {i}" for i in items],
            "genres": ["Comedy"] * len(items),
        }
    ).to_parquet(root / "items.parquet")
    interactions = pd.DataFrame(rows)
    if not event_ids:
        interactions = interactions.drop(columns="interaction_id")
    interactions.to_parquet(root / "interactions.parquet")
    (root / "manifest.json").write_text("{}")
    dataset = CanonicalDataset(
        dataset_name,
        "test-v1",
        root,
        root / "users.parquet",
        root / "items.parquet",
        root / "interactions.parquet",
        root / "manifest.json",
    )
    task = AlignmentInteractionTaskBuilder(
        name="temporal-fixture",
        dataset_filter=MinUserInteractionsFilter(10),
        splitter=RandomFractionSplitter(0.7, 0.1, 0.2),
        sampler=CappedUserInteractionCandidateSampler(
            negative_ratio=1, total_items=20, max_eval_users=2, seed=9
        ),
        history_context_columns=(context,),
        item_feature_builder=TrainItemRatingStatistics(value_column=context),
    ).build(dataset)
    return task, dataset


def run(task, dataset, output_dir, monkeypatch, *, chat=None, **kwargs):
    chat, embeddings = chat or RankingClient(), EmbeddingClient()
    monkeypatch.setattr(
        llmsrec_syn,
        "make_llm_client",
        lambda name: chat if name == "fake_llm" else embeddings,
    )
    result = llmsrec_syn.run_method(
        task,
        output_dir,
        method_name="syn-test",
        client_name="fake_llm",
        model="frozen-test",
        embedding_client_name="fake_embedding",
        canonical_dataset=dataset,
        k_values=kwargs.pop("k_values", (1, 2)),
        max_llm_attempts=2,
        **kwargs,
    )
    return result, chat, embeddings


def read_json(path):
    return json.loads(path.read_text())


@pytest.mark.parametrize("dataset_name", ["ml-1m", "steam"])
@pytest.mark.parametrize("event_ids", [True, False])
def test_verified_training_mapping_preserves_repeated_occurrences_and_all_users(
    tmp_path, dataset_name, event_ids
):
    task, dataset = build_task(
        tmp_path, dataset_name=dataset_name, event_ids=event_ids, repeated=True
    )
    before = deepcopy(task)
    prepared = prepare_training(task, dataset)
    assert prepared.events.index.equals(task.train.index)
    assert prepared.events.event_id.is_unique
    assert prepared.events.user_id.nunique() == 8
    assert task.test.user_id.nunique() == 2
    for user, events in prepared.events.groupby("user_id"):
        assert events.timestamp.tolist() == sorted(events.timestamp)
        repeated_events = events[events.item_id.eq(f"{user}:item:4")]
        assert repeated_events.timestamp.tolist() == [4, 5]
    assert (
        prepared.provenance["canonical_file_sha256"]["interactions"]
        == sha256(dataset.interactions_path.read_bytes()).hexdigest()
    )
    for split in ("train", "val", "test"):
        pd.testing.assert_frame_equal(getattr(task, split), getattr(before, split))


@pytest.mark.parametrize(
    "case",
    [
        "train_order",
        "train_value",
        "missing_timestamp",
        "null_timestamp",
        "random",
        "wrong_dataset",
        "unsupported_filter",
        "bad_event_id",
    ],
)
def test_bad_provenance_fails_before_client_construction(tmp_path, monkeypatch, case):
    task, dataset = build_task(tmp_path)
    if case == "train_order":
        task.train = task.train.iloc[::-1]
    elif case == "train_value":
        task.train.iloc[0, task.train.columns.get_loc("rating")] = -1
    elif case in ("missing_timestamp", "null_timestamp", "bad_event_id"):
        events = dataset.load_interactions()
        if case == "missing_timestamp":
            events = events.drop(columns="timestamp")
        elif case == "null_timestamp":
            events["timestamp"] = events.timestamp.astype(float)
            events.loc[0, "timestamp"] = float("nan")
        else:
            events.loc[0, "interaction_id"] = events.loc[1, "interaction_id"]
        events.to_parquet(dataset.interactions_path)
    elif case == "random":
        task.manifest["splitter"]["shuffle"] = True
    elif case == "wrong_dataset":
        task.manifest["dataset_version"] = "wrong"
    else:
        task.manifest["filter"]["class"] = "Unknown"
    monkeypatch.setattr(
        llmsrec_syn,
        "make_llm_client",
        lambda _: pytest.fail("invalid source must fail before client"),
    )
    with pytest.raises((ValueError, KeyError)):
        llmsrec_syn.run_method(task, tmp_path / "run", canonical_dataset=dataset)


@pytest.mark.parametrize("dataset_name", ["ml-1m", "steam"])
def test_runner_writes_metrics_selection_provenance_and_diagnostics(
    tmp_path, monkeypatch, dataset_name
):
    task, dataset = build_task(tmp_path, dataset_name=dataset_name)
    before = deepcopy(task)
    random_before = random.getstate()
    out = tmp_path / "run"
    result, chat, embedding_client = run(task, dataset, out, monkeypatch, max_workers=2)
    assert random.getstate() == random_before
    selection = read_json(out / "llmsrec_syn_selection.json")
    assert selection["selected_k"] == 1  # same rank output; tie breaks to smaller K
    assert all(row["requested_groups"] == 2 for row in selection["candidates"])
    assert all(row["failed_groups"] == 0 for row in selection["candidates"])
    assert selection["metric"].endswith("failed_group_zero")
    assert "ndcg@5" in selection["metric"]
    manifest = read_json(out / "manifest.json")
    assert manifest["scorer"]["uses_item_stats"] is False
    assert manifest["scorer"]["max_history_items"] == 50
    assert manifest["scorer"]["aggregate_history_items"] == 50
    assert manifest["scorer"]["k"] == 1
    assert (
        manifest["source"]["author_code"]["revision"]
        == "1272c62785d920c406d7233c772a704e960c224a"
    )
    assert "absolute interaction willingness" in manifest["scorer"]["score_limitations"]
    pointwise, ranking = (
        read_json(out / "metrics.json"),
        read_json(out / "metrics_ranking.json"),
    )
    assert (
        pointwise["threshold"]
        == ranking["pointwise_threshold"]
        == result["pointwise_threshold"]
    )
    assert "test_failure_as_zero_group" in ranking
    assert "recall@5" in ranking["test"]["macro_by_user_group_mean"]
    assert (
        "recall@5" in ranking["test_failure_as_zero_group"]["macro_by_user_group_mean"]
    )
    assert "test_failure_as_negative" in pointwise
    records = [
        json.loads(line)
        for line in (out / "llmsrec_syn_attempts.jsonl").read_text().splitlines()
    ]
    assert {r["phase"] for r in records} == {
        "selection_k1",
        "selection_k2",
        "val",
        "test",
    }
    assert len(chat.calls) == 8  # 2 groups * (two K candidates + final val + test)
    assert len(embedding_client.calls) == 2
    embedded_texts = [text for call in embedding_client.calls for text in call["input"]]
    # Eight demonstration prefixes and two distinct full target histories.
    assert len(embedded_texts) == len(set(embedded_texts)) == 10
    for record in records:
        assert record["completed"]
        demo = record["demonstration"]
        assert demo["eligible_user_count"] == 7
        assert all(u["user_id"] != record["user_id"] for u in demo["selected_users"])
        assert all(
            e["timestamp"] <= demo["cutoff"]
            for u in demo["selected_users"]
            for e in [*u["prefix"], u["next_item"]]
        )
        text = record["messages"][0]["content"]
        assert "rating" not in text and "Comedy" not in text and "playtime" not in text
    predictions = pd.read_parquet(out / "predictions.parquet")
    for split in ("val", "test"):
        actual = predictions[predictions.split.eq(split)].reset_index(drop=True)
        original = getattr(task, split).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            actual[["user_id", "item_id", "candidate_group", "target", "sampled"]],
            original[["user_id", "item_id", "candidate_group", "target", "sampled"]],
        )
    for split in ("train", "val", "test"):
        pd.testing.assert_frame_equal(getattr(task, split), getattr(before, split))


def test_test_labels_never_select_k_or_threshold(tmp_path, monkeypatch):
    task, dataset = build_task(tmp_path)
    first, _, _ = run(task, dataset, tmp_path / "first", monkeypatch)
    changed = deepcopy(task)
    changed.test["target"] = 1 - changed.test["target"]
    second, _, _ = run(changed, dataset, tmp_path / "second", monkeypatch)
    assert first["pointwise_threshold"] == second["pointwise_threshold"]
    assert read_json(tmp_path / "first/llmsrec_syn_selection.json") == read_json(
        tmp_path / "second/llmsrec_syn_selection.json"
    )
    a, b = [
        pd.read_parquet(tmp_path / name / "predictions.parquet")
        for name in ("first", "second")
    ]
    pd.testing.assert_series_equal(a.score, b.score)
    pd.testing.assert_series_equal(a.prediction, b.prediction)


def test_failure_aware_k_selection_uses_fixed_groups_and_smaller_k_ties():
    X = pd.DataFrame(
        {
            "user_id": ["u", "u", "v", "v"],
            "item_id": ["a", "b", "c", "d"],
            "candidate_group": ["g1", "g1", "g2", "g2"],
        }
    )
    y = pd.Series([1, 0, 1, 0])

    class Scorer:
        candidate_group_column = "candidate_group"

        def score(self, group):
            if self.k == 3 or (self.k == 2 and group.candidate_group.iloc[0] == "g2"):
                raise ValueError("insufficient neighbors")
            return pd.Series(
                [1, 0] if self.k == 2 else [0, 1],
                index=group.index,
                name="score",
                dtype=float,
            )

    result = llmsrec_syn.select_k(
        Scorer(), X, y, k_values=(4, 3, 2, 1), max_attempts=2, max_workers=2
    )
    assert result["selected_k"] == 1
    by_k = {r["k"]: r for r in result["candidates"]}
    assert by_k[1]["selection_score"] > by_k[2]["selection_score"] == 0.5
    assert by_k[3]["selection_score"] == 0.0
    assert by_k[3]["failed_groups"] == 2
    assert by_k[1]["selection_score"] == by_k[4]["selection_score"]
    assert {r["requested_groups"] for r in by_k.values()} == {2}


def test_k_selection_uses_top_five_even_when_top_ten_prefers_another_k():
    X = pd.DataFrame(
        {"user_id": ["u"] * 10, "item_id": range(10), "candidate_group": ["g"] * 10}
    )
    y = pd.Series([1] * 4 + [0] * 6)

    class Scorer:
        candidate_group_column = "candidate_group"

        def score(self, group):
            ranks = (
                [1, 6, 7, 8, 2, 3, 4, 5, 9, 10]
                if self.k == 1
                else [2, 3, 4, 10, 1, 5, 6, 7, 8, 9]
            )
            return pd.Series(
                [10 - rank for rank in ranks], index=group.index, dtype=float
            )

    result = llmsrec_syn.select_k(
        Scorer(), X, y, k_values=(1, 2), max_attempts=1, max_workers=1
    )
    assert result["selected_k"] == 2
    assert result["metric"] == "macro_by_user_group_mean_ndcg@5_failed_group_zero"
    from beyond_click_sim.evaluation import grouped_ranking_metrics

    scorer = Scorer()
    top_ten = []
    for k in (1, 2):
        scorer.k = k
        top_ten.append(
            grouped_ranking_metrics(y, scorer.score(X), X.candidate_group, ks=(10,))[
                "ndcg@10"
            ]
        )
    assert top_ten[0] > top_ten[1]


def test_all_validation_configurations_fail_without_touching_test(
    tmp_path, monkeypatch
):
    task, dataset = build_task(tmp_path)

    class UnreadableTest:
        def copy(self):
            pytest.fail("test data accessed during K selection")

        def __getitem__(self, key):
            pytest.fail("test data accessed during K selection")

    task.test = UnreadableTest()
    chat, embeddings = RankingClient(), EmbeddingClient()
    monkeypatch.setattr(
        llmsrec_syn,
        "make_llm_client",
        lambda name: chat if name == "llm" else embeddings,
    )
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="no valid rankings"):
        llmsrec_syn.run_method(
            task,
            out,
            canonical_dataset=dataset,
            client_name="llm",
            embedding_client_name="embed",
            k_values=(8, 9),
            max_llm_attempts=2,
        )
    selection = read_json(out / "llmsrec_syn_selection.json")
    assert all(r["selection_score"] == 0 for r in selection["candidates"])
    assert chat.calls == []
    assert not (out / "predictions.parquet").exists()


def test_generation_failures_are_counted_and_final_artifacts_keep_coverage(
    tmp_path, monkeypatch
):
    task, dataset = build_task(tmp_path)
    # Exhaust one validation group's retries during configuration selection.
    client = RankingClient(["C1", "C1"])
    _, client, _ = run(
        task, dataset, tmp_path / "run", monkeypatch, chat=client, k_values=(1,)
    )
    selection = read_json(tmp_path / "run/llmsrec_syn_selection.json")
    assert selection["candidates"][0]["failed_groups"] == 1
    assert selection["candidates"][0]["successful_groups"] == 1
    records = [
        json.loads(line)
        for line in (tmp_path / "run/llmsrec_syn_attempts.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len([r for r in records if not r["completed"]]) == 2


def test_registered_methods_preserve_paper_inputs_and_separate_embedding_backend(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        llmsrec_syn, "run_method", lambda *args, **kwargs: calls.append(kwargs) or {}
    )
    for name in [name for name in METHOD_RUNNERS if name.startswith("llmsrec_syn_")]:
        METHOD_RUNNERS[name](None, tmp_path)
        assert calls[-1]["method_name"] == name
        assert "use_item_stats" not in calls[-1]
        assert (
            "embedding_client_name" not in calls[-1]
        )  # default OpenAI, independent of Qwen endpoint
    assert len(calls) == 6


def test_failed_test_groups_are_reported_without_changing_candidate_population(
    tmp_path, monkeypatch
):
    task, dataset = build_task(tmp_path)

    class FailingTestClient(RankingClient):
        def create(self, **kwargs):
            if len(self.calls) in (4, 5):
                self.replies = ["C1"]
            return super().create(**kwargs)

    out = tmp_path / "run"
    result, _, _ = run(
        task, dataset, out, monkeypatch, chat=FailingTestClient(), k_values=(1,)
    )
    assert result["llm_errors"]["test"] == 1
    ranking = read_json(out / "metrics_ranking.json")
    assert (
        ranking["test_failure_as_zero_group"]["macro_by_user_group_mean"][
            "failed_groups"
        ]
        == 1
    )
    predictions = pd.read_parquet(out / "predictions.parquet")
    test = predictions[predictions.split.eq("test")]
    assert len(test) == len(task.test)
    assert (
        test.groupby("candidate_group").score.apply(lambda s: s.isna().all()).sum() == 1
    )


def test_early_validation_events_of_other_users_are_not_in_demonstration_pool(tmp_path):
    task, dataset = build_task(tmp_path)
    # Move one user's existing training cutoff later while retaining exact source
    # and task projections. Their own ordering/partition membership is unchanged.
    interactions = dataset.load_interactions()
    interactions.loc[interactions.user_id.eq("u7"), "timestamp"] += 100
    interactions.to_parquet(dataset.interactions_path)
    prepared = prepare_training(task, dataset)
    from beyond_click_sim.scorers import LLMSRecSynScorer
    from beyond_click_sim.scorers.history.llmsrec_syn_retrieval import CachedEmbeddings

    embeddings = CachedEmbeddings(
        EmbeddingClient(), "fake", model_identity="fake", training_fingerprint="s"
    )
    scorer = LLMSRecSynScorer(RankingClient(), "fake", embeddings=embeddings, k=7).fit(
        task.train.drop(columns="target"),
        task.train.target,
        train_event_metadata=prepared.events,
        catalog=prepared.catalog,
    )
    demo = scorer.demonstration("u7")
    assert demo["cutoff"] == 109
    # u0's validation/test events at days 10..14 are before 109, but stay out.
    u0 = next(u for u in demo["selected_users"] if u["user_id"] == "u0")
    assert u0["next_item"]["timestamp"] == 9
    assert all(e["timestamp"] < 9 for e in u0["prefix"])


def test_no_eager_clients_or_random_changes_when_importing_new_module():
    import subprocess
    import sys

    script = """
import random
import beyond_click_sim.llm_clients as clients
def fail(*args, **kwargs):
    raise AssertionError("eager client construction")
clients.make_llm_client = fail
state = random.getstate()
from beyond_click_sim.scorers import LLMSRecSynScorer
from runners.in_distribution.interaction_prediction.methods import METHOD_RUNNERS
assert random.getstate() == state
assert "llmsrec_syn_openai_smoke" in METHOD_RUNNERS
"""
    subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, check=True, capture_output=True
    )
