from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from math import log2
from pathlib import Path
import re
from types import SimpleNamespace

import pandas as pd
import pytest

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.scorers import LLMRankScorer
from beyond_click_sim.tasks import (
    AlignmentInteractionTaskBuilder,
    CappedUserInteractionCandidateSampler,
    MinUserInteractionsFilter,
    RandomFractionSplitter,
    Task,
    TaskSchema,
    TrainItemRatingStatistics,
)
from runners.in_distribution.interaction_prediction.methods import METHOD_RUNNERS
from runners.in_distribution.interaction_prediction.methods import (
    llmrank,
    llm_listwise_ranking,
    llm_yes_no,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    def __init__(self, *, usage=None, fail_bad=False):
        self.chat = SimpleNamespace(completions=self)
        self.calls = []
        self.usage = usage
        self.fail_bad = fail_bad

    def create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][1]["content"]
        candidates = dict(re.findall(r"^(C\d+)\. item_title: ([^;\n]+)", prompt, re.M))
        if self.fail_bad and any(
            title.startswith("Bad") for title in candidates.values()
        ):
            text = "C1"  # Deliberately incomplete; no fabricated partial score.
        elif "Fill each line after the colon with either yes or no." in prompt:
            first = min(candidates, key=candidates.get)
            text = "\n".join(
                f"{label}: {'yes' if label == first else 'no'}" for label in candidates
            )
        else:
            text = "\n".join(sorted(candidates, key=candidates.get))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=self.usage,
        )


def toy_task():
    train = pd.DataFrame(
        {
            "user_id": ["u"] * 3,
            "item_id": ["h1", "h2", "h3"],
            "item_title": ["Older", "Middle", "Newest"],
            "item_genres": ["Comedy"] * 3,
            "rating": [1, 3, 5],
            "target": [1] * 3,
        }
    )

    def group(prefix):
        return pd.DataFrame(
            {
                "user_id": ["u"] * 3,
                "item_id": [f"{prefix}{i}" for i in range(3)],
                "candidate_group": [prefix] * 3,
                "item_title": ["A", "B", "C"],
                "item_genres": ["Comedy"] * 3,
                "rating": [pd.NA] * 3,
                "target": [1, 0, 1],
                "sampled": [False, True, False],
            }
        )

    return Task(
        name="toy-temporal",
        train=train,
        val=group("v"),
        test=group("t"),
        schema=TaskSchema(
            candidate_group_column="candidate_group", sampled_column="sampled"
        ),
        manifest={
            "dataset": "ml-1m",
            "splitter": {
                "class": "RandomFractionSplitter",
                "shuffle": False,
                "timestamp_column": "timestamp",
                "group_column": "user_id",
            },
        },
    )


def run(task, output_dir, monkeypatch, *, client=None, **kwargs):
    client = client or FakeClient()
    monkeypatch.setattr(llmrank, "make_llm_client", lambda _: client)
    return llmrank.run_method(
        task,
        output_dir,
        method_name="llmrank_fake",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        **kwargs,
    )


def read_json(path):
    return json.loads(path.read_text())


def test_runner_artifacts_diagnostics_and_source_fingerprints(tmp_path, monkeypatch):
    client = FakeClient()
    task = toy_task()
    result = run(task, tmp_path, monkeypatch, client=client, bootstrap_seed=17)
    assert len(client.calls) == 6
    manifest = read_json(tmp_path / "manifest.json")
    ranking = read_json(tmp_path / "metrics_ranking.json")
    pointwise = read_json(tmp_path / "metrics.json")
    assert result == ranking
    assert result["ranking_evaluation"]["score_mapping"] == LLMRankScorer.score_mapping
    assert (
        manifest["decision_rule"]["ranking"]["score_mapping"]
        == LLMRankScorer.score_mapping
    )
    assert manifest["decision_rule"]["pointwise"]["kind"] == "threshold_on_validation"
    assert pointwise["threshold"] == ranking["pointwise_threshold"]
    assert manifest["task"]["manifest"] == task.manifest
    assert manifest["scorer"]["prompt"]["temporal_language"] is True
    assert manifest["scorer"]["max_history_items"] == 20
    bootstrap = manifest["scorer"]["bootstrap"]
    assert bootstrap["rounds"] == 3 and bootstrap["seed"] == 17
    assert bootstrap["max_scorer_requests_per_group"] == 15
    records = [
        json.loads(line)
        for line in (tmp_path / "llmrank_rounds.jsonl").read_text().splitlines()
    ]
    assert [record["split"] for record in records] == ["val", "test"]
    assert all(record["completed"] and len(record["rounds"]) == 3 for record in records)
    assert all(
        r["elapsed_seconds"] >= 0 for record in records for r in record["rounds"]
    )
    for split in ("val", "test"):
        stats = bootstrap["request_statistics"][split]
        assert stats["scorer_issued_requests"] == 3
        assert stats["requests_without_usage"] == 3 and not stats["usage_complete"]
        assert set(stats["reported_token_totals"].values()) == {None}
    assert (
        read_json(tmp_path / "llmrank_request_summary.json")
        == bootstrap["request_statistics"]
    )
    path = "src/beyond_click_sim/tasks/splitters.py"
    assert (
        manifest["source"]["source_file_sha256"][path]
        == sha256((ROOT / path).read_bytes()).hexdigest()
    )
    assert manifest["source"]["paper"]["implementation_kind"] == "controlled_adaptation"
    assert (
        manifest["source"]["author_code"]["published_version_correspondence_verified"]
        is False
    )
    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert predictions["score"].tolist() == [6, 3, 0, 6, 3, 0]


def test_same_disagreeing_ensemble_supplies_ranking_and_binary_outputs(
    tmp_path, monkeypatch
):
    task = toy_task()
    for split, prefix in (("val", "v"), ("test", "t")):
        setattr(
            task,
            split,
            pd.DataFrame(
                {
                    "user_id": ["u"] * 6,
                    "item_id": [f"{prefix}{i}" for i in range(6)],
                    "candidate_group": [prefix] * 6,
                    "item_title": list("ABCDEF"),
                    "item_genres": ["Comedy"] * 6,
                    "rating": [pd.NA] * 6,
                    "target": [1, 0, 0, 0, 0, 1],
                    "sampled": [False, True, True, True, True, False],
                },
                index=[11, 3, 27, 6, 9, 42],
            ),
        )

    class DisagreeingClient(FakeClient):
        def create(self, **kwargs):
            order = ("ABCDEF", "ACBEFD", "BACFDE")[len(self.calls) % 3]
            self.calls.append(kwargs)
            prompt = kwargs["messages"][1]["content"]
            label_for_title = {
                title: label
                for label, title in re.findall(
                    r"^(C\d+)\. item_title: ([^;\n]+)", prompt, re.M
                )
            }
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="\n".join(label_for_title[title] for title in order)
                        )
                    )
                ]
            )

    client = DisagreeingClient()
    result = run(task, tmp_path, monkeypatch, client=client, bootstrap_seed=17)
    assert len(client.calls) == 6  # Three rounds per split, shared by both views.
    saved = pd.read_parquet(tmp_path / "predictions.parquet")
    pointwise = read_json(tmp_path / "metrics.json")
    assert result == read_json(tmp_path / "metrics_ranking.json")
    assert pointwise["threshold"] == result["pointwise_threshold"]

    # D/E/F tie across positions 4--6, including one of the two positives (F).
    # Expected DCG counts F with probability 1/3 at each tied position.
    expected_ndcg = (1 + (1 / log2(5) + 1 / log2(6)) / 3) / (1 + 1 / log2(3))
    for split in ("val", "test"):
        rows = saved.loc[saved.split.eq(split)]
        assert rows.item_id.tolist() == getattr(task, split).item_id.tolist()
        assert rows.score.tolist() == [14, 12, 10, 3, 3, 3]
        assert rows.prediction.tolist() == [True, False, False, False, False, False]
        assert (
            rows.prediction.tolist() == rows.score.ge(pointwise["threshold"]).tolist()
        )
        assert pointwise[split]["macro_by_user_group_mean"]["f1"] == pytest.approx(
            2 / 3
        )
        for field in (split, f"{split}_failure_as_zero_group"):
            metrics = result[field]["macro_by_user_group_mean"]
            assert metrics["groups_with_score_ties"] == 1
            assert metrics["ndcg@5"] == pytest.approx(expected_ndcg)
            assert metrics["recall@5"] == pytest.approx(5 / 6)
            # Binary labels would instead tie five items and yield recall 9/10.
            assert metrics["recall@5"] != pytest.approx(9 / 10)


def test_reported_usage_and_test_labels_do_not_select_threshold(tmp_path, monkeypatch):
    task = toy_task()
    usage = SimpleNamespace(prompt_tokens=20, completion_tokens=5, total_tokens=25)
    first_client = FakeClient(usage=usage)
    first = run(task, tmp_path / "first", monkeypatch, client=first_client)
    changed = deepcopy(task)
    changed.test["target"] = 1 - changed.test["target"]
    second_client = FakeClient(usage=usage)
    second = run(changed, tmp_path / "second", monkeypatch, client=second_client)
    assert first_client.calls == second_client.calls
    assert first["pointwise_threshold"] == second["pointwise_threshold"]
    a = pd.read_parquet(tmp_path / "first/predictions.parquet")
    b = pd.read_parquet(tmp_path / "second/predictions.parquet")
    pd.testing.assert_series_equal(a["score"], b["score"])
    pd.testing.assert_series_equal(a["prediction"], b["prediction"])
    stats = read_json(tmp_path / "first/llmrank_request_summary.json")
    for phase in stats.values():
        assert phase["usage_complete"]
        assert phase["reported_token_totals"] == {
            "prompt_tokens": 60,
            "completion_tokens": 15,
            "total_tokens": 75,
        }


def test_failed_groups_keep_failure_aware_metrics_and_diagnostics(
    tmp_path, monkeypatch
):
    task = toy_task()
    failed = task.test.assign(
        candidate_group="bad", item_title=["Bad A", "Bad B", "Bad C"]
    )
    task.test = pd.concat([task.test, failed], ignore_index=True)
    result = run(task, tmp_path, monkeypatch, client=FakeClient(fail_bad=True))
    assert result["llm_errors"]["test"] == 1
    assert "test_failure_as_zero_group" in result
    assert "test_failure_as_negative" in read_json(tmp_path / "metrics.json")
    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    missing = predictions.loc[predictions.candidate_group == "bad"]
    assert len(missing) == len(failed)
    assert missing[["score", "prediction"]].isna().all().all()
    assert result["coverage"]["test"]["failed_rows"] == len(failed)
    assert result["coverage"]["test"]["scored_fraction"] == 0.5
    strict = result["test_failure_as_zero_group"]["macro_by_user_group_mean"]
    parsed = result["test"]["macro_by_user_group_mean"]
    assert strict["n_groups"] == 2
    assert strict["scored_groups"] == strict["failed_groups"] == 1
    for metric in ("ndcg@5", "recall@5"):
        assert strict[metric] == pytest.approx(parsed[metric] / 2)
    pointwise = read_json(tmp_path / "metrics.json")
    assert pointwise["test_failure_as_negative"]["macro_by_user_group_mean"][
        "f1"
    ] == pytest.approx(pointwise["test"]["macro_by_user_group_mean"]["f1"] / 2)
    records = [
        json.loads(line)
        for line in (tmp_path / "llmrank_rounds.jsonl").read_text().splitlines()
    ]
    failures = [record for record in records if record["candidate_group"] == "bad"]
    assert len(failures) == 5 and all(not record["completed"] for record in failures)
    assert (
        read_json(tmp_path / "llmrank_request_summary.json")["test"][
            "scorer_issued_requests"
        ]
        == 8
    )


def test_fully_failed_validation_still_flushes_both_phase_diagnostics(
    tmp_path, monkeypatch
):
    task = toy_task()
    task.val["item_title"] = ["Bad A", "Bad B", "Bad C"]
    with pytest.raises(RuntimeError, match="validation"):
        run(task, tmp_path, monkeypatch, client=FakeClient(fail_bad=True))
    stats = read_json(tmp_path / "llmrank_request_summary.json")
    assert stats["val"]["ensemble_attempts"] == 5
    assert stats["val"]["completed_ensembles"] == 0
    assert stats["test"]["completed_ensembles"] == 1


@pytest.mark.parametrize(
    "change", ["steam", "random", "missing", "no_timestamp", "global"]
)
def test_rejects_non_temporal_or_unsupported_tasks_before_client(
    tmp_path, monkeypatch, change
):
    task = toy_task()
    if change == "steam":
        task.manifest["dataset"] = "steam"
    elif change == "random":
        task.manifest["splitter"]["shuffle"] = True
    elif change == "missing":
        task.manifest.pop("splitter")
    elif change == "no_timestamp":
        task.manifest["splitter"].pop("timestamp_column")
    else:
        task.manifest["splitter"]["group_column"] = None
    monkeypatch.setattr(
        llmrank, "make_llm_client", lambda _: pytest.fail("must reject before client")
    )
    with pytest.raises(ValueError):
        llmrank.run_method(
            task,
            tmp_path,
            method_name="bad",
            client_name="bad",
            model="bad",
            max_candidate_groups=None,
        )


def test_registered_qwen_wrappers_use_existing_backends(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        llmrank, "run_method", lambda *args, **kwargs: calls.append(kwargs) or {}
    )
    for model_name, model, workers in [
        ("qwen3_8b", "Qwen/Qwen3-8B", 64),
        ("qwen36_27b", "Qwen/Qwen3.6-27B", 32),
    ]:
        for mode, limit in [("smoke", 25), ("full", None)]:
            name = f"llmrank_litellm_{model_name}_with_item_stats_{mode}"
            METHOD_RUNNERS[name](toy_task(), tmp_path)
            call = calls[-1]
            assert call["method_name"] == name and call["model"] == model
            assert (
                call["max_workers"] == workers and call["max_candidate_groups"] == limit
            )
            assert (
                call["use_item_stats"] is True
                and call["client_name"] == "litellm_local"
            )
            assert call["extra_body"] == {
                "chat_template_kwargs": {"enable_thinking": False}
            }


def build_temporal_task(tmp_path):
    root = tmp_path / "canonical"
    root.mkdir()
    rows = [
        {
            "user_id": user,
            "item_id": f"{user}-{i:02}",
            "timestamp": i,
            "rating": i % 5 + 1,
            "target_interact": 1,
        }
        for user in ("u1", "u2")
        for i in reversed(range(35))
    ]
    pd.DataFrame({"user_id": ["u1", "u2"]}).to_parquet(root / "users.parquet")
    pd.DataFrame(
        {
            "item_id": [row["item_id"] for row in rows],
            "title": [f"Movie {row['item_id']}" for row in rows],
            "genres": ["Comedy"] * len(rows),
        }
    ).to_parquet(root / "items.parquet")
    pd.DataFrame(rows).to_parquet(root / "interactions.parquet")
    (root / "manifest.json").write_text("{}")
    dataset = CanonicalDataset(
        name="ml-1m",
        version="toy",
        root=root,
        users_path=root / "users.parquet",
        items_path=root / "items.parquet",
        interactions_path=root / "interactions.parquet",
        manifest_path=root / "manifest.json",
    )
    return AlignmentInteractionTaskBuilder(
        name="toy-temporal-matched",
        dataset_filter=MinUserInteractionsFilter(10),
        splitter=RandomFractionSplitter(0.7, 0.1, 0.2),
        sampler=CappedUserInteractionCandidateSampler(
            negative_ratio=1, total_items=20, seed=0
        ),
        history_context_columns=("rating",),
        item_feature_builder=TrainItemRatingStatistics(value_column="rating"),
    ).build(dataset)


def test_fake_three_method_comparison_uses_one_temporal_task(tmp_path, monkeypatch):
    task = build_temporal_task(tmp_path)
    original = deepcopy(task)
    clients = {}
    shared = dict(
        client_name="fake",
        model="same-model",
        max_candidate_groups=None,
        max_history_items=20,
        temperature=0,
        max_tokens=512,
        max_llm_attempts=5,
        use_item_stats=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    manifests = []
    histories = []
    for name, runner in [
        ("yes_no", llm_yes_no),
        ("listwise", llm_listwise_ranking),
        ("llmrank", llmrank),
    ]:
        client = FakeClient()
        clients[name] = client
        monkeypatch.setattr(runner, "make_llm_client", lambda _, client=client: client)
        output = tmp_path / name
        runner.run_method(task, output, method_name=f"matched_{name}", **shared)
        manifest = read_json(output / "manifest.json")
        manifests.append(manifest)
        predictions = pd.read_parquet(output / "predictions.parquet")
        scored_test = predictions[predictions.split == "test"].reset_index(drop=True)
        # Parquet materializes the hidden object-column pd.NA ratings as None.
        # Verify missing feedback explicitly and compare every other value.
        assert scored_test["rating"].isna().all() and task.test["rating"].isna().all()
        pd.testing.assert_frame_equal(
            scored_test[task.test.columns].drop(columns="rating"),
            task.test.reset_index(drop=True).drop(columns="rating"),
        )
        history_windows = set()
        for call in client.calls:
            assert (
                call["model"],
                call["temperature"],
                call["max_tokens"],
                call["extra_body"],
            ) == (
                "same-model",
                0,
                512,
                shared["extra_body"],
            )
            prompt = call["messages"][1]["content"]
            lines = tuple(re.findall(r"^H\d+\. .*", prompt, re.M))
            assert len(lines) == 20
            history_windows.add(lines)
            assert "rating: " not in prompt.split("Candidate items:", 1)[1].replace(
                "average rating: ", ""
            )
        histories.append(history_windows)
    assert histories[0] == histories[1] == histories[2]
    assert (
        manifests[0]["scorer"]["prompt_columns"]
        == manifests[1]["scorer"]["prompt_columns"]
        == manifests[2]["scorer"]["prompt_columns"]
    )
    assert manifests[0]["task"] == manifests[1]["task"] == manifests[2]["task"]
    for user, group in task.train.groupby("user_id"):
        assert group.item_id.tolist() == sorted(group.item_id)
        expected = group.tail(20)
        matching = [lines for lines in histories[2] if f"Movie {user}-" in lines[0]]
        assert len(matching) == 1
        assert f"Movie {expected.iloc[-1].item_id}" in matching[0][-1]
    ngroups = task.val.candidate_group.nunique() + task.test.candidate_group.nunique()
    assert len(clients["llmrank"].calls) == 3 * ngroups
    assert len(clients["listwise"].calls) == ngroups
    assert len(clients["yes_no"].calls) == task.test.candidate_group.nunique()
    for split in ("train", "val", "test"):
        pd.testing.assert_frame_equal(getattr(task, split), getattr(original, split))
