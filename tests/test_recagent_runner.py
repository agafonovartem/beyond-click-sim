from copy import deepcopy
import inspect
import json

import pandas as pd
import pytest

from runners.in_distribution.interaction_prediction.methods import (
    METHOD_RUNNERS,
    recagent_yes_no,
)
from runners.in_distribution.interaction_prediction.methods.common import (
    ranking_metrics_for_split,
)
from test_llmsrec_syn_runner import build_task
from test_recagent_memory import FakeEmbeddings
from test_recagent_scorer import FakeClient


def run(task, dataset, out, **kwargs):
    client = kwargs.pop("client", FakeClient(importance="6"))
    result = recagent_yes_no.run_method(
        task,
        out,
        method_name="recagent-test",
        client_name="fake",
        client=client,
        model="fake-model",
        model_revision="fake-revision",
        canonical_dataset=dataset,
        embedding_provider=kwargs.pop("embedding_provider", FakeEmbeddings()),
        use_demographics=False,
        candidate_item_summaries=False,
        max_llm_attempts=kwargs.pop("max_llm_attempts", 1),
        **kwargs,
    )
    return result, client


def read(path):
    return json.loads(path.read_text())


def test_combined_artifacts_use_same_binary_scores_and_no_extra_calls(tmp_path):
    task, data = build_task(tmp_path)
    before = deepcopy(task)
    out = tmp_path / "run"
    result, client = run(task, data, out)
    frame = pd.read_parquet(out / "predictions.parquet")
    assert frame.score.isin([0, 1]).all()
    assert frame.prediction.astype(int).equals(
        frame.score.astype(int).rename("prediction")
    )
    assert (
        client.counts["action"]
        == client.counts["profile_summary"]
        == task.test.candidate_group.nunique()
    )
    assert (
        len(client.calls)
        == read(out / "preflight.json")[
            "base_llm_calls_without_promotions_reflections_retries"
        ]
    )
    ranking = read(out / "metrics_ranking.json")
    expected = ranking_metrics_for_split(
        X=frame,
        y=frame.target,
        scores=frame.score,
        candidate_group_column="candidate_group",
    )
    assert ranking["test"] == expected
    assert "recall@5" in ranking["test"]["macro_by_user_group_mean"]
    assert "ndcg@5" in ranking["test"]["macro_by_user_group_mean"]
    manifest = read(out / "manifest.json")
    assert manifest["decision_rule"]["threshold"] is None
    assert manifest["initialization"]["initialized_users"] == 2
    assert "unused" in manifest["validation"]
    assert result["coverage"]["scored_rows"] == len(task.test)
    for split in ("train", "val", "test"):
        pd.testing.assert_frame_equal(getattr(task, split), getattr(before, split))
    trace = [
        json.loads(line)
        for line in (out / "recagent_trace.jsonl").read_text().splitlines()
    ]
    calls = [r for r in trace if "messages" in r]
    assert all("elapsed_seconds" in r and "usage" in r for r in calls)
    assert all(
        r["max_tokens"] == 16
        if ":importance]" in r["messages"][-1]["content"]
        else r["max_tokens"] == 256
        for r in client.calls
    )
    assert all(
        r["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
        for r in client.calls
    )


def test_test_and_validation_labels_do_not_change_prompts_scores_or_initialization(
    tmp_path,
):
    task, data = build_task(tmp_path)
    a, ca = run(task, data, tmp_path / "a")
    changed = deepcopy(task)
    changed.test[changed.schema.target_column] = (
        1 - changed.test[changed.schema.target_column]
    )
    changed.val[changed.schema.target_column] = (
        1 - changed.val[changed.schema.target_column]
    )
    b, cb = run(changed, data, tmp_path / "b")
    assert ca.calls == cb.calls
    left = pd.read_parquet(tmp_path / "a/predictions.parquet")
    right = pd.read_parquet(tmp_path / "b/predictions.parquet")
    pd.testing.assert_frame_equal(
        left[["score", "prediction"]], right[["score", "prediction"]]
    )
    assert (
        read(tmp_path / "a/manifest.json")["initialization"]["cache_keys"]
        == read(tmp_path / "b/manifest.json")["initialization"]["cache_keys"]
    )


def test_failed_group_missing_outputs_coverage_and_failure_inclusive_metrics(tmp_path):
    task, data = build_task(tmp_path)
    client = FakeClient(
        importance="6",
        fail=lambda s, n, _: "invalid" if s == "action" and n == 1 else None,
    )
    result, _ = run(task, data, tmp_path / "out", client=client)
    frame = pd.read_parquet(tmp_path / "out/predictions.parquet")
    assert frame.score.isna().any() and frame.score.notna().any()
    assert frame.loc[frame.score.isna(), "prediction"].isna().all()
    ranking = read(tmp_path / "out/metrics_ranking.json")
    strict = ranking["test_failure_as_zero_group"]["macro_by_user_group_mean"]
    assert strict["failed_groups"] == 1
    assert strict["ndcg@5"] == pytest.approx(
        ranking["test"]["macro_by_user_group_mean"]["ndcg@5"] / 2
    )
    assert "test_failure_as_negative" in result


def test_initialization_failure_only_removes_that_users_outputs(tmp_path):
    task, data = build_task(tmp_path)
    client = FakeClient(
        importance="6",
        fail=lambda s, n, _: "word " * 121
        if s == "profile_update" and n == 1
        else None,
    )
    run(task, data, tmp_path / "out", client=client)
    manifest = read(tmp_path / "out/manifest.json")
    assert manifest["initialization"]["initialized_users"] == 1
    assert len(manifest["initialization"]["unavailable_users"]) == 1
    frame = pd.read_parquet(tmp_path / "out/predictions.parquet")
    user = next(iter(manifest["initialization"]["unavailable_users"]))
    assert frame.loc[frame.user_id.eq(user), "score"].isna().all()
    assert client.counts["action"] == 1


def test_all_failed_writes_artifacts_before_aborting(tmp_path):
    task, data = build_task(tmp_path)
    with pytest.raises(RuntimeError, match="no valid scores"):
        run(task, data, tmp_path / "out", client=FakeClient(importance="bad"))
    frame = pd.read_parquet(tmp_path / "out/predictions.parquet")
    assert frame.score.isna().all() and frame.prediction.isna().all()
    assert read(tmp_path / "out/manifest.json")["status"] == "all_groups_failed"
    ranking = read(tmp_path / "out/metrics_ranking.json")
    assert ranking["test"] is None
    assert (
        ranking["test_failure_as_zero_group"]["macro_by_user_group_mean"]["ndcg@5"] == 0
    )
    assert (tmp_path / "out/llm_errors.jsonl").read_text()


def test_preflight_never_constructs_clients_or_resolves_weights(tmp_path, monkeypatch):
    task, data = build_task(tmp_path)
    monkeypatch.setattr(
        recagent_yes_no,
        "make_llm_client",
        lambda *_: pytest.fail("No client in preflight"),
    )
    provider = FakeEmbeddings()
    provider.metadata = lambda: pytest.fail("No model metadata resolution in preflight")
    result, client = run(
        task,
        data,
        tmp_path / "out",
        preflight_only=True,
        max_candidate_groups=1,
        embedding_provider=provider,
    )
    assert result["users"] == result["candidate_groups"] == 1
    assert result["training_events"] == 10
    assert client.calls == provider.calls == []
    assert not (tmp_path / "out/predictions.parquet").exists()


@pytest.mark.parametrize(
    "mutation", ["steam", "random", "train_value", "order", "source_id", "timestamp"]
)
def test_bad_provenance_rejected_before_client(tmp_path, monkeypatch, mutation):
    task, data = build_task(tmp_path)
    if mutation == "steam":
        task.manifest["dataset"] = "steam"
    elif mutation == "random":
        task.manifest["splitter"]["shuffle"] = True
    elif mutation == "train_value":
        task.train.loc[task.train.index[0], "rating"] = -999
    elif mutation == "order":
        task.train = task.train.iloc[::-1]
    else:
        interactions = data.load_interactions()
        if mutation == "source_id":
            interactions["interaction_id"] = "duplicate"
        else:
            interactions["timestamp"] = None
        interactions.to_parquet(data.interactions_path)
    monkeypatch.setattr(
        recagent_yes_no,
        "make_llm_client",
        lambda *_: pytest.fail("Validate before client"),
    )
    with pytest.raises(ValueError):
        recagent_yes_no.run_method(
            task,
            tmp_path / "out",
            canonical_dataset=data,
            use_demographics=False,
            candidate_item_summaries=False,
        )


def test_registry_default_wrappers(monkeypatch):
    calls = []
    monkeypatch.setattr(
        recagent_yes_no, "run_method", lambda *args, **kwargs: calls.append(kwargs)
    )
    for model in ("qwen3_8b", "qwen36_27b"):
        for scope in ("smoke", "full"):
            name = f"recagent_yes_no_litellm_{model}_{scope}"
            METHOD_RUNNERS[name](None, None)
            call = calls[-1]
            assert call["method_name"] == name
            assert call.get("max_candidate_groups") == (5 if scope == "smoke" else None)
            assert call.get("model", "Qwen/Qwen3-8B") == (
                "Qwen/Qwen3-8B" if model == "qwen3_8b" else "Qwen/Qwen3.6-27B"
            )


def test_registered_defaults():
    defaults = {
        k: v.default
        for k, v in inspect.signature(recagent_yes_no.run_method).parameters.items()
    }
    assert defaults["max_history_items"] is None
    assert defaults["max_llm_attempts"] == 5
    assert defaults["use_demographics"] is True
    assert defaults["candidate_item_summaries"] is True
    assert defaults["use_item_stats"] is False
    assert defaults["temperature"] == 0
    assert defaults["max_workers"] == 1
