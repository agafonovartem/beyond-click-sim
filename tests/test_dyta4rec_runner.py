from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from beyond_click_sim.scorers.dyta4rec import DyTA4RecConfig
from runners.in_distribution.interaction_prediction.methods import (
    METHOD_RUNNERS,
    dyta4rec,
)
from runners.in_distribution.interaction_prediction.methods.common import (
    pointwise_metrics_for_split,
    ranking_metrics_for_split,
)
from test_dyta4rec_scorer import FakeClient, FakeEmbeddings
from test_llmsrec_syn_runner import build_task as build_base_task


def build_task(path):
    task, dataset = build_base_task(path, repeated=True)
    users = dataset.load_users().assign(age=25, gender="M", occupation=12)
    users.to_parquet(dataset.users_path, index=False)
    for split in ("train", "val", "test"):
        frame = getattr(task, split)
        frame["user_age"], frame["user_gender"], frame["user_occupation"] = 25, "M", 12
    return task, dataset


def run(task, dataset, out, **kwargs):
    client = kwargs.pop("client", FakeClient(flags=("no", "no")))
    result = dyta4rec.run_method(
        task,
        out,
        client_name="fake",
        client=client,
        model="fake",
        model_revision="fake-v1",
        canonical_dataset=dataset,
        embedding_provider=kwargs.pop("embedding_provider", FakeEmbeddings()),
        **kwargs,
    )
    return result, client


def read(path):
    return json.loads(path.read_text())


def test_combined_outputs_ties_and_exactly_ten_group_calls(tmp_path):
    task, dataset = build_task(tmp_path)
    original = deepcopy(task)
    out = tmp_path / "result"
    result, client = run(task, dataset, out, max_candidate_groups=1)
    assert len(client.calls) == 15  # five initialization + ten group calls
    assert (
        sum(
            client.counts[s]
            for s in [
                "cluster_analysis",
                "cluster_ranking",
                "sequence_ranking",
                "profile_analysis",
                "profile_ranking",
            ]
        )
        == 10
    )
    saved = pd.read_parquet(out / "predictions.parquet")
    threshold = result["pointwise_threshold"]
    assert saved.prediction.tolist() == (saved.score >= threshold).tolist()
    traces = [
        json.loads(s) for s in (out / "dyta4rec_trace.jsonl").read_text().splitlines()
    ]
    completed = [r for r in traces if r["event"] == "group_completed"]
    assert len(completed) == 2
    for split, record in zip(("val", "test"), completed):
        rows = saved[saved.split == split]
        assert rows.score.tolist() == record["fusion"]["scores"]
        assert rows.score.iloc[0] == rows.score.iloc[-1]
        assert record["fusion"]["weights"] == [1, 1, 0]
        direct = ranking_metrics_for_split(
            X=rows,
            y=rows.target,
            scores=rows.score,
            candidate_group_column="candidate_group",
        )
        assert direct == result[split]
    manifest = read(out / "manifest.json")
    assert manifest["inference_calls_for_classification"] == 0
    assert manifest["scorer"]["memory_mode"] == "yulan_retrieval_two_level_v1"
    assert manifest["initialization_and_inference"]["usage_complete"]
    for split in ("train", "val", "test"):
        pd.testing.assert_frame_equal(getattr(task, split), getattr(original, split))


def test_hidden_outcomes_do_not_affect_prompts_scores_snapshot_or_threshold(tmp_path):
    task, dataset = build_task(tmp_path)
    a, first = run(task, dataset, tmp_path / "a", max_candidate_groups=1)
    changed = deepcopy(task)
    changed.test["target"] = 1 - changed.test.target
    changed.test["rating"] = 9999
    changed.test["heldout_time"] = "FUTURE_SECRET"
    changed.test[changed.schema.sampled_column] = ~changed.test[
        changed.schema.sampled_column
    ]
    b, second = run(changed, dataset, tmp_path / "b", max_candidate_groups=1)
    assert first.calls == second.calls
    assert a["pointwise_threshold"] == b["pointwise_threshold"]
    assert read(tmp_path / "a/dyta4rec_snapshots.json") == read(
        tmp_path / "b/dyta4rec_snapshots.json"
    )
    p1, p2 = [pd.read_parquet(tmp_path / d / "predictions.parquet") for d in ("a", "b")]
    assert p1.score.tolist() == p2.score.tolist()
    changed.val["target"] = 0
    c, third = run(changed, dataset, tmp_path / "c", max_candidate_groups=1)
    assert third.calls == first.calls
    assert c["pointwise_threshold"] != a["pointwise_threshold"]


def test_full_population_statistics_do_not_shrink_with_scored_cohort(tmp_path):
    task, dataset = build_task(tmp_path)
    run(task, dataset, tmp_path / "a", max_candidate_groups=1)
    run(task, dataset, tmp_path / "b", max_candidate_groups=None)
    one = read(tmp_path / "a/dyta4rec_snapshots.json")
    many = read(tmp_path / "b/dyta4rec_snapshots.json")
    for user, snapshot in one.items():
        assert snapshot == many[user]
    assert read(tmp_path / "a/preflight.json")["population_training_events"] == len(
        task.train
    )


def test_failed_partial_attempt_restarts_all_five_calls_and_has_no_partial_cache(
    tmp_path,
):
    task, dataset = build_task(tmp_path)
    client = FakeClient(
        fail=lambda stage, count, prompt: "{}"
        if stage == "sequence_ranking" and count == 1
        else None
    )
    run(task, dataset, tmp_path / "out", max_candidate_groups=1, client=client)
    assert client.counts["cluster_analysis"] == 3
    assert client.counts["cluster_ranking"] == 3
    assert client.counts["sequence_ranking"] == 3
    assert client.counts["profile_analysis"] == 2
    assert len(list((tmp_path / "out/dyta4rec_cache/groups").glob("*.json"))) == 2


@pytest.mark.parametrize("failed_split", ["val", "test", "all", "initialization"])
def test_total_split_failures_keep_artifacts_and_separate_conversion_coverage(
    tmp_path, failed_split
):
    task, dataset = build_task(tmp_path)
    task.val = task.val[
        task.val.candidate_group.eq(task.val.candidate_group.iloc[0])
    ].copy()
    task.test = task.test[
        task.test.candidate_group.eq(task.test.candidate_group.iloc[0])
    ].copy()
    # Break only the designated label-free candidate payload, after the analyses.
    if failed_split in ("val", "test"):
        getattr(task, failed_split)["item_title"] = "FAIL_GROUP"

    def fail(stage, count, prompt):
        if failed_split == "initialization" and stage == "memory_block":
            return "{}"
        if stage == "cluster_ranking" and (
            failed_split == "all" or "FAIL_GROUP" in prompt
        ):
            return "{}"
        return None

    result, client = run(task, dataset, tmp_path / "out", client=FakeClient(fail=fail))
    saved = pd.read_parquet(tmp_path / "out/predictions.parquet")
    pointwise = read(tmp_path / "out/metrics.json")
    assert (tmp_path / "out/manifest.json").exists()
    if failed_split in ("val", "all", "initialization"):
        assert result["pointwise_threshold"] is None
        assert pointwise["test_failure_as_negative"] is None
        assert saved.prediction.isna().all()
        if failed_split == "val":
            assert saved[saved.split == "test"].score.notna().all()
            assert result["coverage"]["test"]["scored_fraction"] == 1
            assert result["classification_coverage"]["test"]["scored_fraction"] == 0
    else:
        assert result["pointwise_threshold"] is not None
        assert saved[saved.split == "test"][["score", "prediction"]].isna().all().all()
        assert (
            pointwise["test_failure_as_negative"]["macro_by_user_group_mean"]["f1"] == 0
        )
    split = (
        "test"
        if failed_split == "initialization"
        else ("val" if failed_split == "all" else failed_split)
    )
    strict = result[f"{split}_failure_as_zero_group"]["macro_by_user_group_mean"]
    assert strict["ndcg@5"] == strict["recall@5"] == 0
    assert strict["failed_groups"] == 1


def test_mixed_failure_inclusive_threshold_and_metrics_against_brute_force(tmp_path):
    X = pd.DataFrame(
        {
            "user_id": ["u", "u", "u", "u", "v", "v", "v", "v"],
            "candidate_group": ["a", "a", "b", "b", "c", "c", "d", "d"],
        }
    )
    y = pd.Series([1, 0, 1, 0, 0, 1, 1, 0])
    scores = pd.Series([0.3, 0.2, np.nan, np.nan, 0.1, 0.2, np.nan, np.nan])
    selection = dyta4rec.select_threshold(X, y, scores, "candidate_group")
    thresholds = [np.nextafter(0.3, np.inf), 0.1, 0.2, 0.3]
    values = [
        pointwise_metrics_for_split(
            X=X, y=y, predictions=scores.ge(t), candidate_group_column="candidate_group"
        )["macro_by_user_group_mean"]["f1"]
        for t in thresholds
    ]
    assert selection["threshold"] == thresholds[int(np.argmax(values))]
    assert selection["metric_value"] == pytest.approx(max(values))
    task, dataset = build_task(tmp_path)
    extra = task.test.copy()
    extra["candidate_group"] = extra.candidate_group + "-bad"
    extra["item_title"] = "FAIL_GROUP"
    task.test = pd.concat([task.test, extra], ignore_index=True)
    result, client = run(
        task,
        dataset,
        tmp_path / "out",
        client=FakeClient(
            fail=lambda stage, count, prompt: "{}"
            if stage == "profile_ranking" and "FAIL_GROUP" in prompt
            else None
        ),
    )
    strict = result["test_failure_as_zero_group"]["macro_by_user_group_mean"]
    assert strict["ndcg@5"] == pytest.approx(
        result["test"]["macro_by_user_group_mean"]["ndcg@5"] / 2
    )
    assert result["coverage"]["test"]["scored_fraction"] == 0.5


def test_endpoints_and_identical_score_tie_convention():
    X = pd.DataFrame({"user_id": ["u", "u"], "candidate_group": ["g", "g"]})
    s = pd.Series([0.5, 0.5])
    assert (
        dyta4rec.select_threshold(X, pd.Series([1, 1]), s, "candidate_group")[
            "threshold"
        ]
        == 0.5
    )
    assert dyta4rec.select_threshold(X, pd.Series([0, 0]), s, "candidate_group")[
        "threshold"
    ] == np.nextafter(0.5, np.inf)


def test_preflight_is_local_and_same_output_resume(tmp_path, monkeypatch):
    task, dataset = build_task(tmp_path)
    monkeypatch.setattr(
        dyta4rec, "make_llm_client", lambda name: pytest.fail("No client construction")
    )
    provider = FakeEmbeddings()
    provider.metadata = lambda: pytest.fail("No embedding model loading")
    preflight, client = run(
        task,
        dataset,
        tmp_path / "preflight",
        preflight_only=True,
        embedding_provider=provider,
    )
    assert client.calls == provider.calls == []
    assert not (tmp_path / "preflight/manifest.json").exists()
    _, first = run(task, dataset, tmp_path / "out")
    _, second = run(task, dataset, tmp_path / "out")
    assert first.calls and second.calls == []
    with pytest.raises(ValueError, match="different evaluation"):
        run(task, dataset, tmp_path / "out", config=DyTA4RecConfig(seed=2))


@pytest.mark.parametrize(
    "case", ["random", "steam", "preference", "missing_timestamp", "global"]
)
def test_reject_unsupported_protocol_before_calls(tmp_path, case):
    task, dataset = build_task(tmp_path)
    if case == "random":
        task.manifest["splitter"]["shuffle"] = True
    if case == "steam":
        task.manifest["dataset"] = "steam"
    if case == "preference":
        task.manifest["target_source_column"] = "target_like"
    if case == "missing_timestamp":
        task.manifest["splitter"].pop("timestamp_column")
    if case == "global":
        task.manifest["splitter"]["group_column"] = None
    client = FakeClient()
    with pytest.raises(ValueError):
        run(task, dataset, tmp_path / "out", client=client)
    assert client.calls == []


def test_opt_in_registration_and_backbones(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(dyta4rec, "run_method", lambda *a, **k: calls.append(k) or {})
    for model in ("qwen3_8b", "qwen36_27b"):
        for size in ("smoke", "full"):
            name = f"dyta4rec_offline_rrf_litellm_{model}_{size}"
            METHOD_RUNNERS[name](None, tmp_path)
            assert calls[-1]["method_name"] == name
            assert calls[-1].get("max_candidate_groups") == (
                5 if size == "smoke" else None
            )
    from runners.in_distribution.interaction_prediction.run import DEFAULT_METHODS

    assert not any("dyta" in m for m in DEFAULT_METHODS)
