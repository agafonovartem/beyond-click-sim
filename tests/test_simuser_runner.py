from copy import deepcopy
import json

import pandas as pd
import pytest

from beyond_click_sim.scorers.simuser import SimUSERConfig
from beyond_click_sim.scorers.simuser.state import identity
from runners.in_distribution.interaction_prediction.methods import (
    METHOD_RUNNERS,
    simuser_yes_no,
)
from runners.in_distribution.interaction_prediction.methods.common import (
    ranking_metrics_for_split,
)
from test_llmsrec_syn_runner import build_task
from test_simuser_scorer import FakeClient, FakeEmbeddings


def task_fixture(tmp_path):
    task, data = build_task(tmp_path)
    # Catalog-only enrichment: no task mutation or fixture outcome consumption.
    items = data.load_items()
    items["summary"] = items.title.map(lambda t: f"A frozen plot description of {t}.")
    items.to_parquet(data.items_path)
    return task, data


def run(task, data, out, **kwargs):
    client = kwargs.pop("client", FakeClient())
    result = simuser_yes_no.run_method(
        task,
        out,
        client_name="fake",
        client=client,
        embedding_client=kwargs.pop("embedding_client", FakeEmbeddings()),
        model="fake",
        model_revision="frozen",
        canonical_dataset=data,
        config=kwargs.pop("config", SimUSERConfig(initialization_attempts=1)),
        max_llm_attempts=kwargs.pop("max_llm_attempts", 1),
        **kwargs,
    )
    return result, client


def read(path):
    return json.loads(path.read_text())


def test_both_splits_same_binary_scores_metrics_complete_coverage_no_extra_calls(
    tmp_path,
):
    task, data = task_fixture(tmp_path)
    before = deepcopy(task)
    out = tmp_path / "run"
    result, client = run(task, data, out)
    frame = pd.read_parquet(out / "predictions.parquet")
    assert set(frame.split) == {"val", "test"}
    assert (
        frame.score.astype(int)
        .rename("prediction")
        .equals(frame.prediction.astype(int))
    )
    assert frame.score.eq(0).any() and frame.score.eq(1).any()
    ranking = read(out / "metrics_ranking.json")
    manifest = read(out / "manifest.json")
    preflight = read(out / "preflight.json")
    count = sum(getattr(task, s).candidate_group.nunique() for s in ("val", "test"))
    users = len(set(task.val.user_id) | set(task.test.user_id))
    assert len(client.calls) == 17 * users + 3 * count
    assert preflight["population_users"] == 8 and preflight["users"] == users
    assert manifest["successful_metrics_equal_strict"]
    assert manifest["decision_rule"]["threshold"] is None
    for split in ("val", "test"):
        rows = frame.loc[frame.split.eq(split)]
        expected = ranking_metrics_for_split(
            X=rows,
            y=rows.target,
            scores=rows.score,
            candidate_group_column="candidate_group",
        )
        assert ranking[split] == expected
        for metric in ("ndcg@5", "recall@5"):
            assert ranking[split]["macro_by_user_group_mean"][metric] == pytest.approx(
                ranking[f"{split}_failure_as_zero_group"]["macro_by_user_group_mean"][
                    metric
                ]
            )
        for metric in ("f1", "precision", "recall"):
            assert result[split]["macro_by_user_group_mean"][metric] == pytest.approx(
                result[f"{split}_failure_as_negative"]["macro_by_user_group_mean"][
                    metric
                ]
            )
        assert rows.task_row_index.tolist() == getattr(task, split).index.tolist()
    for split in ("train", "val", "test"):
        pd.testing.assert_frame_equal(getattr(task, split), getattr(before, split))


def test_heldout_labels_ratings_timestamps_flags_demographics_do_not_change_inference(
    tmp_path,
):
    task, data = task_fixture(tmp_path)
    _, first = run(task, data, tmp_path / "first")
    changed = deepcopy(task)
    for split in ("val", "test"):
        frame = getattr(changed, split)
        frame[changed.schema.target_column] = 1 - frame[changed.schema.target_column]
        frame["rating"] = -999
        frame["timestamp"] = "hidden future"
        frame["user_age"] = "hidden demographics"
        frame[changed.schema.sampled_column] = ~frame[changed.schema.sampled_column]
    _, second = run(changed, data, tmp_path / "second")
    assert first.calls == second.calls
    one = pd.read_parquet(tmp_path / "first/predictions.parquet")
    two = pd.read_parquet(tmp_path / "second/predictions.parquet")
    pd.testing.assert_frame_equal(
        one[["score", "prediction"]], two[["score", "prediction"]]
    )
    assert (
        read(tmp_path / "first/manifest.json")["initialization"][
            "snapshot_fingerprints"
        ]
        == read(tmp_path / "second/manifest.json")["initialization"][
            "snapshot_fingerprints"
        ]
    )


def test_failed_group_and_successful_all_no_have_different_ranking_utilities(tmp_path):
    task, data = task_fixture(tmp_path)
    client = FakeClient(
        all_no=True, fail=lambda s, n, p: "invalid" if s == "watch" and n == 1 else None
    )
    result, _ = run(task, data, tmp_path / "out", client=client, splits=("test",))
    frame = pd.read_parquet(tmp_path / "out/predictions.parquet")
    assert frame.score.isna().any() and frame.score.eq(0).any()
    assert frame.loc[frame.score.isna(), "prediction"].isna().all()
    ranking = read(tmp_path / "out/metrics_ranking.json")
    strict = ranking["test_failure_as_zero_group"]["macro_by_user_group_mean"]
    assert strict["failed_groups"] == 1
    assert ranking["test"]["macro_by_user_group_mean"]["ndcg@5"] > 0
    assert strict["ndcg@5"] == pytest.approx(
        ranking["test"]["macro_by_user_group_mean"]["ndcg@5"] / 2
    )
    assert not read(tmp_path / "out/manifest.json")["successful_metrics_equal_strict"]
    assert "test_failure_as_negative" in result


def test_initialization_failure_preserves_requested_population(tmp_path):
    task, data = task_fixture(tmp_path)
    result, client = run(
        task,
        data,
        tmp_path / "out",
        splits=("test",),
        client=FakeClient(
            fail=lambda s, n, p: "invalid" if s == "taste" and n == 1 else None
        ),
    )
    manifest = read(tmp_path / "out/manifest.json")
    assert manifest["initialization"]["initialized_users"] == 1
    failed_user = next(iter(manifest["initialization"]["unavailable_users"]))
    frame = pd.read_parquet(tmp_path / "out/predictions.parquet")
    assert frame.loc[frame.user_id.map(identity).eq(failed_user), "score"].isna().all()
    assert result["coverage"]["test"]["requested_rows"] == len(task.test)
    assert client.counts["watch"] == 1


def test_all_failed_writes_missing_artifacts_and_zero_ranking(tmp_path):
    task, data = task_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="no valid scores"):
        run(
            task,
            data,
            tmp_path / "out",
            client=FakeClient(fail=lambda s, n, p: "invalid"),
        )
    frame = pd.read_parquet(tmp_path / "out/predictions.parquet")
    assert frame.score.isna().all() and frame.prediction.isna().all()
    assert read(tmp_path / "out/manifest.json")["status"] == "all_groups_failed"
    ranking = read(tmp_path / "out/metrics_ranking.json")
    for split in ("val", "test"):
        assert ranking[split] is None
        assert (
            ranking[f"{split}_failure_as_zero_group"]["macro_by_user_group_mean"][
                "ndcg@5"
            ]
            == 0
        )


def test_five_whole_group_attempts_without_nested_stage_retries(tmp_path):
    task, data = task_fixture(tmp_path)
    client = FakeClient(
        fail=lambda s, n, p: "invalid" if s == "watch" and n <= 4 else None
    )
    run(
        task,
        data,
        tmp_path / "out",
        client=client,
        splits=("test",),
        max_candidate_groups=1,
        max_llm_attempts=5,
    )
    assert client.counts["self_ask"] == client.counts["watch"] == 5
    assert client.counts["check"] == 1
    assert read(tmp_path / "out/manifest.json")["status"] == "complete"


def test_preflight_constructs_no_clients_and_retains_population(tmp_path, monkeypatch):
    task, data = task_fixture(tmp_path)
    monkeypatch.setattr(
        simuser_yes_no,
        "make_llm_client",
        lambda *_: pytest.fail("No client creation during preflight"),
    )
    result = simuser_yes_no.run_method(
        task,
        tmp_path / "out",
        canonical_dataset=data,
        preflight_only=True,
        splits=("test",),
        max_candidate_groups=1,
    )
    assert result["users"] == 1 and result["population_users"] == 8
    assert result["candidate_groups"] == 1
    assert result["uncached_successful_calls"] == {
        "initialization": 17,
        "groups_min": 3,
        "groups_max": 7,
        "additional_classification_or_ranking": 0,
    }
    assert not (tmp_path / "out/predictions.parquet").exists()
    assert result["configuration"]["seed"] == task.manifest["sampler"]["seed"]


@pytest.mark.parametrize(
    "mutation", ["dataset", "random", "train_rating", "summary", "train_sampled"]
)
def test_protocol_preparation_errors_before_clients(tmp_path, monkeypatch, mutation):
    task, data = task_fixture(tmp_path)
    if mutation == "dataset":
        task.manifest["dataset"] = "steam"
    elif mutation == "random":
        task.manifest["splitter"]["shuffle"] = True
    elif mutation == "train_rating":
        task.train.loc[task.train.index[0], "rating"] = -999
    elif mutation == "train_sampled":
        task.train.loc[task.train.index[0], task.schema.sampled_column] = True
    else:
        items = data.load_items().drop(columns="summary")
        items.to_parquet(data.items_path)
    monkeypatch.setattr(
        simuser_yes_no,
        "make_llm_client",
        lambda *_: pytest.fail("Prepare before client"),
    )
    with pytest.raises(ValueError):
        simuser_yes_no.run_method(task, tmp_path / "out", canonical_dataset=data)


def test_opt_in_registry_wrappers(monkeypatch):
    calls = []
    monkeypatch.setattr(
        simuser_yes_no, "run_method", lambda *args, **kwargs: calls.append(kwargs)
    )
    for model in ("qwen3_8b", "qwen36_27b"):
        for scope in ("smoke", "full"):
            name = f"simuser_zero_text_only_litellm_{model}_{scope}"
            METHOD_RUNNERS[name](None, None)
            assert calls[-1]["method_name"] == name
            assert calls[-1].get("max_candidate_groups") == (
                5 if scope == "smoke" else None
            )
            assert calls[-1].get("model", "Qwen/Qwen3-8B") == (
                "Qwen/Qwen3-8B" if model == "qwen3_8b" else "Qwen/Qwen3.6-27B"
            )
