from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from beyond_click_sim.tasks import TaskSchema
from beyond_click_sim.tasks.cold_start import ColdStartTask

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.in_distribution.cold_start.methods import llm_yes_no  # noqa: E402
from runners.in_distribution.llm_error_budget import LLMErrorRateExceededError  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeChatCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response))]
        )


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        completions = FakeChatCompletions(responses)
        self.chat = SimpleNamespace(completions=completions)
        self.completions = completions


# ---------------------------------------------------------------------------
# Task fixture helper
# ---------------------------------------------------------------------------


def _ml1m_cold_task(*, history_user_id: str = "u1") -> ColdStartTask:
    """Minimal two-item ml-1m ColdStartTask.

    Cold user 'u1' has one history item in online_session_history. train
    contains only warm-user rows (no u1), matching the cold-start contract.
    """
    history_df = pd.DataFrame(
        [
            {
                "user_id": history_user_id,
                "item_id": "ih1",
                "item_title": "Toy Story",
                "item_genres": "Animation|Comedy",
                "rating": 5,
                "target": 1,
            }
        ]
    )
    test_df = pd.DataFrame(
        {
            "user_id": [history_user_id, history_user_id],
            "item_id": ["it1", "it2"],
            "candidate_group": ["g1", "g1"],
            "item_title": ["Lion King", "Godfather"],
            "item_genres": ["Animation", "Crime"],
            "sampled": [False, True],
            "target": [1, 0],
        }
    )
    train_df = pd.DataFrame(
        [
            {
                "user_id": "u_warm",
                "item_id": "iw1",
                "item_title": "Terminator",
                "item_genres": "Action",
                "rating": 4,
                "target": 1,
            }
        ]
    )

    return ColdStartTask(
        name="test_task",
        train=train_df,
        val=pd.DataFrame(columns=["user_id", "item_id", "target"]),
        test=test_df,
        online_session_history=history_df,
        k=1,
        schema=TaskSchema(
            target_column="target",
            candidate_group_column="candidate_group",
            sampled_column="sampled",
            feature_columns=("user_id", "item_id"),
        ),
        manifest={
            "dataset": "ml-1m",
            "dataset_version": "v1",
            "splitter": {"seed": 42},
        },
    )


# ---------------------------------------------------------------------------
# Batch scoring — regression-locks current (unchanged) behavior
# ---------------------------------------------------------------------------


def test_cold_start_llm_runner_batch_scores_one_call_per_group(
    tmp_path: Path, monkeypatch
) -> None:
    client = FakeClient(["C1: yes\nC2: no"])
    monkeypatch.setattr(llm_yes_no, "make_llm_client", lambda _: client)

    task = _ml1m_cold_task()
    result = llm_yes_no.run_method(
        task,
        tmp_path,
        method_name="llm_yes_no_cold_start_test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
        scoring="batch",
    )

    assert result["scoring"] == "batch"
    assert result["scored_rows"] == 2
    assert result["requested_rows"] == 2
    assert result["llm_errors"] == 0
    assert len(client.completions.calls) == 1

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"]["class"] == "LLMInteractionYesNoScorer"
    assert manifest["scorer"]["fit_on"] == "online_session_history"
    assert manifest["scorer"]["prompt_style"] == "batch"

    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert predictions["score"].tolist() == [1.0, 0.0]
    assert predictions["prediction"].tolist() == [True, False]


# ---------------------------------------------------------------------------
# Itemwise scoring — default mode
# ---------------------------------------------------------------------------


def test_cold_start_llm_runner_itemwise_scores_one_call_per_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    """Itemwise (the run_method default) must issue one LLM call per candidate
    row, not one call per ~20-item candidate group, and must record the mode
    in manifest.json for provenance."""

    client = FakeClient(["yes", "no"])
    monkeypatch.setattr(llm_yes_no, "make_llm_client", lambda _: client)

    task = _ml1m_cold_task()
    result = llm_yes_no.run_method(
        task,
        tmp_path,
        method_name="llm_yes_no_cold_start_itemwise_test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
    )

    assert result["scoring"] == "itemwise"
    assert result["scored_rows"] == 2
    assert result["requested_rows"] == 2
    assert result["llm_errors"] == 0
    assert len(client.completions.calls) == 2

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"]["prompt_style"] == "itemwise"

    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert sorted(predictions["score"].tolist()) == [0.0, 1.0]


def test_cold_start_llm_runner_writes_run_metadata_columns(
    tmp_path: Path, monkeypatch
) -> None:
    """Cold-start manifests carry k/m/sampling_seed/strategy; predictions.parquet
    must repeat them as columns so concatenated runs cannot silently merge
    candidate groups that share an id across the grid."""

    client = FakeClient(["yes", "no"])
    monkeypatch.setattr(llm_yes_no, "make_llm_client", lambda _: client)

    task = _ml1m_cold_task()
    task.manifest.update(
        {
            "k": 3,
            "sampling_seed": 2,
            "negative_strategy": "popularity_matched",
            "sampler": {"class": "PerTargetCandidateSampler", "negative_ratio": 9},
        }
    )
    llm_yes_no.run_method(
        task,
        tmp_path,
        method_name="llm_yes_no_cold_start_metadata_test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
    )

    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert predictions["k"].unique().tolist() == [3]
    assert predictions["negative_ratio"].unique().tolist() == [9]
    assert predictions["sampling_seed"].unique().tolist() == [2]
    assert predictions["negative_strategy"].unique().tolist() == [
        "popularity_matched"
    ]


def test_cold_start_llm_runner_rejects_invalid_scoring(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(llm_yes_no, "make_llm_client", lambda _: FakeClient([]))
    task = _ml1m_cold_task()

    with pytest.raises(ValueError, match="scoring"):
        llm_yes_no.run_method(
            task,
            tmp_path,
            method_name="test_bad_scoring",
            client_name="fake",
            model="fake-model",
            max_candidate_groups=None,
            scoring="bogus",
        )


def test_cold_start_llm_runner_raises_without_candidate_group_column(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(llm_yes_no, "make_llm_client", lambda _: FakeClient([]))
    task = _ml1m_cold_task()
    task = ColdStartTask(
        name="test_task",
        train=task.train,
        val=task.val,
        test=task.test,
        online_session_history=task.online_session_history,
        k=task.k,
        schema=TaskSchema(
            target_column="target",
            candidate_group_column=None,
            sampled_column="sampled",
            feature_columns=("user_id", "item_id"),
        ),
        manifest=task.manifest,
    )

    with pytest.raises(ValueError, match="candidate_group_column"):
        llm_yes_no.run_method(
            task,
            tmp_path,
            method_name="test_no_group_col",
            client_name="fake",
            model="fake-model",
            max_candidate_groups=None,
        )


# ---------------------------------------------------------------------------
# Wrapper functions — smoke & full kwargs (batch vs itemwise)
# ---------------------------------------------------------------------------


def test_cold_start_llm_qwen_smoke_and_full_wrappers(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run_method(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(llm_yes_no, "run_method", _fake_run_method)
    task = SimpleNamespace()

    llm_yes_no.run_qwen36_27b_smoke(task, tmp_path)
    llm_yes_no.run_qwen36_27b_full(task, tmp_path)
    llm_yes_no.run_qwen36_27b_itemwise_smoke(task, tmp_path)
    llm_yes_no.run_qwen36_27b_itemwise_full(task, tmp_path)
    llm_yes_no.run_qwen36_35b_a3b_smoke(task, tmp_path)
    llm_yes_no.run_qwen36_35b_a3b_full(task, tmp_path)
    llm_yes_no.run_qwen36_35b_a3b_itemwise_smoke(task, tmp_path)
    llm_yes_no.run_qwen36_35b_a3b_itemwise_full(task, tmp_path)

    assert calls == [
        {
            "method_name": "llm_yes_no_vllm_qwen36_27b_smoke",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": 25,
            "max_workers": llm_yes_no.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "scoring": "batch",
        },
        {
            "method_name": "llm_yes_no_vllm_qwen36_27b_full",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": None,
            "max_workers": llm_yes_no.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "scoring": "batch",
        },
        {
            "method_name": "llm_yes_no_vllm_qwen36_27b_itemwise_smoke",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": 25,
            "max_workers": llm_yes_no.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "scoring": "itemwise",
        },
        {
            "method_name": "llm_yes_no_vllm_qwen36_27b_itemwise_full",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": None,
            "max_workers": llm_yes_no.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "scoring": "itemwise",
        },
        {
            "method_name": "llm_yes_no_vllm_qwen36_35b_a3b_smoke",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-35B-A3B",
            "max_candidate_groups": 25,
            "max_workers": llm_yes_no.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "scoring": "batch",
        },
        {
            "method_name": "llm_yes_no_vllm_qwen36_35b_a3b_full",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-35B-A3B",
            "max_candidate_groups": None,
            "max_workers": llm_yes_no.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "scoring": "batch",
        },
        {
            "method_name": "llm_yes_no_vllm_qwen36_35b_a3b_itemwise_smoke",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-35B-A3B",
            "max_candidate_groups": 25,
            "max_workers": llm_yes_no.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "scoring": "itemwise",
        },
        {
            "method_name": "llm_yes_no_vllm_qwen36_35b_a3b_itemwise_full",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-35B-A3B",
            "max_candidate_groups": None,
            "max_workers": llm_yes_no.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "scoring": "itemwise",
        },
    ]


def test_cold_start_llm_runner_fails_fast_and_persists_partial_errors(
    tmp_path: Path, monkeypatch
) -> None:
    client = FakeClient(["maybe", "maybe"])
    monkeypatch.setattr(llm_yes_no, "make_llm_client", lambda _: client)
    task = _ml1m_cold_task()

    with pytest.raises(LLMErrorRateExceededError):
        llm_yes_no.run_method(
            task,
            tmp_path,
            method_name="llm_yes_no_fail_fast_test",
            client_name="fake",
            model="fake-model",
            max_candidate_groups=None,
            max_llm_attempts=1,
            max_workers=1,
            scoring="itemwise",
            max_error_rate=0.10,
            min_groups_before_check=1,
        )

    assert (tmp_path / "llm_errors.jsonl").exists()
    errors = [
        json.loads(line)
        for line in (tmp_path / "llm_errors.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(errors) == 1
    assert "raw_response='maybe'" in errors[0]["errors"][0]

    assert not (tmp_path / "predictions.parquet").exists()
    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "metrics.json").exists()
    assert not (tmp_path / "metrics_ranking.json").exists()


def _two_group_cold_task() -> ColdStartTask:
    """Two candidate groups (g1, g2) of two rows each for user u1."""
    base = _ml1m_cold_task()
    test_df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1"],
            "item_id": ["it1", "it2", "it3", "it4"],
            "candidate_group": ["g1", "g1", "g2", "g2"],
            "item_title": ["Lion King", "Godfather", "Toy Story", "Heat"],
            "item_genres": ["Animation", "Crime", "Animation", "Crime"],
            "sampled": [False, True, False, True],
            "target": [1, 0, 1, 0],
        }
    )
    return ColdStartTask(
        name=base.name,
        train=base.train,
        val=base.val,
        test=test_df,
        online_session_history=base.online_session_history,
        k=base.k,
        schema=base.schema,
        manifest=base.manifest,
    )


def test_cold_start_llm_runner_reports_failure_aware_metric_variants(
    tmp_path: Path, monkeypatch
) -> None:
    """One group parses, one fails: parsed-only metrics exclude the failed
    group while the failure-aware variants penalize it."""

    # batch mode: one call per group — g1 parses, g2's response is unparseable
    client = FakeClient(["C1: yes\nC2: no", "maybe"])
    monkeypatch.setattr(llm_yes_no, "make_llm_client", lambda _: client)
    task = _two_group_cold_task()

    result = llm_yes_no.run_method(
        task,
        tmp_path,
        method_name="llm_yes_no_failure_variants_test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
        scoring="batch",
        max_error_rate=0.5,
        min_groups_before_check=1,
    )

    assert result["llm_errors"] == 1
    assert result["scored_rows"] == 2
    assert result["requested_rows"] == 4

    coverage = result["coverage"]
    assert coverage["requested_rows"] == 4
    assert coverage["scored_rows"] == 2
    assert coverage["failed_rows"] == 2
    assert coverage["scored_fraction"] == 0.5

    # parsed-only: only g1 rows, predicted [yes, no] against targets [1, 0]
    assert result["test"]["micro"]["recall"] == 1.0
    # failure-as-negative: failed rows count as "no" -> the g2 positive is missed
    assert result["test_failure_as_negative"]["micro"]["recall"] == 0.5
    assert result["test_failure_as_negative"]["micro"]["precision"] == 1.0

    ranking = json.loads(
        (tmp_path / "metrics_ranking.json").read_text(encoding="utf-8")
    )
    # parsed-only ranking: g1 perfectly ranked
    assert ranking["test"]["macro_by_group"]["hit_rate@1"] == 1.0
    # failure-as-zero: g2 contributes zero utility -> macro average drops to 0.5
    assert ranking["test_failure_as_zero_group"]["macro_by_group"]["hit_rate@1"] == 0.5
    assert ranking["coverage"]["failed_fraction"] == 0.5


# ---------------------------------------------------------------------------
# Summary visibility
# ---------------------------------------------------------------------------


def _summary_cold_task() -> ColdStartTask:
    """ColdStartTask whose history and candidate items carry summaries."""
    base = _ml1m_cold_task()
    history_df = base.online_session_history.copy()
    history_df["item_summary"] = "PROFILE SUMMARY MARKER"
    test_df = base.test.copy()
    test_df["item_summary"] = [
        "CANDIDATE SUMMARY MARKER ONE",
        "CANDIDATE SUMMARY MARKER TWO",
    ]
    train_df = base.train.copy()
    train_df["item_summary"] = "WARM TRAIN SUMMARY"
    val_df = base.val.copy()
    if "item_summary" not in val_df.columns:
        val_df = val_df.assign(item_summary=pd.Series(dtype=object))
    manifest = dict(base.manifest)
    manifest["item_enrichment"] = {
        "movie_summaries": {
            "enabled": True,
            "canonical_column": "summary",
            "task_column": "item_summary",
            "source_sha256": "fake-sha256",
        }
    }
    return ColdStartTask(
        name=base.name,
        train=train_df,
        val=val_df,
        test=test_df,
        online_session_history=history_df,
        k=base.k,
        schema=base.schema,
        manifest=manifest,
    )


def test_cold_start_llm_runner_default_summary_visibility_is_none(
    tmp_path: Path, monkeypatch
) -> None:
    client = FakeClient(["C1: yes\nC2: no"])
    monkeypatch.setattr(llm_yes_no, "make_llm_client", lambda _: client)

    task = _summary_cold_task()
    llm_yes_no.run_method(
        task,
        tmp_path,
        method_name="llm_yes_no_summary_default_test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
        scoring="batch",
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"]["summary_visibility"] == "none"
    assert "item_summary" not in (
        manifest["scorer"]["prompt_columns"]["history_description_columns"]
    )
    assert "item_summary" not in (
        manifest["scorer"]["prompt_columns"]["candidate_description_columns"]
    )
    prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "SUMMARY MARKER" not in prompt


def test_cold_start_llm_runner_sumboth_shows_profile_and_candidate_summaries(
    tmp_path: Path, monkeypatch
) -> None:
    """summary_visibility='both': history summaries come from the k-item
    profile (online_session_history), candidate summaries from candidates."""

    client = FakeClient(["C1: yes\nC2: no"])
    monkeypatch.setattr(llm_yes_no, "make_llm_client", lambda _: client)

    task = _summary_cold_task()
    llm_yes_no.run_method(
        task,
        tmp_path,
        method_name="llm_yes_no_sumboth_test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
        summary_visibility="both",
        scoring="batch",
    )

    prompt = client.completions.calls[0]["messages"][1]["content"]
    # Profile item summary appears in the history section (H1).
    assert "H1. item_title: Toy Story" in prompt
    assert "PROFILE SUMMARY MARKER" in prompt
    # Candidate summaries appear on the candidate cards.
    assert "CANDIDATE SUMMARY MARKER ONE" in prompt
    assert "CANDIDATE SUMMARY MARKER TWO" in prompt

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"]["summary_visibility"] == "both"
    assert "item_summary" in (
        manifest["scorer"]["prompt_columns"]["history_description_columns"]
    )
    assert "item_summary" in (
        manifest["scorer"]["prompt_columns"]["candidate_description_columns"]
    )
    assert manifest["scorer"]["item_summaries"]["history_item_summaries"] is True
    assert manifest["scorer"]["item_summaries"]["candidate_item_summaries"] is True


def test_cold_start_llm_runner_summaries_require_canonical_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    """The base fixture has no item_summary columns/provenance: requesting
    summaries must fail cleanly instead of silently degrading."""

    monkeypatch.setattr(llm_yes_no, "make_llm_client", lambda _: FakeClient([]))
    task = _ml1m_cold_task()

    with pytest.raises(ValueError, match="item_summary"):
        llm_yes_no.run_method(
            task,
            tmp_path,
            method_name="llm_yes_no_summary_no_provenance_test",
            client_name="fake",
            model="fake-model",
            max_candidate_groups=None,
            summary_visibility="both",
            scoring="batch",
        )
