from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import re
from threading import Lock
from types import SimpleNamespace

import pandas as pd
import pytest

from beyond_click_sim.scorers.dyta4rec import DyTA4RecConfig, DyTA4RecScorer
from beyond_click_sim.scorers.dyta4rec import prompts
from beyond_click_sim.scorers.recagent.memory import RecAgentMemory
from test_recagent_scorer import training, candidates


class FakeEmbeddings:
    def __init__(self, revision="fake-v1"):
        self.revision, self.calls = revision, []

    def metadata(self):
        return {
            "model": "fake",
            "revision": self.revision,
            "dimensions": 3,
            "precision": "float32",
            "normalize_embeddings": False,
            "max_seq_length": 384,
        }

    def embed(self, text):
        self.calls.append(text)
        return [1, len(text) % 7 / 10, len(text) % 3 / 10]


class FakeClient:
    def __init__(self, fail=None, flags=("yes", "yes")):
        self.calls, self.counts = [], Counter()
        self.fail, self.flags, self.lock = fail, flags, Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        prompt = kwargs["messages"][-1]["content"]
        stage = re.match(r"\[dyta4rec-prompts-v1:(.*?)\]", prompt)[1]
        inputs = json.loads(prompt.split("INPUT_JSON:\n")[1])
        with self.lock:
            self.calls.append(deepcopy(kwargs))
            self.counts[stage] += 1
            count = self.counts[stage]
        override = self.fail(stage, count, prompt) if self.fail else None
        text = override if override is not None else self.response(stage, inputs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=text), finish_reason="stop"
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=40, completion_tokens=10, total_tokens=50
            ),
        )

    def response(self, stage, inputs):
        if stage == "importance":
            return "8"
        if stage in ("memory_block", "memory_merge"):
            rows = inputs.get("events", inputs.get("summaries"))
            return json.dumps(
                {
                    "text": "Observed patterns from "
                    + ", ".join(r["ref"] for r in rows),
                    "evidence": [r["ref"] for r in rows],
                }
            )
        if stage == "long_profile":
            return json.dumps(
                {
                    **{
                        f: f"Tentative {f}, evidence remains limited."
                        for f in prompts.PROFILE_FIELDS
                    },
                    "evidence": [inputs["summary"]["ref"]],
                }
            )
        if stage.startswith("recent_"):
            return json.dumps(
                {
                    "text": "The user watches comedies. Interests may be varied.",
                    "evidence": [inputs["recent_history"][-1]["ref"]],
                }
            )
        if stage.endswith("analysis"):
            return json.dumps(
                {
                    "analysis": f"Observed genre clusters ({stage}).",
                    "pattern_flag": self.flags[0],
                }
            )
        observation = json.loads(inputs["candidate_observation"])
        labels = [c["label"] for c in observation["candidates"]]
        # Disagreeing branches, with a symmetric exact RRF tie when cluster is low.
        ranking = list(reversed(labels)) if stage == "profile_ranking" else labels
        result = {"ranking": ranking}
        if stage == "sequence_ranking":
            result["pattern_flag"] = self.flags[1]
        return json.dumps(result)


def scorer(n=12, client=None, **kwargs):
    X, y, events = training(n)
    return DyTA4RecScorer(
        client=client or FakeClient(),
        model="fake-model",
        model_revision="fake-revision",
        training_events=events,
        embedding_provider=kwargs.pop("embedding_provider", FakeEmbeddings()),
        **kwargs,
    ).fit(X, y)


def test_fit_preflight_no_generation_or_embedding_resolution():
    provider = FakeEmbeddings()
    provider.metadata = lambda: pytest.fail("No embedding resolution in preflight")
    s = scorer(501, embedding_provider=provider)
    p = s.preflight(candidate_groups=2)
    assert p["per_user"]["u"] == {
        "blocks": 11,
        "merges": 3,
        "summary_records": 14,
        "llm_calls": 31,
    }
    assert p["uncached_group_llm_calls"] == 10
    assert s.client.calls == provider.calls == []


@pytest.mark.parametrize(
    "n,blocks,merges", [(1, 1, 0), (50, 1, 0), (51, 2, 1), (501, 11, 3)]
)
def test_summary_tree_provenance_and_exact_initialization_calls(n, blocks, merges):
    s = scorer(n)
    s.build_agents()
    assert not s.initialization_errors
    state = s.agent_snapshots["u"].to_dict()
    assert len(state["records"]) == blocks + merges
    assert len(s.client.calls) == 2 * (blocks + merges) + 3
    assert len(state["records"][-1]["source_event_refs"]) == n
    assert state["reference_time"] == state["evidence"]["cutoff"] + 1
    assert state["evidence"]["demographics"]["age_category"] == "18–24"
    assert all(r["importance"] == pytest.approx(0.72) for r in state["records"])
    assert all(
        r["accessed_at"] <= state["evidence"]["cutoff"] for r in state["records"]
    )
    assert len(s.embeddings.provider.calls) == blocks + merges
    assert all("event_id" not in text for text in s.embeddings.provider.calls)


def test_exact_five_call_graph_latest_ten_and_icl_no_extra_classification():
    s = scorer(14, client=FakeClient(flags=("no", "no"))).build_agents()
    before = s.agent_snapshots["u"]
    s.client.calls.clear()
    result = s.score(candidates())
    stages = [
        re.match(r"\[dyta4rec-prompts-v1:(.*?)\]", c["messages"][-1]["content"])[1]
        for c in s.client.calls
    ]
    assert stages == [
        "cluster_analysis",
        "cluster_ranking",
        "sequence_ranking",
        "profile_analysis",
        "profile_ranking",
    ]
    assert result.iloc[0] == result.iloc[2]
    assert s.agent_snapshots["u"] == before
    for call in s.client.calls:
        assert (call["model"], call["temperature"], call["top_p"], call["seed"]) == (
            "fake-model",
            0,
            1,
            0,
        )
        text = call["messages"][-1]["content"]
        assert all(
            forbidden not in text
            for forbidden in [
                "98765",
                "SECRET_FUTURE_TIME",
                "DO_NOT_USE_ZIP",
                "is_sampled",
                "event_id",
            ]
        )
    sequence = json.loads(
        s.client.calls[2]["messages"][-1]["content"].split("INPUT_JSON:\n")[1]
    )
    assert [e["position"] for e in sequence["full_recent_history"]] == list(
        range(5, 15)
    )
    assert [
        x["next_observed_interaction"]["position"]
        for x in sequence["successive_next_item_examples"]
    ] == [12, 13, 14]
    assert sequence["latest_interaction_reminder"]["position"] == 14
    profile = json.loads(
        s.client.calls[-1]["messages"][-1]["content"].split("INPUT_JSON:\n")[1]
    )
    assert all(
        f in profile
        for f in [
            "long_term_profile",
            "observed_statistics",
            "recorded_demographics",
            "recent_behavior",
            "recent_interests",
            "retrieved_long_term_memory",
        ]
    )


@pytest.mark.parametrize("n,k,expected", [(1, 3, 0), (2, 3, 1), (10, 0, 0), (10, 3, 3)])
def test_short_and_genuine_zero_demonstration_paths(n, k, expected):
    s = scorer(n, config=DyTA4RecConfig(icl_examples=k)).build_agents()
    recent = s.agent_snapshots["u"].to_dict()["recent"]
    inputs = prompts.sequential_inputs(recent, k, "{}")
    assert len(inputs["successive_next_item_examples"]) == expected
    assert len(inputs["initial_context"]) == n - expected


def test_resume_fusion_and_label_boundaries(tmp_path):
    s = scorer(cache_dir=tmp_path).build_agents()
    original = s.score(candidates())
    resumed = scorer(cache_dir=tmp_path).build_agents()
    changed = candidates().assign(
        target=-100, rating=-777, timestamp="CHANGED", is_sampled=False
    )
    pd.testing.assert_series_equal(resumed.score(changed), original)
    assert resumed.client.calls == []
    bc = scorer(cache_dir=tmp_path, config=DyTA4RecConfig(fusion="bc")).build_agents()
    assert bc.score(changed).notna().all()
    assert bc.client.calls == []  # Pure re-fusion of complete branches.
    changed.loc[0, "item_title"] = "Changed candidate metadata"
    resumed.score(changed)
    assert len(resumed.client.calls) == 5
    assert resumed.agent_snapshots["u"] == s.agent_snapshots["u"]


@pytest.mark.parametrize(
    "change", ["model", "seed", "embedding", "evidence", "prompt", "icl"]
)
def test_cache_invalidations(tmp_path, change):
    scorer(cache_dir=tmp_path).build_agents().score(candidates())
    options = {"cache_dir": tmp_path}
    if change == "seed":
        options["config"] = DyTA4RecConfig(seed=2)
    elif change == "embedding":
        options["embedding_provider"] = FakeEmbeddings("fake-v2")
    elif change == "icl":
        options["config"] = DyTA4RecConfig(icl_examples=0)
    s = scorer(**options)
    if change == "model":
        s.model_revision = "different-model-snapshot"
    elif change == "prompt":
        s._sources["dyta4rec/prompts.py"] = "changed"
    elif change == "evidence":
        X, y, events = training(12)
        X.loc[X.index[0], "rating"] = 5
        s.fit(X, y)
    s.build_agents().score(candidates())
    assert len(s.client.calls) == 10


def test_checkpoint_stage_retries_and_no_lifecycle_calls(tmp_path, monkeypatch):
    for method in ("observe", "read", "save", "reinforce"):
        monkeypatch.setattr(
            RecAgentMemory, method, lambda *a, **k: pytest.fail("Forbidden lifecycle")
        )
    first = scorer(
        cache_dir=tmp_path,
        client=FakeClient(
            fail=lambda stage, count, prompt: "bad"
            if stage == "recent_interests"
            else None
        ),
    ).build_agents()
    assert "u" in first.initialization_errors
    assert first.client.counts["recent_interests"] == 5
    second = scorer(cache_dir=tmp_path).build_agents()
    assert second.client.counts == {"recent_interests": 1}
    second.score(candidates())
    snapshot = second.agent_snapshots["u"].to_dict()
    snapshot["records"][0]["accessed_at"] = 999999999
    assert (
        second.agent_snapshots["u"].to_dict()["records"][0]["accessed_at"] != 999999999
    )


def test_concurrent_reordered_groups_do_not_mutate_snapshot():
    s = scorer().build_agents()
    before = s.agent_snapshots["u"]
    groups = [
        candidates(f"g{i}").assign(item_title=lambda x: x.item_title + str(i))
        for i in range(4)
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(s.score, reversed(groups)))
    assert all(r.notna().all() for r in results)
    assert len(s.client.calls) == 5 + 20
    assert s.agent_snapshots["u"] == before


def test_oversized_context_is_rejected_without_call_or_truncation():
    s = scorer(config=DyTA4RecConfig(context_tokens=100)).build_agents()
    assert s.initialization_errors and not s.client.calls


@pytest.mark.parametrize(
    "text,flag",
    [
        ('{"ranking":["C1"]}', False),
        ('{"ranking":["C1","C1"]}', False),
        ('{"ranking":["C1","C3"]}', False),
        ('{"ranking":["C1","C2"],"pattern_flag":"high"}', True),
        ('{"ranking":["C1","C2"]}', True),
        ("```json\n{}\n```", False),
        ('{"ranking":["C1","C2"],"ranking":["C2","C1"]}', False),
    ],
)
def test_strict_ranking_parser(text, flag):
    with pytest.raises(ValueError):
        prompts.parse_ranking(text, ["C1", "C2"], flag=flag)


def test_evidence_reference_validation():
    with pytest.raises(ValueError, match="evidence"):
        prompts.parse_summary('{"text":"A claim", "evidence":["E99"]}', ["E1"], 100)
