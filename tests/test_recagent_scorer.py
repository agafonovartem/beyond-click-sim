from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import re
import subprocess
import sys
from threading import Lock
from types import SimpleNamespace

import pandas as pd
import pytest

from beyond_click_sim.scorers.recagent import RecAgentYesNoScorer
from beyond_click_sim.scorers.recagent.scorer import RecAgentStageError
from beyond_click_sim.scorers.recagent.state import digest
from test_recagent_memory import FakeEmbeddings


class FakeClient:
    def __init__(self, *, importance="8", fail=None, finish_reason="stop"):
        self.importance, self.fail, self.finish_reason = importance, fail, finish_reason
        self.calls, self.counts, self.lock = [], Counter(), Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        prompt = kwargs["messages"][-1]["content"]
        stage = re.match(r"\[recagent-offline-prompts-v1:(.*?)\]", prompt)[1]
        with self.lock:
            self.calls.append(deepcopy(kwargs))
            self.counts[stage] += 1
            count = self.counts[stage]
        if self.fail is not None:
            override = self.fail(stage, count, prompt)
            if override is not None:
                text = override
            else:
                text = self.response(stage, prompt)
        else:
            text = self.response(stage, prompt)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=text),
                    finish_reason=self.finish_reason,
                )
            ],
            usage=SimpleNamespace(
                model_dump=lambda: {"prompt_tokens": 50, "completion_tokens": 8}
            ),
        )

    def response(self, stage, prompt):
        if stage == "importance":
            return self.importance
        if stage == "sensory":
            return "Movie user has a recorded movie rating: " + digest(prompt)[:8]
        if stage == "promotion":
            return "The movie user may have broad interests supported by these observations."
        if stage == "reflection_question":
            return "Which interests recur?"
        if stage == "reflection_insight":
            identity = re.search(r"^\[(\d+)\]", prompt, re.M)[1]
            return f"Movie interests recur [{identity}]"
        if stage == "action":
            return "\n".join(
                f"{label}: {'yes' if i % 2 == 0 else 'no'}"
                for i, label in enumerate(re.findall(r"^(C\d+):", prompt, re.M))
            )
        return "The user has shown interest in movies, with varied recorded ratings."


def training(n=8):
    X = pd.DataFrame(
        {
            "user_id": ["u"] * n,
            "item_id": [f"i{i}" for i in range(n)],
            "item_title": [f"movie{i}" for i in range(n)],
            "item_genres": ['["Comedy"]'] * n,
            "rating": [float(i % 5 + 1) for i in range(n)],
            "user_gender": ["F"] * n,
            "user_age": [18] * n,
            "user_occupation": [12] * n,
            "user_zip_code": ["DO_NOT_USE_ZIP"] * n,
        },
        index=range(100, 100 + n),
    )
    events = X[["user_id", "item_id"]].copy()
    events["event_id"] = [f"event{i}" for i in range(n)]
    events["timestamp"] = [i // 2 * 86400 for i in range(n)]
    return X, pd.Series(1, index=X.index), events


def candidates(group="g", n=3):
    return pd.DataFrame(
        {
            "user_id": ["u"] * n,
            "item_id": [f"c{i}" for i in range(n)],
            "item_title": [f"candidate{i}" for i in range(n)],
            "item_genres": ["Comedy"] * n,
            "candidate_group": [group] * n,
            "rating": [98765] * n,
            "timestamp": ["SECRET_FUTURE_TIME"] * n,
            "is_sampled": [True] * n,
            "target": [999] * n,
        },
        index=range(n),
    )


def scorer(n=8, *, client=None, **options):
    X, y, events = training(n)
    instance = RecAgentYesNoScorer(
        client=client or FakeClient(),
        training_events=events,
        embedding_provider=options.pop("embedding_provider", FakeEmbeddings()),
        model_revision="fixture-revision",
        **options,
    ).fit(X, y)
    return instance


def test_lazy_import_fit_and_preflight_make_no_calls():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from beyond_click_sim.scorers import RecAgentYesNoScorer; import sys; assert 'sentence_transformers' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    provider = FakeEmbeddings()
    provider.metadata = lambda: pytest.fail(
        "Preflight must not resolve or load weights"
    )
    s = scorer(45, embedding_provider=provider)
    report = s.preflight(candidate_groups=3)
    assert (report["training_events"], report["profile_blocks"]) == (45, 3)
    assert report["base_llm_calls_without_promotions_reflections_retries"] == 99
    assert s.client.calls == provider.calls == []


@pytest.mark.parametrize("limit,events,blocks", [(None, 45, 3), (20, 20, 1)])
def test_full_initialization_or_explicit_latest20(limit, events, blocks):
    s = scorer(45, max_history_items=limit)
    s.build_agents()
    assert s.initialization_errors == {}
    state = s.agent_snapshots["u"]
    assert state.replay_events == events and state.profile_blocks == blocks
    assert s.client.counts["sensory"] == events
    assert state.recent == [f"movie{i}" for i in range(35, 45)]
    assert state.demographics == {
        "name": "Movie user",
        "gender": "female",
        "age_category": "18–24",
        "occupation": "programmer",
    }
    assert state.clock == 22 * 86400 + 1
    assert all(r.created_at <= 22 * 86400 for r in state.memory.long_term)
    assert (
        any(r.status in {"merged", "forgotten"} for r in state.memory.long_term)
        if limit is None
        else True
    )
    all_prompts = "\n".join(c["messages"][-1]["content"] for c in s.client.calls)
    assert "DO_NOT_USE_ZIP" not in all_prompts


def test_stable_ties_and_no_demographics():
    X, y, events = training(4)
    order = [1, 0, 3, 2]
    s = RecAgentYesNoScorer(
        client=FakeClient(importance="6"),
        embedding_provider=FakeEmbeddings(),
        training_events=events.iloc[order],
        use_demographics=False,
    ).fit(X.iloc[order], y.iloc[order])
    s.build_agents()
    state = s.agent_snapshots["u"]
    assert state.recent == ["movie1", "movie0", "movie3", "movie2"]
    assert state.demographics == {"name": "Movie user"}


def test_two_memory_loads_yes_no_and_no_candidate_label_leakage():
    s = scorer(10)
    s.build_agents()
    before = digest(s.agent_snapshots["u"].to_dict())
    s.drain_diagnostics()
    X = candidates()
    assert s.score(X).tolist() == [1, 0, 1]
    trace = s.drain_diagnostics()
    assert sum(r.get("event") == "retrieve" for r in trace) >= 2
    assert [
        r["stage"] for r in trace if r.get("stage") in {"profile_summary", "action"}
    ] == ["profile_summary", "action"]
    messages = [r["messages"] for r in trace if "messages" in r]
    for forbidden in ("SECRET_FUTURE_TIME", "98765", "is_sampled", "target", "999"):
        assert forbidden not in json.dumps(messages)
    X["target"], X["rating"], X["timestamp"], X["is_sampled"] = (
        0,
        -123456,
        "OTHER_FUTURE",
        False,
    )
    assert s.score(X).tolist() == [1, 0, 1]
    again = s.drain_diagnostics()
    assert [r["messages"] for r in again if "messages" in r] == messages
    assert digest(s.agent_snapshots["u"].to_dict()) == before


def test_group_order_and_concurrency_do_not_mutate_state():
    s = scorer(20)
    s.build_agents()
    before = digest(s.agent_snapshots["u"].to_dict())
    groups = [candidates(f"g{i}") for i in range(5)]
    expected = [s.score(g).tolist() for g in groups]
    with ThreadPoolExecutor(max_workers=3) as pool:
        actual = list(pool.map(lambda g: s.score(g).tolist(), reversed(groups)))
    assert actual == list(reversed(expected))
    assert digest(s.agent_snapshots["u"].to_dict()) == before


def test_initialization_retry_rolls_back_memory_clock_and_rng():
    client = FakeClient(
        fail=lambda stage, count, _: "bad importance"
        if stage == "importance" and count == 5
        else None
    )
    s = scorer(16, client=client, max_initialization_attempts=2)
    s.build_agents()
    baseline = scorer(16)
    baseline.build_agents()
    assert s.agent_snapshots == baseline.agent_snapshots
    assert any(r.get("event") == "initialization_retry" for r in s.drain_diagnostics())


def test_score_retry_discards_all_private_changes():
    client = FakeClient(
        fail=lambda stage, count, _: "C1: perhaps"
        if stage == "action" and count == 1
        else None
    )
    s = scorer(20, client=client)
    s.build_agents()
    before = s.agent_snapshots
    with pytest.raises(RecAgentStageError, match="action"):
        s.score(candidates())
    assert s.agent_snapshots == before
    assert s.score(candidates()).tolist() == [1, 0, 1]
    actions = [c for c in client.calls if ":action]" in c["messages"][-1]["content"]]
    assert actions[0] == actions[1]
    assert s.agent_snapshots == before


def test_unavailable_user_has_no_fallback_agent():
    s = scorer(client=FakeClient(importance="invalid"), max_initialization_attempts=2)
    s.build_agents()
    assert s.agent_snapshots == {}
    assert "u" in s.initialization_errors
    count = len(s.client.calls)
    with pytest.raises(RecAgentStageError, match="unavailable"):
        s.score(candidates())
    assert len(s.client.calls) == count


@pytest.mark.parametrize(
    "stage,number", [("profile_update", 2), ("sensory", 7), ("importance", 9)]
)
def test_checkpoint_resume_matches_uninterrupted(tmp_path, stage, number):
    def interrupt(s, n, _):
        if s == stage and n == number:
            raise KeyboardInterrupt("simulated interruption")

    first = scorer(25, client=FakeClient(fail=interrupt), cache_dir=tmp_path)
    with pytest.raises(KeyboardInterrupt):
        first.build_agents()
    resumed = scorer(25, cache_dir=tmp_path)
    resumed.build_agents()
    baseline = scorer(25)
    baseline.build_agents()
    assert resumed.cache_resumes == 1
    assert resumed.agent_snapshots == baseline.agent_snapshots
    assert resumed.score(candidates()).equals(baseline.score(candidates()))
    reloaded = scorer(25, cache_dir=tmp_path)
    reloaded.build_agents()
    assert reloaded.cache_hits == 1 and reloaded.client.calls == []


def test_cache_invalidates_changed_training_config_and_embedding_revision(tmp_path):
    base = scorer(cache_dir=tmp_path)
    base.build_agents()
    for options in (
        {"seed": 10},
        {"use_demographics": False},
        {"embedding_provider": FakeEmbeddings(revision="v2")},
        {"max_history_items": 3},
    ):
        changed = scorer(cache_dir=tmp_path, **options)
        changed.build_agents()
        assert changed.cache_hits == 0 and changed.client.calls
    changed = scorer(cache_dir=tmp_path)
    X, y, events = training()
    X.loc[100, "rating"] = 2.5
    changed.fit(X, y)
    changed.build_agents()
    assert changed.cache_hits == 0


def test_corrupted_checkpoint_fails_explicitly(tmp_path):
    s = scorer(cache_dir=tmp_path)
    s.build_agents()
    path = next((tmp_path / "agents").glob("*.json"))
    data = json.loads(path.read_text())
    data["payload"]["interests"] = "tampered"
    path.write_text(json.dumps(data))
    reloaded = scorer(cache_dir=tmp_path)
    reloaded.build_agents()
    assert "checksum" in reloaded.initialization_errors["u"]
    assert reloaded.agent_snapshots == {}


def test_unreported_llm_revision_cannot_reuse_another_sessions_agent(tmp_path):
    X, y, events = training(3)
    instances = [
        RecAgentYesNoScorer(
            client=FakeClient(importance="6"),
            training_events=events,
            embedding_provider=FakeEmbeddings(),
            cache_dir=tmp_path,
        ).fit(X, y)
        for _ in range(2)
    ]
    for s in instances:
        s.build_agents()
        assert s.cache_hits == 0 and s.client.calls
        assert s.configuration["durable_cache_reuse"] is False
    assert len(list((tmp_path / "agents").glob("*.json"))) == 2
    assert instances[0].agent_snapshots == instances[1].agent_snapshots


@pytest.mark.parametrize(
    "column", ["rating", "target", "timestamp", "is_sampled", "user_gender"]
)
def test_forbidden_candidate_columns_rejected(column):
    with pytest.raises(ValueError, match="whitelisted"):
        scorer(candidate_description_columns=("item_title", column))


def test_truncated_or_overlong_output_fails_without_silent_truncation():
    for client in (
        FakeClient(finish_reason="length"),
        FakeClient(fail=lambda *_: "word " * 121),
    ):
        s = scorer(client=client, max_initialization_attempts=1)
        s.build_agents()
        assert "u" in s.initialization_errors
        assert s.agent_snapshots == {}


@pytest.mark.parametrize(
    "mutation",
    ["timestamps", "event_ids", "identity", "negative_train", "demographics"],
)
def test_bad_training_inputs_fail_before_inference(mutation):
    X, y, events = training()
    if mutation == "timestamps":
        events["timestamp"] = float("nan")
    elif mutation == "event_ids":
        events["event_id"] = "same"
    elif mutation == "identity":
        events.loc[100, "item_id"] = "other"
    elif mutation == "negative_train":
        y.iloc[0] = 0
    else:
        X.loc[100, "user_age"] = 25
    client = FakeClient()
    with pytest.raises(ValueError):
        RecAgentYesNoScorer(client=client, training_events=events).fit(X, y)
    assert client.calls == []
