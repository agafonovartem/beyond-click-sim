from collections import Counter
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from beyond_click_sim.scorers.simuser import SimUSERConfig, SimUSERYesNoScorer
from beyond_click_sim.scorers.simuser import episodic, profiles, prompts
from beyond_click_sim.scorers.simuser.config import AGES, OCCUPATIONS, TRAITS
from beyond_click_sim.scorers.simuser.state import identity


class FakeEmbeddings:
    def __init__(self):
        self.embeddings = self
        self.calls = []

    def create(self, *, model, input):
        self.calls.append({"model": model, "input": input})
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=i,
                    embedding=[b / 128 - 1 for b in sha256(text.encode()).digest()[:8]],
                )
                for i, text in enumerate(input)
            ],
            usage={"total_tokens": len(input)},
        )


class FakeClient:
    def __init__(self, *, refine=0, fail=None, all_no=False):
        self.chat = SimpleNamespace(completions=self)
        self.calls, self.counts = [], Counter()
        self.refine, self.fail, self.all_no = refine, fail, all_no

    def create(self, **options):
        self.calls.append(deepcopy(options))
        content = options["messages"][-1]["content"]
        stage = content.split("]", 1)[0].split(":")[-1]
        payload = json.loads(content.split("INPUT_JSON\n", 1)[1])
        self.counts[stage] += 1
        failed = self.fail(stage, self.counts[stage], payload) if self.fail else None
        if failed is not None:
            return self.response(failed)
        if stage == "taste":
            data = {
                "taste": "Enjoys movie stories across familiar and unfamiliar genres."
            }
        elif stage == "personas":
            data = {
                "personas": [
                    {
                        "age": AGES[i],
                        "occupation": OCCUPATIONS[i],
                        "big_five": dict.fromkeys(TRAITS, 2),
                    }
                    for i in range(5)
                ]
            }
        elif stage == "ratings":
            data = {
                "ratings": [
                    {"label": r["label"], "rating": 3} for r in payload["items"]
                ]
            }
        elif stage == "self_ask":
            data = {
                "questions": [
                    "What past movie experiences relate to these genres?",
                    "Which stories prompted an interaction despite dislike?",
                ]
            }
        else:
            evidence = [r["id"] for rows in payload["evidence"].values() for r in rows]
            if stage == "check":
                # Each group starts at watch; revision count is inferred from call payload.
                rounds = sum(
                    1
                    for call in self.calls
                    if ":revise]" in call["messages"][-1]["content"]
                )
                data = {
                    "checks": [
                        {
                            "label": r["label"],
                            "contradiction": False,
                            "insufficient_evidence": rounds < self.refine,
                            "reason": "More history may inform this interaction.",
                            "evidence_ids": evidence[:1],
                        }
                        for r in payload["candidates"]
                    ]
                }
            else:
                data = {
                    "decisions": [
                        {
                            "label": r["label"],
                            "watch": False
                            if self.all_no
                            else (i % 2 == (1 if stage == "revise" else 0)),
                            "reason": "May interact and subsequently dislike the movie.",
                            "evidence_ids": evidence[:1],
                        }
                        for i, r in enumerate(payload["candidates"])
                    ]
                }
        return self.response(json.dumps(data))

    @staticmethod
    def response(text):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=text), finish_reason="stop"
                )
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )


def fixture_data(n=12):
    catalog = pd.DataFrame(
        {
            "item_id": [f"i{i}" for i in range(2 * n + 3)],
            "item_title": [f"Movie {i}" for i in range(2 * n + 3)],
            "item_genres": [
                "Comedy|Drama" if i % 2 else "Action" for i in range(2 * n + 3)
            ],
            "item_summary": [
                f"A short frozen plot about character {i}." for i in range(2 * n + 3)
            ],
        }
    )
    X = pd.DataFrame(
        [
            {"user_id": user, "item_id": f"i{i + offset}", "rating": i % 5 + 1}
            for user, offset in (("u", 0), ("v", n))
            for i in range(n)
        ]
    )
    events = X[["user_id", "item_id"]].copy()
    events["event_id"] = [f"e{i}" for i in range(len(X))]
    events["timestamp"] = list(range(n)) * 2
    group = pd.DataFrame(
        {
            "user_id": ["u"] * 4,
            "item_id": [f"i{i}" for i in range(n, n + 4)],
            "candidate_group": ["g"] * 4,
        },
        index=[100, 101, 102, 103],
    )
    return X, pd.Series(1, index=X.index), events, catalog, group


def fitted(tmp_path=None, *, n=12, client=None, config=None, users=("u",), **kwargs):
    X, y, events, catalog, group = fixture_data(n)
    client = client or FakeClient()
    embeddings = FakeEmbeddings()
    scorer = SimUSERYesNoScorer(
        client=client,
        embedding_client=embeddings,
        model="fake",
        model_revision="frozen",
        training_events=events,
        catalog=catalog,
        config=config,
        cache_dir=tmp_path,
        **kwargs,
    ).fit(X, y, profile_user_ids=users)
    return scorer, group, client, embeddings


def payloads(client, stage):
    return [
        json.loads(r["messages"][-1]["content"].split("INPUT_JSON\n", 1)[1])
        for r in client.calls
        if f":{stage}]" in r["messages"][-1]["content"]
    ]


def test_full_initialization_combined_binary_and_fixed_state(tmp_path):
    scorer, group, client, embeddings = fitted(tmp_path)
    assert not client.calls and not embeddings.calls
    preflight = scorer.preflight(candidate_groups=1)
    assert preflight["uncached_successful_calls"]["initialization"] == 17
    assert preflight["population_users"] == 2 and preflight["users"] == 1
    scorer.build_agents()
    before = scorer.agent_snapshots
    assert sum(client.counts.values()) == 17
    scores = scorer.score(group)
    assert scores.tolist() == [1, 0, 1, 0]
    assert scores.index.equals(group.index)
    assert len(client.calls) == 20
    assert before == scorer.agent_snapshots
    assert client.counts == {
        "taste": 1,
        "personas": 1,
        "ratings": 15,
        "self_ask": 1,
        "watch": 1,
        "check": 1,
    }
    assert (
        scorer.agent_snapshots[identity("u")].to_dict()["selected_persona_index"] == 0
    )
    again = scorer.score(
        group.assign(
            target=[1, 0, 0, 1], rating=1, timestamp=999, user_age=99, is_sampled=True
        )
    )
    pd.testing.assert_series_equal(scores, again)
    assert len(client.calls) == 20
    scorer.close()


def test_every_response_stage_uses_interaction_target_and_allows_dislike():
    scorer, group, client, _ = fitted(client=FakeClient(refine=2))
    scorer.build_agents()
    scores = scorer.score(group)
    assert len(client.calls) == 24  # 17 + 7
    assert scores.tolist() == [0, 1, 0, 1]  # Supported revision changes decisions.
    for call in client.calls:
        text = call["messages"][-1]["content"]
        if any(f":{s}]" in text for s in ("watch", "check", "revise")):
            assert prompts.TARGET in text
            assert "at least 4" in text
    records = scorer.drain_diagnostics()
    retrievals = [r for r in records if r["event"] == "retrieval"]
    assert [(r["k1"], r["k2"]) for r in retrievals] == [(5, 3), (7, 5), (9, 7)]
    sets = [
        {e["id"] for rows in r["evidence"].values() for e in rows} for r in retrievals
    ]
    assert sets[0] <= sets[1] <= sets[2]


def test_exhaustion_keeps_last_valid_watch_and_unresolved_flag():
    scorer, group, client, _ = fitted(
        n=2,
        client=FakeClient(refine=99),
        config=SimUSERConfig(episodic_k=50, graph_k=50),
    )
    scorer.build_agents()
    result = scorer.score(group)
    assert result.tolist() == [1, 0, 1, 0]
    assert len(client.calls) == 20
    completed = [
        r for r in scorer.drain_diagnostics() if r["event"] == "group_complete"
    ][-1]
    assert completed["stop_reason"] == "memory_exhausted" and completed["unresolved"]


def test_refinement_cap_keeps_last_valid_vector():
    scorer, group, client, _ = fitted(n=30, client=FakeClient(refine=99))
    scorer.build_agents()
    scorer.score(group)
    completed = [
        r for r in scorer.drain_diagnostics() if r["event"] == "group_complete"
    ][-1]
    assert completed["stop_reason"] == "refinement_cap" and completed["unresolved"]
    assert len(client.calls) == 24


def test_random_fifty_full_history_habits_and_shared_contrast_samples():
    scorer, _, client, _ = fitted(n=80)
    scorer.build_agents()
    snapshot = scorer.agent_snapshots[identity("u")].to_dict()
    assert len(snapshot["preparation"]["sampled_event_ids"]) == 50
    assert (
        len(snapshot["memory_ids"])
        == snapshot["habits"]["engagement_interaction_count"]
        == 80
    )
    assert any(
        int(i.split('"e')[1].rstrip('"')) < 30
        for i in snapshot["preparation"]["sampled_event_ids"]
    )
    assert len(payloads(client, "taste")[0]["history"]) == 50
    calls = payloads(client, "ratings")
    assert len(calls) == 15
    for b in range(3):
        assert all(calls[b]["items"] == calls[b + 3 * p]["items"] for p in range(5))
    for call in calls:
        assert len(call["items"]) == 20
        assert all(
            set(r) == {"label", "title", "genres", "summary"} for r in call["items"]
        )
    assert snapshot["preparation"]["rho"] == 10


@pytest.mark.parametrize(
    "mean,pickiness",
    [(4.5, "not picky"), (3.5, "moderately picky"), (3, "extremely picky")],
)
def test_habit_boundaries_and_conformity(mean, pickiness):
    history = [{"item": "a", "rating": mean - 0.5}, {"item": "b", "rating": mean + 0.5}]
    catalog = {
        "a": {"genres": ["Drama", "Comedy"], "training_mean_rating": mean},
        "b": {"genres": ["Drama"], "training_mean_rating": mean},
    }
    habits = profiles.habits(history, catalog)
    assert habits["pickiness"] == pickiness
    assert habits["conformity_mean_squared_deviation"] == 0.25
    assert habits["variety_distinct_genres"] == 2


def test_literal_rating_four_asymmetry_and_positive_interactions():
    scorer, _, _, _ = fitted()
    memories = scorer.memories[identity("u")]
    four = next(r for r in memories if r["rating"] == 4)
    assert "neutral" in four["text"]
    assert profiles.preference(4) == "liked"
    assert (
        scorer.graph.edges[(identity("u"), four["item_id"])][0]["relation"] == "liked"
    )
    assert len(memories) == 12
    assert any(r["rating"] == 1 for r in memories)


def test_each_query_prefix_merged_by_max_similarity_and_stable_id():
    records = [{"id": "e0"}, {"id": "e1"}, {"id": "e2"}]
    ranks = episodic.ranked_queries(
        records, np.array([[1, 0], [0, 1], [1, 0]]), np.array([[1, 0], [0, 1], [1, 0]])
    )
    result = episodic.retrieve(ranks, 1)
    assert [r["id"] for r in result] == ["e0", "e1"]
    assert result[0]["query_indices"] == [0, 2]
    assert result[0]["cosine"] == 1
    assert {r["id"] for r in result} <= {r["id"] for r in episodic.retrieve(ranks, 2)}


def test_stage_resume_group_cache_and_model_invalidation(tmp_path):
    client = FakeClient(
        fail=lambda s, n, p: "invalid" if s == "ratings" and n >= 4 else None
    )
    config = SimUSERConfig(initialization_attempts=1)
    scorer, _, _, _ = fitted(tmp_path, client=client, config=config)
    scorer.build_agents()
    assert scorer.initialization_errors and not scorer.agent_snapshots
    assert len(client.calls) == 6  # taste, personas, 3 successful + failed ratings
    scorer.close()
    resumed, group, client2, _ = fitted(tmp_path, config=config)
    resumed.build_agents()
    assert len(client2.calls) == 12
    result = resumed.score(group)
    resumed.close()
    cached, _, client3, embeds3 = fitted(tmp_path, config=config)
    cached.build_agents()
    pd.testing.assert_series_equal(cached.score(group), result)
    assert not client3.calls and not embeds3.calls
    cached.close()
    changed, _, client4, _ = fitted(tmp_path, config=replace(config, seed=3))
    changed.build_agents()
    assert len(client4.calls) == 17
    changed.close()


def test_labels_demographics_history_order_and_user_subset_do_not_change_evidence(
    tmp_path,
):
    X, y, events, catalog, group = fixture_data()
    outputs = []
    for subset, extras in [(True, False), (False, True)]:
        client = FakeClient()
        x = (
            X.assign(user_age=99, user_occupation="hidden", target_like_ge4=0)
            if extras
            else X
        )
        scorer = SimUSERYesNoScorer(
            client=client,
            embedding_client=FakeEmbeddings(),
            model_revision="same",
            training_events=events,
            catalog=catalog,
        ).fit(x, y, profile_user_ids=["u"] if subset else ["u", "v"])
        scorer.build_agents()
        outputs.append(
            (
                scorer.evidence_fingerprint,
                scorer.agent_snapshots[identity("u")],
                scorer.score(group),
            )
        )
    assert outputs[0][:2] == outputs[1][:2]
    pd.testing.assert_series_equal(outputs[0][2], outputs[1][2])


def test_group_order_and_worker_count_preserve_inputs_outputs():
    from runners.in_distribution.interaction_prediction.methods.llm_yes_no import (
        _score_groups,
    )

    results = []
    for workers, reverse in [(1, False), (3, True)]:
        scorer, group, client, _ = fitted()
        scorer.build_agents()
        groups = pd.concat(
            [group, group.set_axis(range(200, 204)).assign(candidate_group="g2")]
        )
        if reverse:
            groups = pd.concat([groups.iloc[4:], groups.iloc[:4]])
        scores, errors = _score_groups(
            scorer,
            groups,
            candidate_group_column="candidate_group",
            max_attempts=1,
            max_workers=workers,
        )
        assert not errors
        results.append(
            (
                scores.sort_index(),
                sorted(json.dumps(c, sort_keys=True) for c in client.calls),
            )
        )
    pd.testing.assert_series_equal(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


@pytest.mark.parametrize(
    "kind", ["negative", "rating", "missing_summary", "event", "no_contrast"]
)
def test_invalid_training_fails_before_calls(kind):
    X, y, events, catalog, _ = fixture_data()
    if kind == "negative":
        y.iloc[0] = 0
    elif kind == "rating":
        X.loc[0, "rating"] = 9
    elif kind == "missing_summary":
        catalog.loc[0, "item_summary"] = None
    elif kind == "event":
        events.loc[0, "item_id"] = "wrong"
    else:
        X, y, events = X.iloc[:12], y.iloc[:12], events.iloc[:12]
    client = FakeClient()
    with pytest.raises(ValueError):
        SimUSERYesNoScorer(client=client, catalog=catalog, training_events=events).fit(
            X, y
        )
    assert not client.calls


def test_context_preflight_fails_without_clients():
    scorer, _, client, embeddings = fitted(config=SimUSERConfig(context_tokens=100))
    with pytest.raises(ValueError, match="context bound"):
        scorer.preflight(candidate_groups=1)
    assert not client.calls and not embeddings.calls


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_candidate",
        "duplicate_candidate",
        "missing_candidate",
        "unknown_evidence",
        "non_bool",
        "duplicate_key",
    ],
)
def test_strict_response_validation(mutation):
    row = {
        "label": "C1",
        "watch": True,
        "reason": "Anticipated dislike is compatible with interaction.",
        "evidence_ids": [],
    }
    data = {"decisions": [row]}
    if mutation == "unknown_candidate":
        row["label"] = "C2"
    elif mutation == "duplicate_candidate":
        data["decisions"].append(row.copy())
    elif mutation == "missing_candidate":
        data["decisions"] = []
    elif mutation == "unknown_evidence":
        row["evidence_ids"] = ["invented"]
    elif mutation == "non_bool":
        row["watch"] = 1
    text = json.dumps(data)
    if mutation == "duplicate_key":
        text = text.replace('"watch": true', '"watch": false, "watch": true')
    with pytest.raises(ValueError):
        prompts.response(text, ["C1"], set())


def test_personas_must_be_distinct_and_in_menu():
    candidate = {
        "age": 25,
        "occupation": "writer",
        "big_five": dict.fromkeys(TRAITS, 2),
    }
    with pytest.raises(ValueError, match="Duplicate personas"):
        prompts.personas(json.dumps({"personas": [candidate] * 5}))


def test_persona_ratings_mapped_to_mixed_labels_and_correct_contrast_sign():
    def ratings(stage, count, payload):
        if stage != "ratings":
            return None
        chosen = payload["persona"]["age"] == AGES[2]
        rows = [
            {
                "label": r["label"],
                "rating": (5 if int(r["title"].split()[-1]) < 12 else 1)
                if chosen
                else 3,
            }
            for r in payload["items"]
        ]
        return json.dumps({"ratings": rows[::-1]})  # Response order is irrelevant.

    scorer, _, _, _ = fitted(client=FakeClient(fail=ratings))
    scorer.build_agents()
    snapshot = scorer.agent_snapshots[identity("u")].to_dict()
    assert snapshot["persona_scores"] == [0, 0, 120, 0, 0]
    assert snapshot["selected_persona_index"] == 2
    subsets = snapshot["preparation"]["subsets"]
    assert all(len(set(s["items"])) == 20 for s in subsets)
    assert all(set(s["target"]) < set(s["items"]) for s in subsets)


@pytest.mark.parametrize(
    "change", ["rating", "timestamp", "summary", "model", "embedding"]
)
def test_effective_evidence_and_model_changes_invalidate_checkpoints(tmp_path, change):
    first, _, _, _ = fitted(tmp_path)
    first.build_agents()
    old = first.agent_snapshots[identity("u")].fingerprint
    first.close()
    X, y, events, catalog, _ = fixture_data()
    kwargs = {}
    if change == "rating":
        X.loc[0, "rating"] = 5
    elif change == "timestamp":
        events.loc[0, "timestamp"] = 100
    elif change == "summary":
        catalog.loc[0, "item_summary"] = "A revised static summary."
    elif change == "embedding":
        kwargs["embedding_identity"] = "new-embedding-revision"
    client = FakeClient()
    scorer = SimUSERYesNoScorer(
        client=client,
        embedding_client=FakeEmbeddings(),
        model="changed" if change == "model" else "fake",
        model_revision="frozen",
        training_events=events,
        catalog=catalog,
        cache_dir=tmp_path,
        **kwargs,
    ).fit(X, y, profile_user_ids=["u"])
    scorer.build_agents()
    assert len(client.calls) == 17
    assert old != scorer.agent_snapshots[identity("u")].fingerprint
    scorer.close()


def test_reordered_training_and_unsampled_older_evidence_are_deterministic():
    X, y, events, catalog, _ = fixture_data(n=70)
    states = []
    for reverse in (False, True):
        index = X.index[::-1] if reverse else X.index
        scorer = SimUSERYesNoScorer(
            client=FakeClient(),
            embedding_client=FakeEmbeddings(),
            model_revision="same",
            training_events=events.loc[index],
            catalog=catalog,
        ).fit(X.loc[index], y.loc[index], profile_user_ids=["u"])
        scorer.build_agents()
        states.append(scorer.agent_snapshots)
    assert states[0] == states[1]


def test_corrupted_checkpoint_is_rejected_without_executing_or_loading_it(tmp_path):
    scorer, _, _, _ = fitted(tmp_path)
    scorer.build_agents()
    scorer.close()
    path = next((tmp_path / "agents").glob("*.json"))
    data = json.loads(path.read_text())
    data["payload"]["taste"] = "tampered"
    path.write_text(json.dumps(data))
    resumed, _, client, _ = fitted(tmp_path)
    resumed.build_agents()
    assert not resumed.agent_snapshots and resumed.initialization_errors
    assert not client.calls
    resumed.close()


def test_transient_embedding_initialization_failure_retries_before_generation():
    scorer, group, client, embeddings = fitted()
    original = embeddings.create
    attempts = []

    def flaky(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise RuntimeError("Temporary embedding error")
        return original(**kwargs)

    embeddings.create = flaky
    scorer.build_agents()
    assert len(scorer.agent_snapshots) == 1
    assert len(client.calls) == 17
    assert scorer.score(group).notna().all()
    assert (
        scorer.initialization_metadata()["diagnostic_counts"][
            "embedding_initialization_retry"
        ]
        == 1
    )


def test_cached_group_keeps_queries_paths_and_exact_requests_auditable(tmp_path):
    scorer, group, client, _ = fitted(tmp_path)
    scorer.build_agents()
    scorer.score(group)
    scorer.close()
    cached, _, new_client, _ = fitted(tmp_path)
    cached.build_agents()
    cached.score(group)
    hit = next(r for r in cached.drain_diagnostics() if r["event"] == "group_cache_hit")
    inference = hit["cached_inference"]
    assert len(inference["queries"]) == 3
    assert len(inference["requests"]) == 3
    assert inference["retrieval_rounds"][0]["evidence"]["graph"]
    assert [r["messages"] for r in inference["requests"]] == [
        r["messages"] for r in client.calls[-3:]
    ]
    assert not new_client.calls
    cached.close()
