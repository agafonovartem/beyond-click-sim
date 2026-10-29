from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import random
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from beyond_click_sim.scorers import LLMSRecSynScorer
from beyond_click_sim.scorers.history import LLMSRecSynScorer as ExportedScorer
from beyond_click_sim.scorers.history.llmsrec_syn_retrieval import CachedEmbeddings
from runners.in_distribution.interaction_prediction.methods.llm_yes_no import (
    _score_groups,
)


class EmbeddingClient:
    def __init__(self):
        self.embeddings = self
        self.calls = []

    def create(self, *, model, input):
        self.calls.append({"model": model, "input": input.copy()})
        vectors = []
        for text in input:
            titles = json.loads(text)
            first = titles[0]
            vector = [0.8, 0.6, 0] if first.startswith("B") else [1, 0, 0]
            if first.startswith("Future"):
                vector = [-1, 0, 0]
            vectors.append(vector)
        # Return out of order to exercise API-index restoration.
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=v)
                for i, v in reversed(list(enumerate(vectors)))
            ],
            usage=SimpleNamespace(prompt_tokens=7, total_tokens=7),
        )


class RankingClient:
    def __init__(self, replies=None):
        self.chat = SimpleNamespace(completions=self)
        self.calls = []
        self.replies = list(replies or [])

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        text = kwargs["messages"][0]["content"].split("Test User:\n", 1)[1]
        labels = re.findall(r"^(C\d+)\. ", text, re.M)
        reply = self.replies.pop(0) if self.replies else "\n".join(reversed(labels))
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
            usage=SimpleNamespace(
                prompt_tokens=30, completion_tokens=10, total_tokens=40
            ),
        )


def fixture_data():
    rows = []
    for user, events in {
        "target": [("t0", "Target 0", 1), ("t1", "Target 1", 10)],
        "a": [("a1", "A1", 2), ("a2", "A2", 7), ("a3", "A3", 15), ("a4", "A4", 18)],
        "b": [("b1", "B1", 1), ("b2", "B2", 4), ("b3", "B3", 8), ("b4", "B4", 17)],
        "later": [("l1", "Target later 1", 12), ("l2", "Target later 2", 20)],
        "solo": [("s1", "Solo", 9)],
        "future": [("f1", "Future 1", 30), ("f2", "Future 2", 40)],
    }.items():
        for item, title, timestamp in events:
            rows.append(
                {
                    "user_id": user,
                    "item_id": item,
                    "item_title": title,
                    "timestamp": timestamp,
                    "event_id": f"{user}:{item}",
                }
            )
    full = pd.DataFrame(rows, index=[i * 3 + 5 for i in range(len(rows))])
    train = full[["user_id", "item_id", "item_title"]].assign(
        rating=5, item_rating_mean=4.5
    )
    events = full[["user_id", "item_id", "event_id", "timestamp"]].copy()
    catalog = pd.concat(
        [
            full[["item_id", "item_title"]],
            pd.DataFrame(
                {
                    "item_id": [f"c{i}" for i in range(30)],
                    "item_title": [f"Candidate {i}" for i in range(30)],
                }
            ),
        ],
        ignore_index=True,
    )
    return train, pd.Series(1, index=train.index), events, catalog


def make_scorer(*, client=None, embed_client=None, data=None, **kwargs):
    train, y, events, catalog = data or fixture_data()
    embeddings = CachedEmbeddings(
        embed_client or EmbeddingClient(),
        "embedding",
        model_identity="fake-v1",
        training_fingerprint="train-v1",
    )
    return LLMSRecSynScorer(
        client or RankingClient(), "llm", embeddings=embeddings, **kwargs
    ).fit(
        train,
        y,
        train_event_metadata=events,
        catalog=catalog,
    )


def candidates(n=4, user="target", group="g"):
    return pd.DataFrame(
        {
            "user_id": [user] * n,
            "item_id": [f"c{i}" for i in range(n)],
            "item_title": [f"Candidate {i}" for i in range(n)],
            "candidate_group": [group] * n,
            "rating": [999] * n,
        },
        index=[i * 7 + 13 for i in range(n)],
    )


def test_temporal_retrieval_label_reservation_interleaving_and_titles_only():
    embeddings = EmbeddingClient()
    scorer = make_scorer(embed_client=embeddings)
    demo = scorer.demonstration("target")
    assert ExportedScorer is LLMSRecSynScorer
    assert demo["cutoff"] == 10
    assert [u["user_id"] for u in demo["selected_users"]] == ["a", "b"]
    assert [u["next_item"]["item_id"] for u in demo["selected_users"]] == ["a2", "b3"]
    assert [e["item_id"] for e in demo["aggregate_history"]] == ["a1", "b2", "b1"]
    assert demo["excluded_users"] == {"self": 1, "fewer_than_two_eligible_events": 3}
    assert all(
        e["timestamp"] <= 10
        for u in demo["selected_users"]
        for e in [*u["prefix"], u["next_item"]]
    )
    texts = [json.loads(t) for call in embeddings.calls for t in call["input"]]
    assert texts == [["Target 0", "Target 1"], ["A1"], ["B1", "B2"]]
    by_label = {c["label"]: c["item_id"] for c in demo["candidates"]}
    assert [by_label[label] for label in demo["answer"][:2]] == ["a2", "b3"]
    assert len(by_label) == len(set(by_label.values())) == 20
    assert set(demo["answer"]) == set(by_label)
    scorer.score(candidates())
    prompt = scorer.client.calls[0]["messages"][0]["content"]
    assert "Aggregated Demonstration Example" in prompt
    assert "rating" not in prompt and "999" not in prompt
    assert "Test User:" in prompt and "C20" in prompt
    assert len(scorer.client.calls) == 1


def test_future_suffix_changes_cannot_influence_retrieval_or_demo():
    train, y, events, catalog = fixture_data()
    base = make_scorer(data=(train, y, events, catalog)).demonstration("target")
    modified_train, modified_events = train.copy(), events.copy()
    pos = train.index[train.item_id.eq("a3")][0]
    modified_train.loc[pos, ["item_id", "item_title"]] = ["c25", "Candidate 25"]
    modified_events.loc[pos, ["item_id", "timestamp"]] = ["c25", 19]
    changed = make_scorer(
        data=(modified_train, y, modified_events, catalog)
    ).demonstration("target")
    assert base == changed


def test_same_demo_user_has_distinct_labels_and_embeddings_for_later_cutoff():
    scorer = make_scorer()
    early = scorer.demonstration("target")
    late = scorer.demonstration("later")
    assert early["selected_users"][0]["next_item"]["item_id"] == "a2"
    assert late["selected_users"][0]["next_item"]["item_id"] == "a4"
    assert [e["item_id"] for e in late["selected_users"][0]["prefix"]] == [
        "a1",
        "a2",
        "a3",
    ]
    assert (
        early["selected_users"][0]["prefix_fingerprint"]
        != late["selected_users"][0]["prefix_fingerprint"]
    )
    assert scorer.demonstration("target") == early


def test_training_ties_are_stable_and_inclusive():
    train, y, events, catalog = fixture_data()
    events.loc[train.item_id.eq("a3"), "timestamp"] = 10
    scorer = make_scorer(data=(train, y, events, catalog))
    user = scorer.demonstration("target")["selected_users"][0]
    assert user["next_item"]["item_id"] == "a3"
    events.loc[train.item_id.eq("a2"), "timestamp"] = 10
    scorer = make_scorer(data=(train, y, events, catalog))
    assert (
        scorer.demonstration("target")["selected_users"][0]["next_item"]["item_id"]
        == "a3"
    )


def test_repeated_history_and_shared_positive_ids_keep_unique_candidates():
    train, y, events, catalog = fixture_data()
    pos = train.index[train.item_id.eq("b3")][0]
    train.loc[pos, ["item_id", "item_title"]] = ["a2", "A2"]
    events.loc[pos, "item_id"] = "a2"
    pos = train.index[train.item_id.eq("b2")][0]
    train.loc[pos, ["item_id", "item_title"]] = ["b1", "B1"]
    events.loc[pos, "item_id"] = "b1"
    demo = make_scorer(data=(train, y, events, catalog)).demonstration("target")
    assert [e["item_id"] for e in demo["aggregate_history"]] == ["a1", "b1", "b1"]
    assert len({e["item_id"] for e in demo["candidates"]}) == 20
    by_label = {e["label"]: e["item_id"] for e in demo["candidates"]}
    assert by_label[demo["answer"][0]] == "a2"


def test_duplicate_titles_are_distinct_items_and_rank_scores_preserve_row_index():
    train, y, events, catalog = fixture_data()
    catalog.loc[catalog.item_id.isin(["c0", "c1"]), "item_title"] = "Same title"
    scorer = make_scorer(data=(train, y, events, catalog))
    X = candidates()
    X.loc[X.item_id.isin(["c0", "c1"]), "item_title"] = "Same title"
    before = X.copy(deep=True)
    rng_state = random.getstate()
    result = scorer.score(X)
    assert result.index.equals(X.index) and result.name == "score"
    assert result.tolist() == [0, 1, 2, 3]
    assert random.getstate() == rng_state
    pd.testing.assert_frame_equal(X, before)
    record = scorer.drain_diagnostics()[0]
    assert record["completed"] and record["usage"]["total_tokens"] == 40
    assert scorer.drain_diagnostics() == []


@pytest.mark.parametrize("n", [1, 2, 7, 20])
def test_variable_target_group_sizes(n):
    assert make_scorer().score(candidates(n)).tolist() == list(range(n))


@pytest.mark.parametrize(
    "invalid", ["", "C1", "C1\nC1\nC2\nC3", "C1\nC2\nC3\nC99", RuntimeError("offline")]
)
def test_retries_preserve_demonstration_and_only_return_complete_rankings(invalid):
    scorer = make_scorer(client=RankingClient([invalid]))
    scores, errors = _score_groups(
        scorer, candidates(), candidate_group_column="candidate_group", max_attempts=2
    )
    assert not errors and scores.tolist() == [0, 1, 2, 3]
    records = scorer.drain_diagnostics()
    assert [r["completed"] for r in records] == [False, True]
    assert records[0]["demonstration"] == records[1]["demonstration"]
    assert scorer.client.calls[0] == scorer.client.calls[1]


def test_no_neighbors_reports_failure_without_generation_or_future_fallback():
    scorer = make_scorer(k=3)
    scores, errors = _score_groups(
        scorer, candidates(), candidate_group_column="candidate_group", max_attempts=2
    )
    assert scores.isna().all() and len(errors) == 1
    assert scorer.client.calls == []
    assert all(not r["request_issued"] for r in scorer.drain_diagnostics())


def test_parallel_reproducibility_cache_reuse_and_external_demo_mutation():
    scorer = make_scorer(seed=7)
    before = scorer.demonstration("target")
    copy = scorer.demonstration("target")
    copy["answer"].reverse()
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(
            pool.map(lambda i: scorer.score(candidates(group=f"g{i}")), range(6))
        )
    assert all(r.tolist() == [0, 1, 2, 3] for r in results)
    assert scorer.demonstration("target") == before
    assert len(scorer.embeddings.client.calls) == 1
    assert make_scorer(seed=7).demonstration("target") == before
    assert (
        make_scorer(seed=8).demonstration("target")["candidates"]
        != before["candidates"]
    )


def test_history_budget_keeps_latest_events_and_fills_nondivisible_round():
    scorer = make_scorer(max_history_items=1, aggregate_history_items=1)
    demo = scorer.demonstration("target")
    assert len(demo["aggregate_history"]) == 1
    assert [e["item_id"] for e in demo["selected_users"][1]["prefix"]] == ["b2"]
    scorer.score(candidates())
    assert [e["item_id"] for e in scorer.drain_diagnostics()[0]["target_history"]] == [
        "t1"
    ]
    demo = make_scorer(aggregate_history_items=3).demonstration("target")
    assert len(demo["aggregate_history"]) == 3  # not floor(3/2)*2


def test_steam_and_datetime_control_metadata():
    train, y, events, catalog = fixture_data()
    events["timestamp"] = pd.to_datetime(
        events.timestamp, unit="D", origin="2020-01-01", utc=True
    )
    scorer = make_scorer(data=(train, y, events, catalog), dataset="steam")
    scorer.score(candidates())
    record = scorer.drain_diagnostics()[0]
    assert record["demonstration"]["timestamp_kind"] == "utc_nanoseconds"
    assert "Reviewed Games" in record["messages"][0]["content"]
    assert "purchase" not in record["messages"][0]["content"].lower()


@pytest.mark.parametrize(
    "case",
    [
        "missing_time",
        "nan",
        "infinite",
        "string_time",
        "duplicate_event",
        "bad_alignment",
        "bad_pair",
        "negative",
        "missing_title",
    ],
)
def test_invalid_training_provenance_is_rejected(case):
    train, y, events, catalog = fixture_data()
    if case == "missing_time":
        events = events.drop(columns="timestamp")
    elif case in ("nan", "infinite"):
        events["timestamp"] = events.timestamp.astype(float)
        events.iloc[0, events.columns.get_loc("timestamp")] = (
            np.nan if case == "nan" else np.inf
        )
    elif case == "string_time":
        events["timestamp"] = events.timestamp.astype(str)
    elif case == "duplicate_event":
        events.iloc[1, events.columns.get_loc("event_id")] = events.iloc[0].event_id
    elif case == "bad_alignment":
        events = events.iloc[::-1]
    elif case == "bad_pair":
        events.iloc[0, events.columns.get_loc("item_id")] = "a4"
    elif case == "negative":
        y.iloc[0] = 0
    else:
        train.iloc[0, train.columns.get_loc("item_title")] = ""
    with pytest.raises(ValueError):
        make_scorer(data=(train, y, events, catalog))


@pytest.mark.parametrize(
    "case",
    ["duplicate_index", "duplicate_item", "mixed_user", "null_group", "unknown_user"],
)
def test_invalid_candidate_groups(case):
    X = candidates()
    if case == "duplicate_index":
        X.index = [0] * len(X)
    elif case == "duplicate_item":
        X.iloc[1, X.columns.get_loc("item_id")] = "c0"
    elif case == "mixed_user":
        X.iloc[0, X.columns.get_loc("user_id")] = "a"
    elif case == "null_group":
        X["candidate_group"] = None
    else:
        X["user_id"] = "unknown"
    with pytest.raises(ValueError):
        make_scorer().score(X)


def test_empty_candidates_and_unfitted_scorer():
    scorer = make_scorer()
    assert scorer.score(candidates(0)).empty and not scorer.client.calls
    fresh = LLMSRecSynScorer(RankingClient(), "fake", embeddings=scorer.embeddings)
    with pytest.raises(RuntimeError, match="fitted"):
        fresh.score(candidates())


def test_movie_title_normalization_is_local_and_shared_by_both_models():
    train, y, events, catalog = fixture_data()
    train.loc[train.item_id.eq("a1"), "item_title"] = "Matrix, The (1999)"
    catalog.loc[catalog.item_id.eq("a1"), "item_title"] = "Matrix, The (1999)"
    original = train.copy(deep=True)
    scorer = make_scorer(data=(train, y, events, catalog))
    demo = scorer.demonstration("target")
    assert demo["aggregate_history"][0]["title"] == "The Matrix"
    assert '["The Matrix"]' in scorer.embeddings.client.calls[0]["input"]
    scorer.score(candidates())
    assert "Matrix, The (1999)" not in scorer.client.calls[0]["messages"][0]["content"]
    pd.testing.assert_frame_equal(train, original)


def test_corrupt_embedding_batch_indices_are_rejected():
    client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **_: SimpleNamespace(
                data=[
                    SimpleNamespace(index=0, embedding=[1, 2]),
                    SimpleNamespace(index=0, embedding=[2, 1]),
                ],
                usage=None,
            )
        )
    )
    cache = CachedEmbeddings(
        client, "m", model_identity="fake", training_fingerprint="s"
    )
    with pytest.raises(ValueError, match="indices"):
        cache.encode(['["A"]', '["B"]'])


def test_embedding_cache_normalization_is_identical_on_reload(tmp_path):
    class Client:
        embeddings = None

        def __init__(self):
            self.embeddings = self

        def create(self, **kwargs):
            return SimpleNamespace(
                data=[SimpleNamespace(index=0, embedding=[1.12345, 2.34567, 3.56789])],
                usage=None,
            )

    options = dict(
        model_identity="fake",
        training_fingerprint="s",
        cache_path=tmp_path / "cache.sqlite3",
    )
    cache = CachedEmbeddings(Client(), "m", **options)
    first = cache.encode(['["A"]'])
    cache.close()
    cached = CachedEmbeddings(None, "m", **options)
    np.testing.assert_array_equal(first, cached.encode(['["A"]']))
    cached.close()


def test_integer_event_ids_are_not_coerced_by_float_timestamps():
    train, y, events, catalog = fixture_data()
    user_ids = {user: i for i, user in enumerate(train.user_id.unique())}
    item_ids = {item: i for i, item in enumerate(catalog.item_id)}
    train["user_id"] = train.user_id.map(user_ids)
    train["item_id"] = train.item_id.map(item_ids)
    events["user_id"] = events.user_id.map(user_ids)
    events["item_id"] = events.item_id.map(item_ids)
    events["event_id"] = range(len(events))
    events["timestamp"] = events.timestamp.astype(float)
    catalog["item_id"] = catalog.item_id.map(item_ids)
    scorer = make_scorer(data=(train, y, events, catalog))
    demo = scorer.demonstration(user_ids["target"])
    assert demo["cutoff"] == 10.0
    assert all(isinstance(e["event_id"], int) for e in demo["aggregate_history"])


def test_embedding_cache_is_persistent_and_separated_by_input_model_and_split(tmp_path):
    path = tmp_path / "cache.sqlite3"
    texts = ['["A1"]', '["B1"]', '["A1"]']
    first = EmbeddingClient()
    a = CachedEmbeddings(
        first,
        "m",
        model_identity="backend:m-v1",
        training_fingerprint="train",
        cache_path=path,
        batch_size=2,
    )
    expected = a.encode(texts)
    assert len(first.calls) == 1 and len(first.calls[0]["input"]) == 2
    a.close()
    second = EmbeddingClient()
    b = CachedEmbeddings(
        second,
        "m",
        model_identity="backend:m-v1",
        training_fingerprint="train",
        cache_path=path,
    )
    np.testing.assert_allclose(b.encode(texts), expected)
    assert second.calls == []
    b.encode(['["A2"]'])
    assert len(second.calls) == 1
    b.close()
    for model_identity, split in [
        ("backend:m-v2", "train"),
        ("backend:m-v1", "other-train"),
    ]:
        client = EmbeddingClient()
        cache = CachedEmbeddings(
            client,
            "m",
            model_identity=model_identity,
            training_fingerprint=split,
            cache_path=path,
        )
        cache.encode(texts)
        assert len(client.calls) == 1
        cache.close()


@pytest.mark.parametrize(
    "vectors", [[[0, 0]], [[float("nan"), 1]], [[1, 0], [1, 0, 0]]]
)
def test_invalid_embedding_vectors_fail_without_ranking(vectors):
    client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **_: SimpleNamespace(
                data=[
                    SimpleNamespace(index=i, embedding=v) for i, v in enumerate(vectors)
                ],
                usage=None,
            )
        )
    )
    cache = CachedEmbeddings(
        client, "m", model_identity="fake", training_fingerprint="s"
    )
    with pytest.raises(ValueError):
        cache.encode([f'["item{i}"]' for i in range(len(vectors))])
    assert not cache.statistics()["requests"][0]["completed"]
