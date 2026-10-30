from copy import deepcopy
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from beyond_click_sim.scorers.agent4rec.profiles import (
    Agent4RecProfileGenerator,
    _three_tier_group,
    parse_agent4rec_modify_taste_response,
)
from beyond_click_sim.scorers.agent4rec.yes_no import (
    Agent4RecYesNoScorer,
    Agent4RecPreferenceYesNoScorer,
)
from beyond_click_sim.scorers.agent4rec.listwise import (
    Agent4RecListwiseRankingScorer,
    Agent4RecPreferenceListwiseRankingScorer,
)
from beyond_click_sim.scorers.agent4rec.regression import Agent4RecRegressor
from beyond_click_sim.tasks import Task, TaskSchema
from runners.in_distribution.agent4rec_items import (
    prepare_item_profiles,
    prepare_agent4rec_task,
    GENERATED_GENRES_COLUMN,
)
from runners.in_distribution.agent4rec_version import (
    compatible_result,
    protect_output_directory,
)

PATTERNS = "TASTE: I like dramas.\nREASON: Observed ratings.\nHIGH RATINGS: Thoughtful stories\nLOW RATINGS: Insufficient evidence"


class FakeClient:
    def __init__(self, response=PATTERNS):
        self.response, self.calls = response, []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **request):
        self.calls.append(deepcopy(request))
        value = self.response(request) if callable(self.response) else self.response
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=value))]
        )


def history():
    rows = [
        dict(
            user_id="a",
            item_id=f"i{i:02d}",
            item_title=f"Movie{i:02d}",
            item_genres="Old" if i == 0 else "Drama",
            timestamp=i,
            rating=2 if i == 0 else 3,
        )
        for i in range(26)
    ]
    rows += [
        dict(
            user_id="b",
            item_id="i00",
            item_title="Movie00",
            item_genres="Old",
            timestamp=1,
            rating=5,
        ),
        dict(
            user_id="c",
            item_id="other",
            item_title="Other",
            item_genres="Comedy|Action|Drama",
            timestamp=0,
            rating=4,
        ),
    ]
    return pd.DataFrame(rows).sample(frac=1, random_state=3).reset_index(drop=True)


def generator(tmp_path, client, **kwargs):
    return Agent4RecProfileGenerator(
        profile_components=("traits", "taste"),
        taste_client=client,
        taste_client_name="fake",
        taste_model="profile-model",
        taste_cache_path=tmp_path / "taste.jsonl",
        **kwargs,
    )


def test_full_trait_history_separate_from_recent_taste_and_population_invariant(
    tmp_path,
):
    X = history()
    client = FakeClient()
    pg = generator(tmp_path, client)
    scorer = Agent4RecYesNoScorer(
        client,
        "response-model",
        profile_generator=pg,
        candidate_description_columns=("item_title",),
    ).fit(X, pd.Series(1, index=X.index), profile_user_ids=["a"])
    assert scorer.history_by_user_["a"].item_ids == tuple(
        f"i{i:02d}" for i in range(6, 26)
    )
    evidence = pg.manifest()["trait_evidence"]
    assert evidence["rows_by_user"]["a"] == 26
    assert evidence["values_by_trait"]["activity"]["a"] == 26
    assert evidence["values_by_trait"]["diversity"]["a"] == 2
    assert evidence["conformity_item_means"]["i00"] == 3.5
    assert evidence["values_by_trait"]["conformity"]["a"] == pytest.approx(2.25 / 26)
    all_pg = Agent4RecProfileGenerator()
    all_profiles = all_pg.build_traits(X, pd.Series(0, index=X.index))
    assert pg.trait_thresholds_ == all_pg.trait_thresholds_
    assert scorer.profile_by_user_["a"] == all_profiles["a"]
    scorer.build_taste(pd.DataFrame({"user_id": ["a"], "rating": [1], "target": [0]}))
    prompt = client.calls[0]["messages"][1]["content"]
    assert "Movie00" not in prompt and "Movie05" not in prompt
    assert prompt.index("Movie06") < prompt.index("Movie25")
    assert "Ratings 3-5 are liked; ratings 1-2 are disliked" in prompt
    before = scorer.profile_by_user_["a"]
    scorer.build_taste(pd.DataFrame({"user_id": ["a"], "rating": [5], "target": [1]}))
    assert scorer.profile_by_user_["a"] == before
    assert len(client.calls) == 1
    assert pg.manifest()["taste_evidence"]["rows_by_user"] == {"a": 20}


def test_tier_boundaries_missing_and_constant_traits():
    assert [_three_tier_group(v, low=1, high=2) for v in (0, 1, 1.5, 2, 3)] == [
        1,
        1,
        2,
        2,
        3,
    ]
    X = history().iloc[:2].copy()
    X["user_id"] = ["a", "b"]
    X["rating"] = 5
    X["item_genres"] = "Drama"
    pg = Agent4RecProfileGenerator()
    profiles = pg.build_traits(X, pd.Series([1, 1]))
    assert all(
        p.activity_group is None
        and p.conformity_group is None
        and p.diversity_group is None
        for p in profiles.values()
    )
    assert set(pg.manifest()["trait_evidence"]["omitted_traits"]["a"].values()) == {
        "constant_reference_population"
    }
    X.loc[X.index[0], "item_genres"] = None
    pg.build_traits(X, pd.Series([1, 1]))
    assert "diversity" in pg.manifest()["trait_evidence"]["omitted_traits"]["a"]


@pytest.mark.parametrize(
    "change",
    ["rating", "title", "model", "temperature", "tokens", "prompt_version", "client"],
)
def test_taste_cache_fingerprints_all_generation_inputs(tmp_path, change):
    X = history()
    client = FakeClient()
    pg = generator(tmp_path, client)
    h = pg.select_taste_histories(
        X, user_column="user_id", item_column="item_id", max_history_items=20
    )
    from beyond_click_sim.scorers.agent4rec.profiles import Agent4RecUserProfile

    profiles = {"a": Agent4RecUserProfile("a")}
    first = pg.build_taste(profiles=profiles, histories=h, user_ids=["a"])
    cached = pg.build_taste(profiles=profiles, histories=h, user_ids=["a"])
    assert first == cached and len(client.calls) == 1
    assert cached["a"].high_rating.strip() == "Thoughtful stories"
    assert cached["a"].low_rating.strip() == "Insufficient evidence"
    assert cached["a"].taste_reason.strip() == "Observed ratings."
    if change in ("rating", "title"):
        column = "rating" if change == "rating" else "item_title"
        h["a"].rows.loc[h["a"].rows.index[-1], column] = (
            5 if change == "rating" else "Changed title"
        )
    else:
        attr, value = {
            "model": ("taste_model", "new"),
            "temperature": ("taste_temperature", 0.4),
            "tokens": ("taste_max_tokens", 123),
            "prompt_version": ("taste_prompt_version", "new"),
            "client": ("taste_client_name", "different"),
        }[change]
        setattr(pg, attr, value)
    pg.build_taste(profiles=profiles, histories=h, user_ids=["a"])
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    "scorer_kind",
    ["interaction", "preference", "ranking", "preference_ranking", "regression"],
)
def test_patterns_reach_all_scorers_on_cache_hit_and_miss(tmp_path, scorer_kind):
    X = history()
    taste = FakeClient()
    pg = generator(tmp_path, taste)
    config = dict(
        client=FakeClient(),
        model="response",
        profile_generator=pg,
        candidate_description_columns=("item_title",),
    )
    if scorer_kind == "regression":
        scorer = Agent4RecRegressor(
            **config, target_description="1 to 5", valid_values=(1, 2, 3, 4, 5)
        )
    else:
        klass = {
            "interaction": Agent4RecYesNoScorer,
            "preference": Agent4RecPreferenceYesNoScorer,
            "ranking": Agent4RecListwiseRankingScorer,
            "preference_ranking": Agent4RecPreferenceListwiseRankingScorer,
        }[scorer_kind]
        if "preference" in scorer_kind:
            config["target_description"] = "rating at least 4"
        scorer = klass(**config)
    scorer.fit(X, pd.Series(1, index=X.index), profile_user_ids=["a"])
    eval_frame = pd.DataFrame(
        {"user_id": ["a"], "item_title": ["Candidate"], "candidate_group": ["g"]}
    )
    for _ in range(2):
        scorer.build_taste(eval_frame)
        if scorer_kind == "regression":
            messages = scorer._build_messages(row=eval_frame.iloc[0])
        else:
            messages = scorer._build_messages(
                user_id="a", candidates=eval_frame, candidate_labels=["C1"]
            )
        assert "Thoughtful stories" in messages[0]["content"]
        assert "Insufficient evidence" in messages[0]["content"]
        assert "Observed ratings." not in messages[0]["content"]
    assert len(taste.calls) == 1


def test_patterns_required_and_itemwise_preference_target_is_explicit():
    with pytest.raises(ValueError, match="both HIGH and LOW"):
        parse_agent4rec_modify_taste_response("TASTE: dramas\nHIGH RATINGS: drama")
    scorer = Agent4RecPreferenceYesNoScorer(
        client=FakeClient("yes"),
        model="response",
        target_description="rating >= 4",
        candidate_description_columns=("item_title",),
        prompt_style="itemwise",
    )
    assert "rating >= 4" in scorer._build_itemwise_user_prompt(
        candidate="A", taste=None
    )
    for klass in (
        Agent4RecListwiseRankingScorer,
        Agent4RecPreferenceListwiseRankingScorer,
    ):
        kwargs = (
            {"target_description": "rating >= 4"}
            if klass is Agent4RecPreferenceListwiseRankingScorer
            else {}
        )
        with pytest.raises(ValueError, match="batch mode"):
            klass(
                client=FakeClient(),
                model="r",
                candidate_description_columns=("item_title",),
                prompt_style="itemwise",
                **kwargs,
            )


def toy_task():
    train = history().assign(target=1)
    test = pd.DataFrame(
        [
            dict(
                user_id="a",
                item_id=i,
                item_title=t,
                item_genres="Canonical",
                candidate_group="g",
                target=int(j < 2),
                sampled=j >= 2,
                rating=99,
            )
            for j, (i, t) in enumerate(
                [
                    ("i00", "Movie00"),
                    ("i01", "Movie01"),
                    ("x2", "X2"),
                    ("x3", "X3"),
                    ("x4", "X4"),
                    ("x5", "X5"),
                ]
            )
        ]
    )
    return Task(
        "fixture",
        train,
        test.iloc[:0].copy(),
        test,
        TaskSchema(candidate_group_column="candidate_group", sampled_column="sampled"),
        {
            "dataset": "ml-1m",
            "dataset_version": "v1",
            "splitter": {"seed": 0},
            "target_source_column": "target_interact",
        },
    )


def metadata_artifact(tmp_path, task):
    client = FakeClient(
        json.dumps({"genres": ["GeneratedDifferent"], "summary": "Title-derived plot."})
    )
    items = pd.concat([task.train, task.val, task.test]).drop_duplicates("item_id")
    path = tmp_path / "items.jsonl"
    prepare_item_profiles(
        items,
        client=client,
        client_name="fake",
        model="metadata-model",
        output_path=path,
    )
    return path, client


def test_metadata_complete_training_only_no_pruning_and_no_shared_mutation(tmp_path):
    task = toy_task()
    original = deepcopy(task)
    path, client = metadata_artifact(tmp_path, task)
    enriched = prepare_agent4rec_task(task, item_profile_path=path)
    pd.testing.assert_frame_equal(task.test, original.test)
    assert enriched.test.item_id.tolist() == task.test.item_id.tolist()
    assert enriched.test[GENERATED_GENRES_COLUMN].eq("GeneratedDifferent").all()
    assert enriched.test.item_genres.eq("Canonical").all()
    assert enriched.test.item_rating_mean.iloc[0] == 3.5
    assert enriched.test.item_rating_count.iloc[0] == 2
    assert enriched.test.item_rating_count.iloc[2] == 0
    assert pd.isna(enriched.test.item_rating_mean.iloc[2])
    changed = deepcopy(task)
    changed.test["target"] = 1 - changed.test.target
    changed.test["rating"] = -500
    again = prepare_agent4rec_task(changed, item_profile_path=path)
    pd.testing.assert_frame_equal(
        enriched.test.drop(columns=["target", "rating"]),
        again.test.drop(columns=["target", "rating"]),
    )
    for call in client.calls:
        assert len(call["messages"]) == 2
        assert call["messages"][1]["content"].startswith("Movie title: ")
        assert "99" not in call["messages"][1]["content"]
    # Freeze and reuse metadata; changed generation settings invalidate the cache.
    n = len(client.calls)
    items = pd.concat([task.train, task.test]).drop_duplicates("item_id")
    prepare_item_profiles(
        items,
        client=client,
        client_name="fake",
        model="metadata-model",
        output_path=path,
    )
    assert len(client.calls) == n
    prepare_item_profiles(
        items.iloc[:1],
        client=client,
        client_name="fake",
        model="other-model",
        output_path=path,
    )
    assert len(client.calls) == n + 1
    with pytest.raises(ValueError, match="Missing Agent4Rec generated item profiles"):
        prepare_agent4rec_task(task, item_profile_path=tmp_path / "absent")


def test_combined_outputs_exact_binary_ties_and_test_label_independence(
    tmp_path, monkeypatch
):
    from runners.in_distribution.interaction_prediction.methods import (
        agent4rec_yes_no as runner,
    )
    from runners.in_distribution.interaction_prediction.methods.common import (
        ranking_metrics_for_split,
    )

    task = toy_task()
    path, _ = metadata_artifact(tmp_path, task)
    monkeypatch.setenv("AGENT4REC_ITEM_PROFILE_PATH", str(path))
    response = "\n".join(
        f"ID: C{i}; MOVIE: item; WATCH: {'yes' if i <= 3 else 'no'}; REASON: profile"
        for i in range(1, 7)
    )
    scoring = FakeClient(response)
    taste = FakeClient()
    monkeypatch.setattr(
        runner, "make_llm_client", lambda name: taste if name == "taste" else scoring
    )
    results = []
    for number in range(2):
        candidate_task = deepcopy(task)
        if number:
            candidate_task.test["target"] = 1 - candidate_task.test.target
        directory = tmp_path / f"run{number}"
        runner.run_method(
            candidate_task,
            directory,
            method_name="agent4rec_fixture",
            client_name="score",
            model="response",
            max_candidate_groups=None,
            profile_components=("traits", "taste"),
            taste_client_name="taste",
            taste_model="profile",
            taste_cache_path=tmp_path / "taste.jsonl",
        )
        saved = pd.read_parquet(directory / "predictions.parquet")
        assert saved.score.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        assert saved.prediction.tolist() == saved.score.astype(bool).tolist()
        metrics = json.loads((directory / "metrics_ranking.json").read_text())
        expected = ranking_metrics_for_split(
            X=candidate_task.test,
            y=candidate_task.test.target,
            scores=pd.Series(saved.score.to_numpy(), index=candidate_task.test.index),
            candidate_group_column="candidate_group",
        )
        assert metrics["test"] == expected
        assert 5 in metrics["ranking_evaluation"]["ks"]
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["decision_rule"]["threshold"] is None
        assert compatible_result(directory, "agent4rec_fixture")
        results.append(manifest["scorer"]["profile_generator"]["trait_evidence"])
    assert results[0] == results[1]
    assert len(scoring.calls) == 2 and len(taste.calls) == 1
    assert scoring.calls[0]["messages"] == scoring.calls[1]["messages"]
    assert (
        "popularity (training rating count):0"
        in scoring.calls[0]["messages"][1]["content"]
    )


def test_old_results_cannot_resume_or_be_overwritten(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"scorer": {"profile_generator": {"implementation_version": "old"}}})
    )
    assert not compatible_result(tmp_path, "agent4rec_fixture")
    assert compatible_result(tmp_path, "llm_yes_no_fixture")
    with pytest.raises(ValueError, match="fresh output directory"):
        protect_output_directory(tmp_path)


def test_materialized_temporal_order_survives_timestamp_column_removal():
    X = history().sort_values("timestamp", kind="stable")
    pg = Agent4RecProfileGenerator()
    with_time = pg.select_taste_histories(
        X, user_column="user_id", item_column="item_id", max_history_items=20
    )
    without = pg.select_taste_histories(
        X.drop(columns="timestamp"),
        user_column="user_id",
        item_column="item_id",
        max_history_items=20,
    )
    assert with_time["a"].item_ids == without["a"].item_ids


@pytest.mark.usefixtures("agent4rec_item_artifact")
@pytest.mark.parametrize(
    "source,field,kind",
    [
        ("target_interact", "WATCH", "Agent4RecYesNoScorer"),
        ("target_like_ge4", "PREFERENCE", "Agent4RecPreferenceYesNoScorer"),
    ],
)
def test_cold_start_routes_target_and_uses_only_k_observations(
    tmp_path, monkeypatch, source, field, kind
):
    from beyond_click_sim.tasks.cold_start import ColdStartTask
    from runners.in_distribution.cold_start.methods import agent4rec_yes_no as runner

    base = toy_task()
    cold = base.train.loc[base.train.user_id.eq("a")].sort_values("timestamp").head(3)
    task = ColdStartTask(
        name=base.name,
        train=base.train.loc[base.train.user_id.ne("a")],
        val=base.val,
        test=base.test,
        schema=base.schema,
        manifest={**base.manifest, "target_source_column": source},
        online_session_history=cold,
        k=3,
    )
    response = "\n".join(
        f"ID: C{i}; MOVIE: item; {field}: yes; REASON: profile" for i in range(1, 7)
    )
    client = FakeClient(response)
    monkeypatch.setattr(runner, "make_llm_client", lambda _: client)
    runner.run_method(
        task,
        tmp_path / "run",
        method_name="agent4rec_cold_fixture",
        client_name="fake",
        model="response",
        max_candidate_groups=None,
        scoring="batch",
    )
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["scorer"]["class"] == kind
    evidence = manifest["scorer"]["profile_generator"]["trait_evidence"]
    assert evidence["rows_by_user"] == {"a": 3}
    assert manifest["scorer"]["profile_generator"]["include_activity"] is False
    assert field in client.calls[0]["messages"][1]["content"]
    if source == "target_like_ge4":
        assert "4 or 5" in client.calls[0]["messages"][1]["content"]
    with pytest.raises(ValueError, match="external/lifetime"):
        runner.run_method(
            task,
            tmp_path / "other",
            method_name="agent4rec_cold_fixture",
            client_name="fake",
            model="response",
            max_candidate_groups=None,
            activity_counts={"a": 1000},
        )


def test_coat_history_budget_and_metadata_restrictions():
    from runners.in_distribution.policy_ranking_agreement_rerank.methods import (
        agent4rec_yes_no,
        llm_yes_no,
    )

    assert agent4rec_yes_no.DATASET_MAX_HISTORY_ITEMS["coat"] == 20
    assert llm_yes_no.DATASET_MAX_HISTORY_ITEMS["coat"] == 20
    X = history().loc[lambda x: x.user_id.eq("a")].drop(columns="timestamp")
    pg = Agent4RecProfileGenerator(
        taste_prompt_kind="coat_rating", include_activity=False
    )
    a = pg.select_taste_histories(
        X, user_column="user_id", item_column="item_id", max_history_items=20
    )
    b = pg.select_taste_histories(
        X.sample(frac=1, random_state=8),
        user_column="user_id",
        item_column="item_id",
        max_history_items=20,
    )
    assert a["a"].item_ids == b["a"].item_ids and len(a["a"].rows) == 20
    assert (
        agent4rec_yes_no.DATASET_PROFILE_GENERATOR_KWARGS["coat"]["include_activity"]
        is False
    )
    assert "item_rating_mean" not in agent4rec_yes_no.DATASET_CANDIDATE_COLUMNS["coat"]
    assert "item_rating_count" not in agent4rec_yes_no.DATASET_CANDIDATE_COLUMNS["coat"]


def test_coat_runner_executes_without_demographic_persona(tmp_path, monkeypatch):
    from runners.in_distribution.policy_ranking_agreement_rerank.methods import (
        agent4rec_yes_no as runner,
    )

    base = toy_task()
    train = base.train.loc[base.train.user_id.eq("a")].drop(columns="timestamp")
    candidates = base.test.iloc[:2].copy()
    candidates["item_style"] = "casual"
    candidates["item_colour"] = "blue"
    candidates["user_age_group"] = "SECRET_DEMOGRAPHIC"
    test = pd.concat(
        [candidates.assign(policy="p1"), candidates.assign(policy="p2")],
        ignore_index=True,
    )
    task = Task(
        "coat_fixture",
        train,
        test.iloc[:0],
        test,
        base.schema,
        {"dataset": "coat", "dataset_version": "v1", "splitter": {"seed": 0}},
    )
    taste = FakeClient()
    scoring = FakeClient(
        "ID: C1; ITEM: coat; PREFERENCE: yes; REASON: fit\nID: C2; ITEM: coat; PREFERENCE: no; REASON: fit"
    )
    monkeypatch.setattr(
        runner, "make_llm_client", lambda name: taste if name == "taste" else scoring
    )
    monkeypatch.setattr(
        runner,
        "agent4rec_taste_cache_path",
        lambda *args, **kwargs: tmp_path / "taste.jsonl",
    )
    runner.run_method(
        task,
        tmp_path / "run",
        method_name="agent4rec_coat_fixture",
        client_name="score",
        model="response",
        max_candidate_groups=None,
        max_workers=1,
        summary_usage="none",
        profile_components=("traits", "taste"),
        taste_client_name="taste",
        taste_model="profile",
    )
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    profile = manifest["scorer"]["profile_generator"]
    assert profile["trait_evidence"]["rows_by_user"] == {"a": 20}
    assert profile["taste_evidence"]["rows_by_user"] == {"a": 20}
    assert profile["include_activity"] is False
    assert len(taste.calls) == 1 and len(scoring.calls) == 2
    assert "Movie00" not in taste.calls[0]["messages"][1]["content"]
    for call in scoring.calls:
        text = str(call["messages"])
        assert "SECRET_DEMOGRAPHIC" not in text
        assert "high rating patterns" in text and "low rating patterns" in text
        assert "quality (training mean" not in text and "popularity" not in text
