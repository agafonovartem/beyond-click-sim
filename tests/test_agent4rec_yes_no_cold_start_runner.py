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

from runners.in_distribution.cold_start.methods import agent4rec_yes_no  # noqa: E402


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
# Task fixture helpers
# ---------------------------------------------------------------------------


def _ml1m_cold_task(
    *,
    history_user_id: str = "u1",
    train_rows: list[dict] | None = None,
    include_item_rating_mean: bool = False,
    dataset: str = "ml-1m",
) -> ColdStartTask:
    """Minimal two-item ml-1m ColdStartTask.

    Cold user 'u1' has three history items in online_session_history.
    train contains only warm-user rows (no u1), matching the cold-start contract.
    """
    base_history = [
        {
            "user_id": history_user_id,
            "item_id": "ih1",
            "item_title": "Toy Story",
            "item_genres": "Animation|Comedy",
            "item_rating_mean": 4.15,
            "rating": 5,
            "target": 1,
        },
        {
            "user_id": history_user_id,
            "item_id": "ih2",
            "item_title": "Aladdin",
            "item_genres": "Animation",
            "item_rating_mean": 3.95,
            "rating": 4,
            "target": 1,
        },
        {
            "user_id": history_user_id,
            "item_id": "ih3",
            "item_title": "Heat",
            "item_genres": "Crime",
            "item_rating_mean": 3.60,
            "rating": 2,
            "target": 1,
        },
    ]

    history_df = pd.DataFrame(base_history)
    if not include_item_rating_mean:
        history_df = history_df.drop(columns=["item_rating_mean"])

    test_data: dict[str, list] = {
        "user_id": [history_user_id, history_user_id],
        "item_id": ["it1", "it2"],
        "candidate_group": ["g1", "g1"],
        "item_title": ["Lion King", "Godfather"],
        "item_genres": ["Animation", "Crime"],
        "sampled": [False, True],
        "target": [1, 0],
    }
    if include_item_rating_mean:
        test_data["item_rating_mean"] = [4.15, 4.57]

    test_df = pd.DataFrame(test_data)

    if train_rows is not None:
        train_df = pd.DataFrame(train_rows)
    else:
        # Default: warm user; no cold-user rows in train (cold-start contract).
        warm_train = [
            {
                "user_id": "u_warm",
                "item_id": "iw1",
                "item_title": "Terminator",
                "item_genres": "Action",
                "item_rating_mean": 4.00,
                "rating": 4,
                "target": 1,
            }
        ]
        train_df = pd.DataFrame(warm_train)
        if not include_item_rating_mean:
            train_df = train_df.drop(columns=["item_rating_mean"])

    return ColdStartTask(
        name="test_task",
        train=train_df,
        val=pd.DataFrame(columns=["user_id", "item_id", "target"]),
        test=test_df,
        online_session_history=history_df,
        k=3,
        schema=TaskSchema(
            target_column="target",
            candidate_group_column="candidate_group",
            sampled_column="sampled",
            feature_columns=("user_id", "item_id"),
        ),
        manifest={
            "dataset": dataset,
            "dataset_version": "v1",
            "splitter": {"seed": 42},
        },
    )


# ---------------------------------------------------------------------------
# Happy path — traits-only, ml-1m (basic columns, no item_rating_mean)
# ---------------------------------------------------------------------------


def test_cold_start_agent4rec_runner_writes_all_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    client = FakeClient(
        [
            "ID: C2; MOVIE: Godfather; WATCH: no; REASON: crime movies do not fit\n"
            "ID: C1; MOVIE: Lion King; WATCH: yes; REASON: animated fits the taste"
        ]
    )
    monkeypatch.setattr(agent4rec_yes_no, "make_llm_client", lambda _: client)

    task = _ml1m_cold_task()
    result = agent4rec_yes_no.run_method(
        task,
        tmp_path,
        method_name="agent4rec_yes_no_cold_start_test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
    )

    assert result["scored_rows"] == 2
    assert result["requested_rows"] == 2
    assert result["llm_errors"] == 0

    assert (tmp_path / "predictions.parquet").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "llm_errors.jsonl").exists()

    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert predictions["score"].tolist() == [1.0, 0.0]
    assert predictions["prediction"].tolist() == [True, False]


# ---------------------------------------------------------------------------
# Critical validity: scorer is fitted on online_session_history, not train
# ---------------------------------------------------------------------------


def test_cold_start_agent4rec_runner_fits_on_online_session_history_not_train(
    tmp_path: Path, monkeypatch
) -> None:
    """Cold user 'u1' must NOT appear in train; the scorer must still produce
    valid scores for 'u1' by fitting on online_session_history.

    If run_method accidentally fit on task.train (which has no 'u1' rows), the
    profile generator would have no user profile for 'u1', and all candidate
    groups for 'u1' would fail to score, giving scored_rows == 0.
    """
    client = FakeClient(
        [
            "ID: C2; MOVIE: Godfather; WATCH: no; REASON: not aligned\n"
            "ID: C1; MOVIE: Lion King; WATCH: yes; REASON: fits taste"
        ]
    )
    monkeypatch.setattr(agent4rec_yes_no, "make_llm_client", lambda _: client)

    # Train is empty (no warm users) — cold-start scenario where all users are cold.
    # task_xy still calls split_xy on train, so it needs the target column schema.
    # If the scorer accidentally fit on this empty train, the profile generator
    # would have no user profile for 'u1', giving scored_rows == 0.
    task = _ml1m_cold_task(
        train_rows=[{"user_id": "placeholder", "item_id": "placeholder",
                     "item_title": "placeholder", "item_genres": "placeholder",
                     "rating": 0, "target": 0}]
    )
    # Drop the placeholder row so train is empty but schema-valid.
    task = ColdStartTask(
        name=task.name,
        train=task.train.iloc[:0].copy(),
        val=task.val,
        test=task.test,
        online_session_history=task.online_session_history,
        k=task.k,
        schema=task.schema,
        manifest=task.manifest,
    )
    result = agent4rec_yes_no.run_method(
        task,
        tmp_path,
        method_name="test_fit_on_history",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
    )

    assert result["scored_rows"] == 2, (
        "Scored rows must be 2 even with empty train, proving the scorer used "
        "online_session_history rather than task.train."
    )
    assert result["llm_errors"] == 0


# ---------------------------------------------------------------------------
# Manifest correctness
# ---------------------------------------------------------------------------


def test_cold_start_agent4rec_runner_manifest_records_fit_on_history_and_k(
    tmp_path: Path, monkeypatch
) -> None:
    client = FakeClient(
        [
            "ID: C2; MOVIE: Godfather; WATCH: no; REASON: not fit\n"
            "ID: C1; MOVIE: Lion King; WATCH: yes; REASON: fits"
        ]
    )
    monkeypatch.setattr(agent4rec_yes_no, "make_llm_client", lambda _: client)

    task = _ml1m_cold_task()
    agent4rec_yes_no.run_method(
        task,
        tmp_path,
        method_name="test_manifest",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"]["fit_on"] == "online_session_history"
    assert manifest["scorer"]["k"] == task.k
    assert manifest["scorer"]["class"] == "Agent4RecYesNoScorer"
    # Basic ml-1m config must NOT include item_rating_mean
    assert manifest["scorer"]["candidate_description_columns"] == [
        "item_title",
        "item_genres",
    ]


# ---------------------------------------------------------------------------
# item_stats path — rating_mean column selected in prompt
# ---------------------------------------------------------------------------


def test_cold_start_agent4rec_runner_item_stats_includes_rating_mean_column(
    tmp_path: Path, monkeypatch
) -> None:
    client = FakeClient(
        [
            "ID: C2; MOVIE: Godfather; WATCH: no; REASON: not fit\n"
            "ID: C1; MOVIE: Lion King; WATCH: yes; REASON: fits"
        ]
    )
    monkeypatch.setattr(agent4rec_yes_no, "make_llm_client", lambda _: client)

    task = _ml1m_cold_task(include_item_rating_mean=True)
    agent4rec_yes_no.run_method(
        task,
        tmp_path,
        method_name="test_item_stats",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
        use_item_stats=True,
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"]["use_item_stats"] is True
    assert "item_rating_mean" in manifest["scorer"]["candidate_description_columns"]

    user_prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "History ratings" in user_prompt


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_cold_start_agent4rec_runner_raises_for_unknown_dataset(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(agent4rec_yes_no, "make_llm_client", lambda _: FakeClient([]))
    task = _ml1m_cold_task(dataset="steam")  # steam not supported in cold-start config

    with pytest.raises(ValueError, match="no prompt config"):
        agent4rec_yes_no.run_method(
            task,
            tmp_path,
            method_name="test_bad_dataset",
            client_name="fake",
            model="fake-model",
            max_candidate_groups=None,
        )


def test_cold_start_agent4rec_runner_raises_without_candidate_group_column(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(agent4rec_yes_no, "make_llm_client", lambda _: FakeClient([]))
    task = _ml1m_cold_task()
    # Rebuild with candidate_group_column=None
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
        agent4rec_yes_no.run_method(
            task,
            tmp_path,
            method_name="test_no_group_col",
            client_name="fake",
            model="fake-model",
            max_candidate_groups=None,
        )


# ---------------------------------------------------------------------------
# Taste profile path
# ---------------------------------------------------------------------------


def test_cold_start_agent4rec_runner_taste_profile_writes_cache(
    tmp_path: Path, monkeypatch
) -> None:
    scoring_client = FakeClient(
        [
            "ID: C2; MOVIE: Godfather; WATCH: no; REASON: not animated\n"
            "ID: C1; MOVIE: Lion King; WATCH: yes; REASON: animated taste"
        ]
    )
    taste_client = FakeClient(
        [
            "TASTE: I enjoy animated family movies.\n"
            "REASON: I rated animated movies highly.\n"
            "HIGH RATINGS: animated family movies\n"
            "LOW RATINGS: crime movies"
        ]
    )

    def _fake_make_llm_client(client_name: str) -> FakeClient:
        if client_name == "openai":
            return taste_client
        if client_name == "fake":
            return scoring_client
        raise AssertionError(f"unexpected client: {client_name!r}")

    monkeypatch.setattr(agent4rec_yes_no, "make_llm_client", _fake_make_llm_client)
    cache_path = tmp_path / "taste-cache.jsonl"
    monkeypatch.setattr(
        agent4rec_yes_no,
        "agent4rec_taste_cache_path",
        lambda *_a, **_kw: cache_path,
    )

    task = _ml1m_cold_task()
    result = agent4rec_yes_no.run_method(
        task,
        tmp_path,
        method_name="test_taste",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_llm_attempts=1,
        max_workers=1,
        profile_components=("traits", "taste"),
        taste_client_name="openai",
        taste_model="gpt-4o-mini",
    )

    assert result["scored_rows"] == 2
    assert len(taste_client.completions.calls) == 1
    assert cache_path.exists()

    cached_rows = [
        json.loads(line)
        for line in cache_path.read_text(encoding="utf-8").splitlines()
    ]
    assert cached_rows[0]["history_item_ids"] == ["ih1", "ih2", "ih3"]

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    profile_manifest = manifest["scorer"]["profile_generator"]
    assert profile_manifest["profile_components"] == ["traits", "taste"]
    assert profile_manifest["taste"]["client_name"] == "openai"
    assert profile_manifest["taste"]["model"] == "gpt-4o-mini"


def test_cold_start_agent4rec_runner_taste_requires_client_name(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(agent4rec_yes_no, "make_llm_client", lambda _: FakeClient([]))
    task = _ml1m_cold_task()

    with pytest.raises(ValueError, match="taste_client_name"):
        agent4rec_yes_no.run_method(
            task,
            tmp_path,
            method_name="test",
            client_name="fake",
            model="fake-model",
            max_candidate_groups=None,
            profile_components=("traits", "taste"),
            taste_model="gpt-4o-mini",
        )


# ---------------------------------------------------------------------------
# agent4rec_taste_cache_path — path construction
# ---------------------------------------------------------------------------


def test_agent4rec_taste_cache_path_builds_expected_filename(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        agent4rec_yes_no,
        "repo_root",
        lambda: tmp_path,
    )
    task = _ml1m_cold_task()

    path = agent4rec_yes_no.agent4rec_taste_cache_path(
        task,
        taste_model="gpt-4o-mini",
        taste_prompt_version="agent4rec_taste_v1",
    )

    assert path.parent == tmp_path / "outputs" / "agent4rec_taste_cache"
    assert path.name == "ml-1m_v1_seed42_gpt-4o-mini_agent4rec_taste_v1.jsonl"


def test_agent4rec_taste_cache_path_slugifies_model_name(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(agent4rec_yes_no, "repo_root", lambda: tmp_path)
    task = _ml1m_cold_task()

    path = agent4rec_yes_no.agent4rec_taste_cache_path(
        task,
        taste_model="Qwen/Qwen3.6-27B",
        taste_prompt_version="v1",
    )

    assert "Qwen-Qwen3.6-27B" in path.name
    assert "/" not in path.name


# ---------------------------------------------------------------------------
# _cache_slug — unit tests
# ---------------------------------------------------------------------------


def test_cache_slug_leaves_safe_characters_unchanged() -> None:
    assert agent4rec_yes_no._cache_slug("gpt-4o-mini") == "gpt-4o-mini"
    assert agent4rec_yes_no._cache_slug("llama3.1:8b") == "llama3.1-8b"
    assert agent4rec_yes_no._cache_slug("Qwen/Qwen3.6-27B") == "Qwen-Qwen3.6-27B"


def test_cache_slug_raises_on_empty_result() -> None:
    with pytest.raises(ValueError, match="Cannot build cache slug"):
        agent4rec_yes_no._cache_slug("///")


# ---------------------------------------------------------------------------
# Wrapper functions — smoke & full kwargs
# ---------------------------------------------------------------------------


def test_cold_start_agent4rec_qwen_smoke_and_full_wrappers(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run_method(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(agent4rec_yes_no, "run_method", _fake_run_method)
    task = SimpleNamespace()

    agent4rec_yes_no.run_qwen36_27b_smoke(task, tmp_path)
    agent4rec_yes_no.run_qwen36_27b_full(task, tmp_path)

    assert calls == [
        {
            "method_name": "agent4rec_yes_no_vllm_qwen36_27b_smoke",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": 25,
            "max_workers": agent4rec_yes_no.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        {
            "method_name": "agent4rec_yes_no_vllm_qwen36_27b_full",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": None,
            "max_workers": agent4rec_yes_no.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    ]


def test_cold_start_agent4rec_qwen_traits_taste_wrappers(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run_method(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(agent4rec_yes_no, "run_method", _fake_run_method)
    task = SimpleNamespace()

    agent4rec_yes_no.run_qwen36_27b_traits_taste_gpt4o_mini_smoke(task, tmp_path)
    agent4rec_yes_no.run_qwen36_27b_traits_taste_gpt4o_mini_full(task, tmp_path)

    assert calls == [
        {
            "method_name": "agent4rec_yes_no_vllm_qwen36_27b_traits_taste_gpt4o_mini_smoke",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": 25,
            "max_workers": agent4rec_yes_no.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "profile_components": ("traits", "taste"),
            "taste_client_name": "openai",
            "taste_model": "gpt-4o-mini",
        },
        {
            "method_name": "agent4rec_yes_no_vllm_qwen36_27b_traits_taste_gpt4o_mini_full",
            "client_name": "vllm_local",
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": None,
            "max_workers": agent4rec_yes_no.VLLM_MAX_WORKERS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "profile_components": ("traits", "taste"),
            "taste_client_name": "openai",
            "taste_model": "gpt-4o-mini",
        },
    ]
