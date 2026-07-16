from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd

from beyond_click_sim.tasks import Task, TaskSchema


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.in_distribution.interaction_prediction.methods import (  # noqa: E402
    agent4rec_listwise_ranking as interaction_agent4rec_listwise,
)
from runners.in_distribution.preference_prediction.methods import (  # noqa: E402
    METHOD_RUNNERS as PREFERENCE_METHOD_RUNNERS,
)
from runners.in_distribution.preference_prediction.methods import (  # noqa: E402
    agent4rec_listwise_ranking as preference_agent4rec_listwise,
)
from runners.in_distribution.preference_prediction.methods import (  # noqa: E402
    llm_listwise_ranking as preference_llm_listwise,
)


class FakeChatCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Fake LLM response queue is empty")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.responses.pop(0)),
                )
            ]
        )


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        completions = FakeChatCompletions(responses)
        self.chat = SimpleNamespace(completions=completions)
        self.completions = completions


def _task(*, preference: bool) -> Task:
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_id": ["h1", "h2"],
            "item_title": ["Toy Story", "Heat"],
            "item_genres": ["Animation", "Crime"],
            "rating": [5, 2],
            "target": [1, 0] if preference else [1, 1],
        }
    )

    def split(prefix: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": [f"{prefix}1", f"{prefix}2"],
                "candidate_group": [f"{prefix}g1", f"{prefix}g1"],
                "item_title": ["Aladdin", "Casino"],
                "item_genres": ["Animation", "Crime"],
                "item_rating_mean": [4.1, 3.5],
                "target": [1, 0],
            }
        )

    return Task(
        name="preference-toy" if preference else "interaction-toy",
        train=train,
        val=split("v"),
        test=split("t"),
        schema=TaskSchema(
            target_column="target",
            candidate_group_column="candidate_group",
            feature_columns=("item_title", "item_genres"),
            history_context_columns=("rating",),
        ),
        manifest={
            "dataset": "ml-1m",
            "dataset_version": "v1",
            "target_source_column": (
                "target_like_ge4" if preference else "target_interact"
            ),
            "splitter": {"seed": 0},
        },
    )


def test_preference_history_listwise_runner_uses_target_and_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = FakeClient(["1. C1\n2. C2", "1. C1\n2. C2"])
    monkeypatch.setattr(preference_llm_listwise, "make_llm_client", lambda _: client)
    monkeypatch.setattr(preference_llm_listwise, "repo_root", lambda: REPO_ROOT)

    result = preference_llm_listwise.run_method(
        _task(preference=True),
        tmp_path,
        method_name="preference-listwise-test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_workers=1,
        max_llm_attempts=1,
        extra_body={},
        serving_metadata={},
        source_metadata={},
    )

    assert result["pointwise_threshold"] == 1.0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"]["class"] == "LLMPreferenceListwiseRankingScorer"
    prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "Positive-preference target:" in prompt
    assert "rate the candidate movie at least 4 out of 5" in prompt


def test_interaction_agent4rec_listwise_runner_uses_profiles_and_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = FakeClient(["1. C1\n2. C2", "1. C1\n2. C2"])
    monkeypatch.setattr(
        interaction_agent4rec_listwise,
        "make_llm_client",
        lambda _: client,
    )
    monkeypatch.setattr(
        interaction_agent4rec_listwise,
        "repo_root",
        lambda: REPO_ROOT,
    )

    result = interaction_agent4rec_listwise.run_method(
        _task(preference=False),
        tmp_path,
        method_name="agent4rec-interaction-listwise-test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_workers=1,
        max_llm_attempts=1,
        profile_components=("traits",),
        summary_usage="none",
    )

    assert result["pointwise_threshold"] == 1.0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"]["class"] == "Agent4RecListwiseRankingScorer"
    prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "##recommended list##" in prompt
    assert "Use every candidate ID exactly once" in prompt


def test_preference_agent4rec_listwise_runner_uses_target_and_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = FakeClient(["1. C1\n2. C2", "1. C1\n2. C2"])
    monkeypatch.setattr(
        preference_agent4rec_listwise,
        "make_llm_client",
        lambda _: client,
    )
    monkeypatch.setattr(
        preference_agent4rec_listwise,
        "repo_root",
        lambda: REPO_ROOT,
    )

    result = preference_agent4rec_listwise.run_method(
        _task(preference=True),
        tmp_path,
        method_name="agent4rec-preference-listwise-test",
        client_name="fake",
        model="fake-model",
        max_candidate_groups=None,
        max_workers=1,
        max_llm_attempts=1,
        profile_components=("traits",),
        summary_usage="none",
        extra_body={},
        serving_metadata={},
        source_metadata={},
    )

    assert result["pointwise_threshold"] == 1.0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert (
        manifest["scorer"]["class"]
        == "Agent4RecPreferenceListwiseRankingScorer"
    )
    prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "Positive-preference target:" in prompt
    assert "rate the candidate movie at least 4 out of 5" in prompt


def test_preference_listwise_methods_are_registered() -> None:
    assert (
        "llm_preference_listwise_ranking_vllm_qwen36_27b_full"
        in PREFERENCE_METHOD_RUNNERS
    )
    assert (
        "agent4rec_preference_listwise_ranking_vllm_qwen36_27b_traits_full"
        in PREFERENCE_METHOD_RUNNERS
    )


def test_preference_history_listwise_smoke_uses_direct_vllm_without_item_stats(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_method(
        task: Task,
        output_dir: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(preference_llm_listwise, "run_method", fake_run_method)

    preference_llm_listwise.run_qwen36_27b_smoke(
        _task(preference=True),
        tmp_path,
    )

    assert captured["client_name"] == "vllm_local"
    assert captured["use_item_stats"] is False
