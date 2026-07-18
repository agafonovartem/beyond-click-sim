from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from beyond_click_sim.tasks import Task, TaskSchema
from runners.in_distribution.preference_prediction.methods import (
    _grouped_agent4rec_yes_no as grouped_agent4rec_yes_no,
)
from runners.in_distribution.preference_prediction.methods import (
    METHOD_RUNNERS as PREFERENCE_METHOD_RUNNERS,
)
from runners.in_distribution.preference_prediction.methods import (
    agent4rec_yes_no as preference_agent4rec_yes_no,
)


class _FakeChatCompletions:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.response))]
        )


class _FakeClient:
    def __init__(self, response: str) -> None:
        completions = _FakeChatCompletions(response)
        self.chat = SimpleNamespace(completions=completions)
        self.completions = completions


def test_preference_agent4rec_runner_records_target_and_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = _FakeClient(
        "ID: C1; MOVIE: Aladdin; PREFERENCE: yes; REASON: likely high rating\n"
        "ID: C2; MOVIE: Casino; PREFERENCE: no; REASON: likely low rating"
    )
    monkeypatch.setattr(
        grouped_agent4rec_yes_no,
        "make_llm_client",
        lambda _: client,
    )
    task = Task(
        name="ml-1m_preference_toy",
        train=pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1"],
                "item_id": ["h1", "h2", "h3"],
                "item_title": ["Toy Story", "Heat", "Aladdin"],
                "item_genres": ["Animation", "Crime", "Animation"],
                "item_summary": [
                    "Toys come alive when people are absent.",
                    "A detective pursues a disciplined thief.",
                    "A street thief discovers a magical lamp.",
                ],
                "rating": [5, 2, 4],
                "target": [1, 0, 1],
            }
        ),
        val=pd.DataFrame(
            columns=["user_id", "item_id", "item_summary", "target"]
        ),
        test=pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["i1", "i2"],
                "candidate_group": ["g1", "g1"],
                "item_title": ["Aladdin", "Casino"],
                "item_genres": ["Animation", "Crime"],
                "item_summary": [
                    "A street thief discovers a magical lamp.",
                    "A mob-connected casino manager loses control.",
                ],
                "target": [1, 0],
            }
        ),
        schema=TaskSchema(
            target_column="target",
            candidate_group_column="candidate_group",
            feature_columns=("item_title", "item_genres"),
            history_context_columns=("rating",),
        ),
        manifest={
            "dataset": "ml-1m",
            "dataset_version": "v1",
            "target_source_column": "target_like_ge4",
            "splitter": {"seed": 0},
            "item_enrichment": {
                "movie_summaries": {
                    "enabled": True,
                    "canonical_column": "summary",
                    "task_column": "item_summary",
                    "source_sha256": "fake-sha256",
                }
            },
        },
    )

    result = preference_agent4rec_yes_no.run_method(
        task,
        tmp_path,
        method_name="agent4rec-preference-test",
        model="Qwen/Qwen3-8B",
        max_candidate_groups=None,
        max_workers=1,
    )

    assert result["coverage"]["scored_fraction"] == 1.0
    assert result["test"]["micro"]["f1"] == 1.0
    prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "rate the candidate movie at least 4 out of 5" in prompt
    assert "PREFERENCE: [yes or no]" in prompt
    assert "WATCH:" not in prompt
    assert "Toy Story" not in prompt
    assert "A street thief discovers a magical lamp." in prompt

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"]["class"] == "Agent4RecPreferenceYesNoScorer"
    assert manifest["scorer"]["profile_generator"]["profile_components"] == [
        "traits"
    ]
    assert manifest["scorer"]["candidate_description_columns"] == [
        "item_title",
        "item_genres",
        "item_summary",
    ]
    assert manifest["scorer"]["summary_usage"] == "candidate"
    assert manifest["scorer"]["scorer_kwargs"] == {
        "target_description": (
            "The user would rate the candidate movie at least 4 out of 5."
        )
    }
    assert manifest["decision_rule"]["parser_contract"] == (
        "agent4rec_labeled_id_entity_preference_reason"
    )
    assert manifest["scorer"]["serving"]["backend"] == (
        "litellm_proxy_over_vllm"
    )
    assert manifest["source"] == {
        "base_git_commit": None,
        "diff_sha256": None,
        "snapshot_sha256": None,
    }


def test_preference_agent4rec_qwen_smoke_methods_are_registered() -> None:
    assert PREFERENCE_METHOD_RUNNERS[
        "agent4rec_preference_yes_no_litellm_qwen3_8b_traits_smoke"
    ] is preference_agent4rec_yes_no.run_qwen3_8b_smoke
    assert PREFERENCE_METHOD_RUNNERS[
        "agent4rec_preference_yes_no_litellm_qwen36_27b_traits_smoke"
    ] is preference_agent4rec_yes_no.run_qwen36_27b_smoke


def test_preference_agent4rec_qwen_wrappers_use_planned_models(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_method(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(preference_agent4rec_yes_no, "run_method", fake_run_method)
    task = SimpleNamespace()

    preference_agent4rec_yes_no.run_qwen3_8b_smoke(task, tmp_path)
    preference_agent4rec_yes_no.run_qwen36_27b_smoke(task, tmp_path)

    assert calls == [
        {
            "method_name": (
                "agent4rec_preference_yes_no_litellm_qwen3_8b_traits_smoke"
            ),
            "model": "Qwen/Qwen3-8B",
            "max_candidate_groups": 25,
            "max_workers": 128,
        },
        {
            "method_name": (
                "agent4rec_preference_yes_no_litellm_qwen36_27b_traits_smoke"
            ),
            "model": "Qwen/Qwen3.6-27B",
            "max_candidate_groups": 25,
            "max_workers": 64,
        },
    ]


def test_preference_agent4rec_qwen3_taste_uses_vk_proxy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_method(*args: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(grouped_agent4rec_yes_no, "run_method", fake_run_method)
    monkeypatch.setattr(
        preference_agent4rec_yes_no,
        "_serving_metadata",
        lambda: {"backend": "test"},
    )
    monkeypatch.setattr(
        preference_agent4rec_yes_no,
        "_source_metadata",
        lambda: {"snapshot": "test"},
    )

    preference_agent4rec_yes_no.run_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_smoke(
        SimpleNamespace(manifest={"dataset": "ml-1m"}),
        tmp_path,
    )

    assert captured["taste_client_name"] == "openai_vk_proxy"
    assert captured["taste_model"] == "gpt-4o-mini"


def test_preference_agent4rec_qwen36_taste_summary_uses_matching_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_method(*args: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(grouped_agent4rec_yes_no, "run_method", fake_run_method)
    monkeypatch.setattr(
        preference_agent4rec_yes_no,
        "_serving_metadata",
        lambda: {"backend": "test"},
    )
    monkeypatch.setattr(
        preference_agent4rec_yes_no,
        "_source_metadata",
        lambda: {"snapshot": "test"},
    )

    preference_agent4rec_yes_no.run_qwen36_27b_traits_taste_gpt4o_mini_candidate_summary_smoke(
        SimpleNamespace(manifest={"dataset": "ml-1m"}),
        tmp_path,
    )

    assert captured["method_name"] == (
        "agent4rec_preference_yes_no_litellm_qwen36_27b_traits_taste_"
        "gpt4o_mini_candidate_summary_smoke"
    )
    assert captured["model"] == "Qwen/Qwen3.6-27B"
    assert captured["max_workers"] == 64
    assert captured["taste_client_name"] == "openai_vk_proxy"
    assert captured["summary_usage"] == "candidate"
