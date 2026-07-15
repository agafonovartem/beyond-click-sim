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
    llm_yes_no as grouped_llm_yes_no,
)
from runners.in_distribution.preference_prediction.methods import (  # noqa: E402
    llm_yes_no as preference_llm_yes_no,
)
from runners.in_distribution.preference_prediction.methods import (  # noqa: E402
    METHOD_RUNNERS as PREFERENCE_METHOD_RUNNERS,
)


class _FakeChatCompletions:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.response),
                )
            ]
        )


class _FakeClient:
    def __init__(self, response: str) -> None:
        completions = _FakeChatCompletions(response)
        self.chat = SimpleNamespace(completions=completions)
        self.completions = completions


def test_preference_llm_runner_records_target_and_scores_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = _FakeClient("C1: yes\nC2: no")
    monkeypatch.setattr(grouped_llm_yes_no, "make_llm_client", lambda _: client)
    task = Task(
        name="ml-1m_preference_toy",
        train=pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["h1", "h2"],
                "item_title": ["Toy Story", "Heat"],
                "item_genres": ["Animation", "Crime"],
                "rating": [5, 2],
                "target": [1, 0],
            }
        ),
        val=pd.DataFrame(columns=["user_id", "item_id", "target"]),
        test=pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["i1", "i2"],
                "candidate_group": ["g1", "g1"],
                "item_title": ["Aladdin", "Casino"],
                "item_genres": ["Animation", "Crime"],
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
        },
    )

    result = preference_llm_yes_no.run_method(
        task,
        tmp_path,
        method_name="preference-test",
        max_candidate_groups=None,
        max_workers=1,
    )

    assert result["coverage"]["scored_fraction"] == 1.0
    assert result["test"]["micro"]["f1"] == 1.0
    prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "rate the candidate movie at least 4 out of 5" in prompt
    assert "H1. item_title: Toy Story; item_genres: Animation; rating: 5" in prompt

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"]["class"] == "LLMPreferenceYesNoScorer"
    assert manifest["scorer"]["client_name"] == "litellm_local"
    assert manifest["scorer"]["model"] == "Qwen/Qwen3-8B"
    assert manifest["scorer"]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert manifest["scorer"]["scorer_kwargs"] == {
        "target_description": "The user would rate the candidate movie at least 4 out of 5."
    }
    assert manifest["scorer"]["serving"] == {
        "backend": "litellm_proxy_over_vllm",
        "gpu_memory_utilization": 0.9,
        "litellm_base_url": "http://127.0.0.1:8080/v1",
        "litellm_version": "unknown",
        "max_model_len": 4096,
        "model_revision": "unknown",
        "routing_strategy": "simple-shuffle",
        "tensor_parallel_size_per_replica": 1,
        "thinking_enabled": False,
        "vllm_ports": [8000, 8001, 8002, 8003],
        "vllm_replicas": 4,
        "vllm_version": "unknown",
    }
    assert manifest["source"] == {
        "base_git_commit": None,
        "diff_sha256": None,
        "snapshot_sha256": None,
    }


def test_preference_llm_runner_uses_canonical_summaries_in_both_prompts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = _FakeClient("C1: yes\nC2: no")
    monkeypatch.setattr(grouped_llm_yes_no, "make_llm_client", lambda _: client)
    task = Task(
        name="ml-1m_preference_summary_toy",
        train=pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["h1", "h2"],
                "item_title": ["Toy Story", "Heat"],
                "item_genres": ["Animation", "Crime"],
                "item_summary": ["Toys plan a rescue.", "A crime thriller."],
                "rating": [5, 2],
                "target": [1, 0],
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
                "item_summary": ["A magical lamp adventure.", "A casino drama."],
                "target": [1, 0],
            }
        ),
        schema=TaskSchema(
            target_column="target",
            candidate_group_column="candidate_group",
            feature_columns=("item_title", "item_genres", "item_summary"),
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
                    "source_sha256": "fixture-summary-sha256",
                }
            },
        },
    )

    preference_llm_yes_no.run_method(
        task,
        tmp_path,
        method_name="preference-summary-test",
        max_candidate_groups=None,
        max_workers=1,
        summary_visibility="both",
    )

    prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "summary: Toys plan a rescue." in prompt
    assert "summary: A magical lamp adventure." in prompt
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorer"]["summary_visibility"] == "both"
    assert manifest["scorer"]["item_summaries"] == {
        "uses_item_summaries": True,
        "summary_column": "item_summary",
        "history_item_summaries": True,
        "profile_item_summaries": False,
        "candidate_item_summaries": True,
        "canonical_enrichment": task.manifest["item_enrichment"],
    }


def test_qwen36_27b_methods_are_registered() -> None:
    assert (
        PREFERENCE_METHOD_RUNNERS[
            "llm_preference_yes_no_litellm_qwen36_27b_smoke"
        ]
        is preference_llm_yes_no.run_qwen36_27b_smoke
    )
    assert (
        PREFERENCE_METHOD_RUNNERS[
            "llm_preference_yes_no_litellm_qwen36_27b_full"
        ]
        is preference_llm_yes_no.run_qwen36_27b_full
    )
    assert (
        PREFERENCE_METHOD_RUNNERS[
            "llm_preference_yes_no_litellm_qwen3_8b_summary_full"
        ]
        is preference_llm_yes_no.run_qwen3_8b_summary_full
    )
    assert (
        PREFERENCE_METHOD_RUNNERS[
            "llm_preference_yes_no_litellm_qwen36_27b_summary_full"
        ]
        is preference_llm_yes_no.run_qwen36_27b_summary_full
    )


def test_qwen36_27b_wrapper_uses_planned_model(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run_method(*args, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(preference_llm_yes_no, "run_method", fake_run_method)

    result = preference_llm_yes_no.run_qwen36_27b_full(object(), tmp_path)

    assert result == {"ok": True}
    assert captured == {
        "method_name": "llm_preference_yes_no_litellm_qwen36_27b_full",
        "model": "Qwen/Qwen3.6-27B",
        "max_candidate_groups": None,
        "max_workers": 64,
    }
