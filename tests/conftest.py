"""Explicit fixture metadata for Agent4Rec runner tests; no service calls."""

from types import SimpleNamespace
import sys
import json

import pandas as pd
import pytest


@pytest.fixture
def agent4rec_item_artifact(tmp_path, monkeypatch):
    from runners.in_distribution import agent4rec_items

    original = agent4rec_items.prepare_agent4rec_task

    def prepare(task, **kwargs):
        if task.manifest.get("dataset") != "ml-1m":
            return original(task, **kwargs)
        frames = [
            getattr(task, name)
            for name in ("train", "val", "test", "online_session_history")
            if hasattr(task, name)
        ]
        rows = pd.concat(frames, ignore_index=True)
        if "item_title" not in rows:
            return original(task, **kwargs)
        items = rows[["item_id", "item_title"]].drop_duplicates()
        # Test summaries are synthetic. Feed only title text through the actual
        # preparation API and leave task frames/labels untouched.
        descriptions = {}
        for _, row in rows.iterrows():
            title = row.get("item_title")
            if pd.isna(title):
                continue
            summary = row.get("item_summary")
            descriptions[title] = (
                summary if isinstance(summary, str) else f"Plot of {title}."
            )

        class MetadataCompletions:
            def create(self, **request):
                title = request["messages"][1]["content"].removeprefix("Movie title: ")
                response = {
                    "genres": ["Generated genre"],
                    "summary": descriptions[title],
                }
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=json.dumps(response))
                        )
                    ]
                )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=MetadataCompletions())
        )
        path = tmp_path / "generated_items.jsonl"
        agent4rec_items.prepare_item_profiles(
            items,
            client=client,
            client_name="fixture",
            model="fake-metadata",
            output_path=path,
        )
        return original(task, item_profile_path=path)

    monkeypatch.setattr(agent4rec_items, "prepare_agent4rec_task", prepare)
    for module in tuple(sys.modules.values()):
        if (
            module
            and getattr(module, "__name__", "").startswith("runners.in_distribution.")
            and getattr(module, "prepare_agent4rec_task", None) is original
        ):
            monkeypatch.setattr(module, "prepare_agent4rec_task", prepare)
