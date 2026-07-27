from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from beyond_click_sim.data.kuairand_translation import (
    TranslationBatch,
    TranslationResponseError,
    assemble_item_translations,
    category_batches,
    category_nodes,
    item_batches,
    parse_translation_response,
    translate_batch,
)


def test_item_response_requires_exact_ids_fields_and_empty_parity() -> None:
    batch = TranslationBatch(
        "items",
        (
            {"id": "1", "caption": "篮球", "cover_text": ""},
            {"id": "2", "caption": "", "cover_text": "教程"},
        ),
    )
    response = json.dumps(
        {
            "translations": [
                {
                    "id": "1",
                    "caption_en": "Basketball",
                    "cover_text_en": "",
                },
                {
                    "id": "2",
                    "caption_en": "",
                    "cover_text_en": "Tutorial",
                },
            ]
        }
    )

    parsed = parse_translation_response(response, batch=batch)

    assert [record["id"] for record in parsed] == ["1", "2"]
    assert parsed[0]["caption_en"] == "Basketball"


@pytest.mark.parametrize(
    "translations",
    [
        [
            {"id": "1", "caption_en": "Basketball", "cover_text_en": ""},
        ],
        [
            {"id": "1", "caption_en": "", "cover_text_en": ""},
            {"id": "2", "caption_en": "", "cover_text_en": "Tutorial"},
        ],
        [
            {
                "id": "1",
                "caption_en": "Basketball",
                "cover_text_en": "",
                "extra": "bad",
            },
            {"id": "2", "caption_en": "", "cover_text_en": "Tutorial"},
        ],
    ],
)
def test_item_response_rejects_incomplete_or_changed_shape(
    translations: list[dict[str, str]],
) -> None:
    batch = TranslationBatch(
        "items",
        (
            {"id": "1", "caption": "篮球", "cover_text": ""},
            {"id": "2", "caption": "", "cover_text": "教程"},
        ),
    )

    with pytest.raises(TranslationResponseError):
        parse_translation_response(
            json.dumps({"translations": translations}),
            batch=batch,
        )


def test_batches_and_assembly_preserve_source_order_and_category_tree() -> None:
    items = pd.DataFrame(
        {
            "item_id": ["1", "2"],
            "caption": ["篮球视频", "游戏视频"],
            "show_cover_text": ["", "教程"],
            "category_path": ["运动 > 篮球", "游戏 > MOBA"],
        }
    )
    nodes = category_nodes(items)
    assert nodes == ["MOBA", "游戏", "篮球", "运动"]
    assert len(item_batches(items, batch_size=1)) == 2
    assert len(category_batches(nodes, batch_size=3)) == 2

    from beyond_click_sim.data.kuairand_translation import TranslationResult

    item_result = TranslationResult(
        kind="items",
        records=(
            {
                "id": "1",
                "caption_en": "Basketball video",
                "cover_text_en": "",
            },
            {
                "id": "2",
                "caption_en": "Gaming video",
                "cover_text_en": "Tutorial",
            },
        ),
        usage={},
        checkpoint_key="a" * 64,
        cached=False,
    )
    category_result = TranslationResult(
        kind="categories",
        records=(
            {"id": "category:0", "text_en": "MOBA"},
            {"id": "category:1", "text_en": "Gaming"},
            {"id": "category:2", "text_en": "Basketball"},
            {"id": "category:3", "text_en": "Sports"},
        ),
        usage={},
        checkpoint_key="b" * 64,
        cached=False,
    )

    output, glossary = assemble_item_translations(
        items=items,
        item_results=[item_result],
        nodes=nodes,
        category_results=[category_result],
    )

    assert output["category_path_en"].tolist() == [
        "Sports > Basketball",
        "Gaming > MOBA",
    ]
    assert glossary["text_zh"].tolist() == nodes


def test_translation_checkpoint_prevents_a_second_api_call(tmp_path) -> None:
    batch = TranslationBatch(
        "items",
        (
            {"id": "1", "caption": "篮球", "cover_text": ""},
        ),
    )

    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_: object) -> SimpleNamespace:
            self.calls += 1
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "translations": [
                                        {
                                            "id": "1",
                                            "caption_en": "Basketball",
                                            "cover_text_en": "",
                                        }
                                    ]
                                }
                            )
                        )
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=4,
                    total_tokens=14,
                ),
            )

    completions = FakeCompletions()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    first = translate_batch(
        client=client,
        batch=batch,
        checkpoint_dir=tmp_path,
        model="gpt-test",
        max_attempts=1,
        max_tokens=100,
    )
    replay = translate_batch(
        client=client,
        batch=batch,
        checkpoint_dir=tmp_path,
        model="gpt-test",
        max_attempts=1,
        max_tokens=100,
    )

    assert not first.cached
    assert replay.cached
    assert replay.records == first.records
    assert replay.usage == first.usage
    assert completions.calls == 1
