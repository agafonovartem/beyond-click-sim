from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import beyond_click_sim.data.kuairand_semantics as semantics_module
from beyond_click_sim.data.canonical import sha256_file
from beyond_click_sim.data.kuairand_semantics import (
    load_kuairand_english_semantics,
)


def test_loads_verified_translation_and_preserves_string_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_path, translation_dir = _translation_fixture(tmp_path)
    _pin_fixture_output(monkeypatch, translation_dir)

    artifact = load_kuairand_english_semantics(
        translation_dir,
        canonical_items_path=canonical_path,
    )

    assert artifact.items["item_id"].tolist() == ["001", "2"]
    assert list(artifact.items.columns) == [
        "item_id",
        "item_caption_en",
        "item_cover_text_en",
        "item_category_path_en",
    ]
    assert artifact.items.loc[0, "item_caption_en"] == "English caption"
    assert artifact.items.loc[1, "item_cover_text_en"] == "English cover"


def test_rejects_modified_output_after_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_path, translation_dir = _translation_fixture(tmp_path)
    _pin_fixture_output(monkeypatch, translation_dir)
    output = translation_dir / "kuairand_items_en.csv"
    frame = pd.read_csv(output, dtype={"item_id": "string"}, keep_default_na=False)
    frame.loc[0, "caption_en"] = "Modified"
    frame.to_csv(output, index=False)

    with pytest.raises(ValueError, match="SHA256"):
        load_kuairand_english_semantics(
            translation_dir,
            canonical_items_path=canonical_path,
        )


def test_rejects_translation_bound_to_other_canonical_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_path, translation_dir = _translation_fixture(tmp_path)
    _pin_fixture_output(monkeypatch, translation_dir)
    canonical = pd.read_parquet(canonical_path)
    canonical.loc[0, "caption"] = "Different Chinese source"
    canonical.to_parquet(canonical_path, index=False)

    with pytest.raises(ValueError, match="different canonical"):
        load_kuairand_english_semantics(
            translation_dir,
            canonical_items_path=canonical_path,
        )


def test_rejects_partial_or_source_mismatched_translation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_path, translation_dir = _translation_fixture(tmp_path)
    _pin_fixture_output(monkeypatch, translation_dir)
    manifest_path = translation_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selection"]["partial"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="partial"):
        load_kuairand_english_semantics(
            translation_dir,
            canonical_items_path=canonical_path,
        )

    canonical_path, translation_dir = _translation_fixture(tmp_path / "source")
    output = translation_dir / "kuairand_items_en.csv"
    frame = pd.read_csv(output, dtype={"item_id": "string"}, keep_default_na=False)
    frame.loc[0, "caption_zh"] = "Wrong source text"
    frame.to_csv(output, index=False)
    manifest_path = translation_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["items_sha256"] = sha256_file(output)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _pin_fixture_output(monkeypatch, translation_dir)
    with pytest.raises(ValueError, match="does not match canonical"):
        load_kuairand_english_semantics(
            translation_dir,
            canonical_items_path=canonical_path,
        )


def test_rejects_unpinned_model_or_internally_consistent_item_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_path, translation_dir = _translation_fixture(tmp_path)
    output = translation_dir / "kuairand_items_en.csv"
    manifest_path = translation_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model"] = "another-model"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _pin_fixture_output(monkeypatch, translation_dir)
    with pytest.raises(ValueError, match="requires translation model"):
        load_kuairand_english_semantics(
            translation_dir,
            canonical_items_path=canonical_path,
        )

    manifest["model"] = "gpt-5.5"
    manifest["outputs"]["items_sha256"] = sha256_file(output)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        semantics_module,
        "PRIMARY_TRANSLATED_ITEMS_SHA256",
        "f" * 64,
    )
    with pytest.raises(ValueError, match="pinned"):
        load_kuairand_english_semantics(
            translation_dir,
            canonical_items_path=canonical_path,
        )


def _pin_fixture_output(
    monkeypatch: pytest.MonkeyPatch,
    translation_dir: Path,
) -> None:
    monkeypatch.setattr(
        semantics_module,
        "PRIMARY_TRANSLATED_ITEMS_SHA256",
        sha256_file(translation_dir / "kuairand_items_en.csv"),
    )


def _translation_fixture(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    canonical_path = tmp_path / "items.parquet"
    canonical = pd.DataFrame(
        {
            "item_id": pd.Series(["001", "2", "3"], dtype="string"),
            "caption": ["中文标题", "", ""],
            "show_cover_text": ["", "中文封面", ""],
            "category_path": ["音乐 > 流行", "生活", ""],
        }
    )
    canonical.to_parquet(canonical_path, index=False)

    translation_dir = tmp_path / "translation"
    translation_dir.mkdir()
    output = translation_dir / "kuairand_items_en.csv"
    pd.DataFrame(
        {
            "item_id": ["001", "2"],
            "caption_zh": ["中文标题", ""],
            "cover_text_zh": ["", "中文封面"],
            "category_path_zh": ["音乐 > 流行", "生活"],
            "caption_en": ["English caption", ""],
            "cover_text_en": ["", "English cover"],
            "category_path_en": ["Music > Pop", "Lifestyle"],
        }
    ).to_csv(output, index=False)
    manifest = {
        "dataset": "kuairand-pure",
        "model": "gpt-5.5",
        "selection": {"items": 2, "partial": False},
        "source": {"items_parquet_sha256": sha256_file(canonical_path)},
        "outputs": {
            "items": output.name,
            "items_sha256": sha256_file(output),
        },
    }
    (translation_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return canonical_path, translation_dir
