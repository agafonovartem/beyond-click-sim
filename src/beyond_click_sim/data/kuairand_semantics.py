from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from beyond_click_sim.data.canonical import sha256_file


REQUIRED_TRANSLATION_COLUMNS = (
    "item_id",
    "caption_zh",
    "cover_text_zh",
    "category_path_zh",
    "caption_en",
    "cover_text_en",
    "category_path_en",
)
PRIMARY_TRANSLATION_MODEL = "gpt-5.5"
PRIMARY_TRANSLATED_ITEMS_SHA256 = (
    "4fca7b9153c73e0b5206357b95e0ebee2759a1647f2d366a7fcf8078f492a1eb"
)


@dataclass(frozen=True)
class KuaiRandEnglishSemantics:
    """Verified English item metadata generated from the official supplement."""

    items: pd.DataFrame
    manifest: dict[str, Any]
    root: Path
    items_path: Path
    manifest_path: Path
    items_sha256: str
    manifest_sha256: str


def load_kuairand_english_semantics(
    translation_dir: Path | str,
    *,
    canonical_items_path: Path | str,
) -> KuaiRandEnglishSemantics:
    """Load and verify a complete KuaiRand English-translation artifact.

    The generated translation is deliberately kept outside the canonical
    dataset. This loader binds it to the exact canonical ``items.parquet`` by
    content hash and verifies the Chinese source fields before exposing the
    English columns to task builders.
    """

    root = Path(translation_dir).expanduser().resolve()
    canonical_path = Path(canonical_items_path).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not canonical_path.exists():
        raise FileNotFoundError(canonical_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "kuairand-pure":
        raise ValueError("Translation manifest is not for KuaiRand-Pure")
    if manifest.get("model") != PRIMARY_TRANSLATION_MODEL:
        raise ValueError(
            "Primary KuaiRand protocol requires translation model "
            f"{PRIMARY_TRANSLATION_MODEL!r}"
        )
    selection = manifest.get("selection", {})
    if selection.get("partial") is not False:
        raise ValueError("A partial translation artifact cannot build the task")

    outputs = manifest.get("outputs", {})
    output_name = outputs.get("items")
    expected_output_sha256 = outputs.get("items_sha256")
    if not output_name or not expected_output_sha256:
        raise ValueError("Translation manifest lacks the translated-items output")
    if expected_output_sha256 != PRIMARY_TRANSLATED_ITEMS_SHA256:
        raise ValueError(
            "Translation manifest does not reference the item CSV pinned by "
            "the primary KuaiRand protocol"
        )
    items_path = root / str(output_name)
    if not items_path.exists():
        raise FileNotFoundError(items_path)
    observed_output_sha256 = sha256_file(items_path)
    if observed_output_sha256 != expected_output_sha256:
        raise ValueError("Translated-items SHA256 does not match its manifest")

    expected_source_sha256 = manifest.get("source", {}).get(
        "items_parquet_sha256"
    )
    observed_source_sha256 = sha256_file(canonical_path)
    if not expected_source_sha256:
        raise ValueError("Translation manifest lacks source items SHA256")
    if observed_source_sha256 != expected_source_sha256:
        raise ValueError(
            "Translation artifact was generated from a different canonical "
            "items.parquet"
        )

    translated = pd.read_csv(
        items_path,
        dtype={"item_id": "string"},
        keep_default_na=False,
    )
    _require_columns(translated, REQUIRED_TRANSLATION_COLUMNS)
    translated = translated.loc[:, REQUIRED_TRANSLATION_COLUMNS].copy()
    translated["item_id"] = translated["item_id"].astype("string").str.strip()
    for column in REQUIRED_TRANSLATION_COLUMNS[1:]:
        translated[column] = _normalized_text(translated[column])
    _require_unique_nonempty_ids(translated["item_id"])

    expected_rows = selection.get("items")
    if expected_rows is None or int(expected_rows) != len(translated):
        raise ValueError("Translated item count does not match the manifest")
    source_semantic = (
        translated["caption_zh"].ne("")
        | translated["cover_text_zh"].ne("")
    ) & translated["category_path_zh"].ne("")
    if not source_semantic.all():
        raise ValueError("Translation contains source-ineligible semantic rows")
    if translated["caption_zh"].ne("").ne(translated["caption_en"].ne("")).any():
        raise ValueError("Caption source/translation empty-field parity failed")
    if (
        translated["cover_text_zh"]
        .ne("")
        .ne(translated["cover_text_en"].ne(""))
        .any()
    ):
        raise ValueError("Cover-text source/translation empty-field parity failed")
    if translated["category_path_en"].eq("").any():
        raise ValueError("Translated category paths must be non-empty")

    canonical = pd.read_parquet(
        canonical_path,
        columns=["item_id", "caption", "show_cover_text", "category_path"],
    )
    canonical["item_id"] = canonical["item_id"].astype("string").str.strip()
    for column in ("caption", "show_cover_text", "category_path"):
        canonical[column] = _normalized_text(canonical[column])
    _require_unique_nonempty_ids(canonical["item_id"])
    canonical = canonical.loc[
        (canonical["caption"].ne("") | canonical["show_cover_text"].ne(""))
        & canonical["category_path"].ne(""),
        ["item_id", "caption", "show_cover_text", "category_path"],
    ].sort_values("item_id", kind="mergesort").reset_index(drop=True)
    translated = translated.sort_values(
        "item_id", kind="mergesort"
    ).reset_index(drop=True)
    if not canonical["item_id"].equals(translated["item_id"]):
        raise ValueError(
            "Translated item IDs do not exactly match canonical semantic items"
        )
    source_pairs = (
        ("caption", "caption_zh"),
        ("show_cover_text", "cover_text_zh"),
        ("category_path", "category_path_zh"),
    )
    for canonical_column, translation_column in source_pairs:
        if not canonical[canonical_column].equals(
            translated[translation_column]
        ):
            raise ValueError(
                f"Translation source field {translation_column!r} does not "
                f"match canonical {canonical_column!r}"
            )

    task_items = translated[
        [
            "item_id",
            "caption_en",
            "cover_text_en",
            "category_path_en",
        ]
    ].rename(
        columns={
            "caption_en": "item_caption_en",
            "cover_text_en": "item_cover_text_en",
            "category_path_en": "item_category_path_en",
        }
    )
    return KuaiRandEnglishSemantics(
        items=task_items,
        manifest=manifest,
        root=root,
        items_path=items_path,
        manifest_path=manifest_path,
        items_sha256=observed_output_sha256,
        manifest_sha256=sha256_file(manifest_path),
    )


def _normalized_text(values: pd.Series) -> pd.Series:
    return (
        values.fillna("")
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Translation artifact is missing columns: {missing}")


def _require_unique_nonempty_ids(values: pd.Series) -> None:
    if values.eq("").any():
        raise ValueError("Translation artifact contains an empty item_id")
    if values.duplicated().any():
        raise ValueError("Translation artifact contains duplicate item_id values")
