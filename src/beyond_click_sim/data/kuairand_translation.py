from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


TRANSLATION_FORMAT_VERSION = 1
TRANSLATION_PROMPT_VERSION = "kuairand-zh-en-v1"
SYSTEM_PROMPT = """You are a professional Chinese-to-English translator for short-video recommendation metadata.
Translate every provided field faithfully into concise, natural English.
Preserve the original meaning, proper nouns, game and media titles, numbers, emoji, hashtags, and @mentions.
Translate meaningful hashtag text when possible, but do not delete identifiers.
Do not summarize, censor, explain, infer missing context, or add information.
An empty input field must remain an empty output field.
Return only the requested JSON object with exactly the requested records and fields."""


class TranslationResponseError(ValueError):
    """Raised when a translation response does not match the requested batch."""


@dataclass(frozen=True)
class TranslationBatch:
    kind: str
    records: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class TranslationResult:
    kind: str
    records: tuple[dict[str, str], ...]
    usage: dict[str, int]
    checkpoint_key: str
    cached: bool


def load_eligible_items(path: Path | str) -> pd.DataFrame:
    """Load KuaiRand items with both textual content and a category path."""

    items = pd.read_parquet(
        Path(path).expanduser().resolve(),
        columns=[
            "item_id",
            "caption",
            "show_cover_text",
            "category_path",
        ],
    )
    for column in ("caption", "show_cover_text", "category_path"):
        items[column] = (
            items[column].fillna("").astype("string").str.strip()
        )
    has_text = items["caption"].ne("") | items["show_cover_text"].ne("")
    has_category = items["category_path"].ne("")
    eligible = items.loc[has_text & has_category].copy()
    eligible = eligible.sort_values("item_id", kind="mergesort").reset_index(
        drop=True
    )
    if eligible["item_id"].duplicated().any():
        raise ValueError("KuaiRand item_id must be unique")
    return eligible


def category_nodes(items: pd.DataFrame) -> list[str]:
    """Return the sorted unique non-empty nodes in category paths."""

    nodes = {
        node.strip()
        for path in items["category_path"]
        for node in str(path).split(">")
        if node.strip()
    }
    return sorted(nodes)


def item_batches(
    items: pd.DataFrame,
    *,
    batch_size: int,
) -> list[TranslationBatch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    records = [
        {
            "id": str(row.item_id),
            "caption": str(row.caption),
            "cover_text": str(row.show_cover_text),
        }
        for row in items.itertuples(index=False)
    ]
    return [
        TranslationBatch("items", tuple(records[start : start + batch_size]))
        for start in range(0, len(records), batch_size)
    ]


def category_batches(
    nodes: Sequence[str],
    *,
    batch_size: int,
) -> list[TranslationBatch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    records = [
        {"id": f"category:{position}", "text": node}
        for position, node in enumerate(nodes)
    ]
    return [
        TranslationBatch(
            "categories",
            tuple(records[start : start + batch_size]),
        )
        for start in range(0, len(records), batch_size)
    ]


def translate_batches(
    *,
    client: Any,
    batches: Sequence[TranslationBatch],
    checkpoint_dir: Path | str,
    model: str,
    workers: int,
    max_attempts: int,
    max_tokens: int,
) -> list[TranslationResult]:
    """Translate batches concurrently with content-addressed checkpoints."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
    results: list[TranslationResult | None] = [None] * len(batches)

    def run(position: int, batch: TranslationBatch) -> tuple[int, TranslationResult]:
        result = translate_batch(
            client=client,
            batch=batch,
            checkpoint_dir=checkpoint_root,
            model=model,
            max_attempts=max_attempts,
            max_tokens=max_tokens,
        )
        return position, result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run, position, batch): position
            for position, batch in enumerate(batches)
        }
        completed = 0
        for future in as_completed(futures):
            position, result = future.result()
            results[position] = result
            completed += 1
            print(
                json.dumps(
                    {
                        "event": "batch_completed",
                        "kind": result.kind,
                        "completed": completed,
                        "total": len(batches),
                        "cached": result.cached,
                        "records": len(result.records),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if any(result is None for result in results):
        raise RuntimeError("A translation batch completed without a result")
    return [result for result in results if result is not None]


def translate_batch(
    *,
    client: Any,
    batch: TranslationBatch,
    checkpoint_dir: Path,
    model: str,
    max_attempts: int,
    max_tokens: int,
) -> TranslationResult:
    messages = build_messages(batch)
    request_config = {
        "model": model,
        "temperature": "provider_default",
        "max_completion_tokens": max_tokens,
    }
    checkpoint_key = translation_checkpoint_key(
        batch=batch,
        messages=messages,
        request_config=request_config,
    )
    checkpoint_path = (
        checkpoint_dir
        / batch.kind
        / checkpoint_key[:2]
        / f"{checkpoint_key}.json"
    )
    cached = load_translation_checkpoint(
        checkpoint_path,
        expected_key=checkpoint_key,
        expected_batch=batch,
    )
    if cached is not None:
        return TranslationResult(
            kind=batch.kind,
            records=tuple(cached["translations"]),
            usage=_normalize_usage(cached.get("usage")),
            checkpoint_key=checkpoint_key,
            cached=True,
        )

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if not isinstance(content, str):
                raise TranslationResponseError(
                    "Translation response content is not text"
                )
            translations = parse_translation_response(
                content,
                batch=batch,
            )
            usage = response_usage(getattr(response, "usage", None))
            payload = {
                "format_version": TRANSLATION_FORMAT_VERSION,
                "prompt_version": TRANSLATION_PROMPT_VERSION,
                "checkpoint_key": checkpoint_key,
                "kind": batch.kind,
                "model": model,
                "source_records": list(batch.records),
                "translations": translations,
                "usage": usage,
                "created_at_utc": datetime.now(UTC).isoformat(),
            }
            atomic_write_json(checkpoint_path, payload)
            return TranslationResult(
                kind=batch.kind,
                records=tuple(translations),
                usage=usage,
                checkpoint_key=checkpoint_key,
                cached=False,
            )
        except Exception as error:
            last_error = error
            if attempt == max_attempts:
                break
            time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(
        f"Translation failed after {max_attempts} attempts for "
        f"{batch.kind} batch {checkpoint_key}"
    ) from last_error


def build_messages(batch: TranslationBatch) -> list[dict[str, str]]:
    if batch.kind == "items":
        instruction = (
            "Translate each caption and cover_text. Return "
            '{"translations":[{"id":"...","caption_en":"...",'
            '"cover_text_en":"..."}]}.'
        )
    elif batch.kind == "categories":
        instruction = (
            "Translate each category label. Use concise recommender-system "
            "taxonomy terminology. Return "
            '{"translations":[{"id":"...","text_en":"..."}]}.'
        )
    else:
        raise ValueError(f"Unsupported translation batch kind: {batch.kind}")
    user_payload = {
        "instruction": instruction,
        "records": list(batch.records),
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False),
        },
    ]


def parse_translation_response(
    text: str,
    *,
    batch: TranslationBatch,
) -> list[dict[str, str]]:
    """Parse and validate one exact-ID translation response."""

    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, TranslationResponseError) as error:
        raise TranslationResponseError(
            f"Invalid translation JSON: {error}"
        ) from error
    if not isinstance(payload, dict) or set(payload) != {"translations"}:
        raise TranslationResponseError(
            "Response must contain only the translations field"
        )
    translations = payload["translations"]
    if not isinstance(translations, list):
        raise TranslationResponseError("translations must be a list")

    expected_ids = [record["id"] for record in batch.records]
    expected_fields = (
        {"id", "caption_en", "cover_text_en"}
        if batch.kind == "items"
        else {"id", "text_en"}
    )
    by_id: dict[str, dict[str, str]] = {}
    for raw_record in translations:
        if not isinstance(raw_record, dict) or set(raw_record) != expected_fields:
            raise TranslationResponseError(
                f"Translation record must have fields {sorted(expected_fields)}"
            )
        if not all(isinstance(value, str) for value in raw_record.values()):
            raise TranslationResponseError(
                "Every translation record value must be a string"
            )
        record_id = raw_record["id"]
        if record_id in by_id:
            raise TranslationResponseError(
                f"Duplicate translation id: {record_id}"
            )
        by_id[record_id] = {
            key: value.strip() for key, value in raw_record.items()
        }
    if set(by_id) != set(expected_ids):
        raise TranslationResponseError(
            "Translation IDs do not exactly match the requested IDs"
        )

    source_by_id = {record["id"]: record for record in batch.records}
    ordered = [by_id[record_id] for record_id in expected_ids]
    for translated in ordered:
        source = source_by_id[translated["id"]]
        if batch.kind == "items":
            _require_empty_parity(
                source["caption"],
                translated["caption_en"],
                field="caption",
            )
            _require_empty_parity(
                source["cover_text"],
                translated["cover_text_en"],
                field="cover_text",
            )
        else:
            _require_empty_parity(
                source["text"],
                translated["text_en"],
                field="category",
            )
    return ordered


def assemble_item_translations(
    *,
    items: pd.DataFrame,
    item_results: Sequence[TranslationResult],
    nodes: Sequence[str],
    category_results: Sequence[TranslationResult],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    item_records = [
        record for result in item_results for record in result.records
    ]
    item_translation = pd.DataFrame(item_records)
    if item_translation["id"].duplicated().any():
        raise ValueError("Duplicate translated item ID")
    item_translation = item_translation.rename(columns={"id": "item_id"})

    category_records = [
        record for result in category_results for record in result.records
    ]
    category_translation = pd.DataFrame(category_records)
    expected_category_ids = [
        f"category:{position}" for position in range(len(nodes))
    ]
    if category_translation["id"].tolist() != expected_category_ids:
        raise ValueError("Translated category order does not match source nodes")
    category_translation.insert(1, "text_zh", list(nodes))
    node_map = dict(
        zip(
            category_translation["text_zh"],
            category_translation["text_en"],
            strict=True,
        )
    )

    output = items.rename(
        columns={
            "caption": "caption_zh",
            "show_cover_text": "cover_text_zh",
            "category_path": "category_path_zh",
        }
    ).merge(
        item_translation,
        on="item_id",
        how="left",
        validate="one_to_one",
    )
    if output[["caption_en", "cover_text_en"]].isna().any().any():
        raise ValueError("Missing translated item text")
    output["category_path_en"] = output["category_path_zh"].map(
        lambda path: " > ".join(
            node_map[node.strip()]
            for node in str(path).split(">")
            if node.strip()
        )
    )
    return output, category_translation


def translation_audit(
    *,
    output: pd.DataFrame,
    category_glossary: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "items": int(len(output)),
        "categories": int(len(category_glossary)),
        "source_nonempty_caption": int(output["caption_zh"].ne("").sum()),
        "translated_nonempty_caption": int(output["caption_en"].ne("").sum()),
        "source_nonempty_cover_text": int(
            output["cover_text_zh"].ne("").sum()
        ),
        "translated_nonempty_cover_text": int(
            output["cover_text_en"].ne("").sum()
        ),
        "translated_nonempty_category_path": int(
            output["category_path_en"].ne("").sum()
        ),
        "english_caption_with_cjk": int(
            output["caption_en"].map(contains_cjk).sum()
        ),
        "english_cover_text_with_cjk": int(
            output["cover_text_en"].map(contains_cjk).sum()
        ),
        "english_category_with_cjk": int(
            category_glossary["text_en"].map(contains_cjk).sum()
        ),
        "caption_length_ratio": _length_ratio_summary(
            output["caption_zh"],
            output["caption_en"],
        ),
        "cover_text_length_ratio": _length_ratio_summary(
            output["cover_text_zh"],
            output["cover_text_en"],
        ),
    }


def total_usage(results: Iterable[TranslationResult]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for result in results:
        for field in totals:
            totals[field] += int(result.usage.get(field, 0))
    return totals


def response_usage(usage: Any) -> dict[str, int]:
    if usage is None:
        return _normalize_usage(None)
    return _normalize_usage(
        {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
    )


def translation_checkpoint_key(
    *,
    batch: TranslationBatch,
    messages: Sequence[Mapping[str, str]],
    request_config: Mapping[str, Any],
) -> str:
    payload = {
        "format_version": TRANSLATION_FORMAT_VERSION,
        "prompt_version": TRANSLATION_PROMPT_VERSION,
        "batch": {"kind": batch.kind, "records": list(batch.records)},
        "messages": list(messages),
        "request_config": dict(request_config),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_translation_checkpoint(
    path: Path,
    *,
    expected_key: str,
    expected_batch: TranslationBatch,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TranslationResponseError(
            f"Cannot parse translation checkpoint {path}"
        ) from error
    if payload.get("format_version") != TRANSLATION_FORMAT_VERSION:
        raise TranslationResponseError(
            f"Unsupported translation checkpoint version in {path}"
        )
    if payload.get("checkpoint_key") != expected_key:
        raise TranslationResponseError(
            f"Translation checkpoint key mismatch in {path}"
        )
    if payload.get("kind") != expected_batch.kind:
        raise TranslationResponseError(
            f"Translation checkpoint kind mismatch in {path}"
        )
    translations = payload.get("translations")
    parsed = parse_translation_response(
        json.dumps({"translations": translations}, ensure_ascii=False),
        batch=expected_batch,
    )
    payload["translations"] = parsed
    return payload


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_csv(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contains_cjk(text: Any) -> bool:
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", str(text)))


def _length_ratio_summary(
    source: pd.Series,
    translated: pd.Series,
) -> dict[str, float]:
    mask = source.ne("")
    ratios = (
        translated.loc[mask].str.len()
        / source.loc[mask].str.len().clip(lower=1)
    )
    quantiles = ratios.quantile([0.0, 0.01, 0.5, 0.99, 1.0])
    return {
        "min": float(quantiles.loc[0.0]),
        "p01": float(quantiles.loc[0.01]),
        "median": float(quantiles.loc[0.5]),
        "p99": float(quantiles.loc[0.99]),
        "max": float(quantiles.loc[1.0]),
    }


def _require_empty_parity(
    source: str,
    translated: str,
    *,
    field: str,
) -> None:
    if bool(source.strip()) != bool(translated.strip()):
        raise TranslationResponseError(
            f"Empty/non-empty mismatch for {field}"
        )


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TranslationResponseError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _normalize_usage(usage: Mapping[str, Any] | None) -> dict[str, int]:
    values = usage or {}
    return {
        field: int(values.get(field, 0))
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
