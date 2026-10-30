"""Agent4Rec-only item preparation; never changes shared materialized tasks.

Prepare title-derived metadata explicitly, before scoring. Scoring only reads the
artifact and refuses missing/mismatched entries; it never starts paid generation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from beyond_click_sim.scorers.agent4rec.profiles import (
    Agent4RecProfileGenerator,
    client_identity,
)
from beyond_click_sim.tasks.item_statistics import TrainItemRatingStatistics

ITEM_PROFILE_VERSION = "agent4rec_title_metadata_v1"
GENERATED_GENRES_COLUMN = "item_agent4rec_generated_genres"
MOVIE_CANDIDATE_COLUMNS = (
    "item_title",
    "item_rating_mean",
    "item_rating_count",
    GENERATED_GENRES_COLUMN,
)
MOVIE_COLUMN_LABELS = {
    "item_title": "movie title",
    "item_rating_mean": "quality (training mean rating)",
    "item_rating_count": "popularity (training rating count)",
    GENERATED_GENRES_COLUMN: "LLM-generated genres",
    "item_genres": "canonical genres",
}
DEFAULT_ITEM_PROFILE_PATH = (
    Path(__file__).resolve().parents[2]
    / "outputs/agent4rec_item_profiles/ml-1m_v1.jsonl"
)


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _messages(title):
    return [
        {
            "role": "system",
            "content": "Describe a movie using only its title and your movie knowledge. Do not infer user feedback. Return JSON with genres (a nonempty list of genre names) and summary (a short nonempty plot summary).",
        },
        {"role": "user", "content": f"Movie title: {title}"},
    ]


def _load(path):
    if not path.exists():
        raise ValueError(
            f"Missing Agent4Rec generated item profiles: {path}. Prepare title-only metadata first; see python -m runners.in_distribution.agent4rec_items --help."
        )
    records = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("version") != ITEM_PROFILE_VERSION:
            continue
        payload = {k: v for k, v in row.items() if k != "content_sha256"}
        if row.get("content_sha256") != _hash(payload):
            raise ValueError("Agent4Rec item metadata content hash mismatch")
        if (
            not row.get("genres")
            or not isinstance(row["genres"], list)
            or not all(isinstance(g, str) and g.strip() for g in row["genres"])
        ):
            raise ValueError("Agent4Rec item metadata requires generated genres")
        if not isinstance(row.get("summary"), str) or not row["summary"].strip():
            raise ValueError("Agent4Rec item metadata requires a summary")
        generation = row.get("generation", {})
        if not generation.get("model") or generation.get("messages") != _messages(
            row["title"]
        ):
            raise ValueError(
                "Agent4Rec item metadata must have title-only generation provenance"
            )
        records[(str(row["item_id"]), row["title"])] = row
    return records


def prepare_item_profiles(
    items,
    *,
    client,
    client_name,
    model,
    output_path,
    temperature=0.0,
    max_tokens=512,
    max_attempts=5,
):
    """Generate/cache genres and summaries from titles, ignoring all other inputs."""
    if not model or not client_name or max_attempts < 1 or max_tokens < 1:
        raise ValueError(
            "Model/client identity and positive generation limits are required"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    title_column = "item_title" if "item_title" in items else "title"
    selected = items[["item_id", title_column]].drop_duplicates()
    if selected.item_id.duplicated().any() or selected.isna().any().any():
        raise ValueError("Item ids must have one nonmissing title")
    with Agent4RecProfileGenerator._taste_cache_lock(output_path):
        cached = _load(output_path) if output_path.exists() else {}
        for item, title in selected.itertuples(index=False, name=None):
            title = str(title).strip()
            if not title:
                raise ValueError("Empty movie title")
            generation = {
                "client_name": client_name,
                "model": model,
                "client_identity": client_identity(client),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": _messages(title),
            }
            key = _hash(
                {
                    "version": ITEM_PROFILE_VERSION,
                    "item_id": str(item),
                    "generation": generation,
                }
            )
            previous = cached.get((str(item), title))
            if previous and previous["cache_key"] == key:
                continue
            last_error = None
            for _ in range(max_attempts):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=generation["messages"],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    parsed = json.loads(response.choices[0].message.content)
                    genres, summary = parsed["genres"], parsed["summary"]
                    if (
                        not isinstance(genres, list)
                        or not genres
                        or not all(isinstance(g, str) and g.strip() for g in genres)
                    ):
                        raise ValueError("Expected a nonempty genre list")
                    if not isinstance(summary, str) or not summary.strip():
                        raise ValueError("Expected a nonempty summary")
                    break
                except Exception as error:
                    last_error = error
            else:
                raise RuntimeError(
                    f"Item metadata generation failed for {item}"
                ) from last_error
            row = {
                "version": ITEM_PROFILE_VERSION,
                "item_id": str(item),
                "title": title,
                "genres": list(dict.fromkeys(g.strip() for g in genres)),
                "summary": summary.strip(),
                "generation": generation,
                "cache_key": key,
            }
            row["content_sha256"] = _hash(row)
            with output_path.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            cached[(str(item), title)] = row
    return output_path


def prepare_agent4rec_task(task, *, item_profile_path=None):
    """Attach frozen descriptions and full-training statistics to a private copy.

    Cold-start candidate statistics use warm training users; profile traits still
    use only the permitted k observations. Coat deliberately omits quality and
    popularity signals from self-selected observations, as well as movie fields.
    """
    manifest = deepcopy(task.manifest)
    dataset = manifest.get("dataset")
    if dataset != "ml-1m":
        manifest["agent4rec_item_profiles"] = {
            "domain": dataset,
            "omissions": (
                {
                    "quality": "self_selected_item_quality_excluded_by_rerank_condition",
                    "popularity": "self_selected_frequency_excluded_by_rerank_condition",
                    "movie_genres_and_plot": "clothing_attributes_substituted",
                }
                if dataset == "coat"
                else {
                    "quality": "playtime_is_not_rating",
                    "movie_genres_and_plot": "game_genres_and_tags_substituted",
                }
            ),
        }
        if dataset == "steam":
            train = task.train
            sampled = task.schema.sampled_column
            if sampled and sampled in train:
                train = train.loc[~train[sampled].fillna(False).astype(bool)]
            counts = train.groupby("item_id").size()
            frames = {
                name: getattr(task, name).assign(
                    item_interaction_count=getattr(task, name)
                    .item_id.map(counts)
                    .fillna(0)
                    .astype(int)
                )
                for name in ("train", "val", "test")
            }
            manifest["agent4rec_item_profiles"]["popularity"] = (
                "observed_training_interaction_count"
            )
            return replace(task, **frames, manifest=manifest)
        return replace(task, manifest=manifest)
    path = Path(
        item_profile_path
        or os.environ.get("AGENT4REC_ITEM_PROFILE_PATH", DEFAULT_ITEM_PROFILE_PATH)
    )
    records = _load(path)
    frames = {name: getattr(task, name) for name in ("train", "val", "test")}
    if hasattr(task, "online_session_history"):
        frames["online_session_history"] = task.online_session_history
    train = task.train
    sampled = task.schema.sampled_column
    if sampled and sampled in train:
        train = train.loc[~train[sampled].fillna(False).astype(bool)]
    item_ids = pd.concat([f[["item_id"]] for f in frames.values()]).drop_duplicates()
    stats, stat_manifest = TrainItemRatingStatistics().enrich_items(
        items=item_ids, train_interactions=train, item_column="item_id"
    )
    stats = stats.set_index("item_id")
    enriched = {}
    used = {}
    for name, frame in frames.items():
        copied = frame.copy()
        metadata = []
        for item, title in frame.reindex(columns=["item_id", "item_title"]).itertuples(
            index=False, name=None
        ):
            key = (str(item), str(title).strip())
            if key not in records:
                raise ValueError(
                    f"Missing Agent4Rec generated metadata for item/title {key!r}; prepare artifacts before scoring"
                )
            metadata.append(records[key])
            used[key] = records[key]
        copied[GENERATED_GENRES_COLUMN] = ["|".join(r["genres"]) for r in metadata]
        copied["item_summary"] = [r["summary"] for r in metadata]
        copied["item_rating_mean"] = frame.item_id.map(stats.rating_mean)
        copied["item_rating_count"] = frame.item_id.map(stats.rating_count)
        enriched[name] = copied
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["agent4rec_item_profiles"] = {
        "version": ITEM_PROFILE_VERSION,
        "path": str(path),
        "source_sha256": digest,
        "generation": list(
            {
                json.dumps(
                    {k: v for k, v in r["generation"].items() if k != "messages"},
                    sort_keys=True,
                ): {k: v for k, v in r["generation"].items() if k != "messages"}
                for r in used.values()
            }.values()
        ),
        "prompt_template": _messages("{title}"),
        "item_content_sha256": {
            item: r["content_sha256"] for (item, _), r in used.items()
        },
        "canonical_genres_retained_for_traits": True,
        "pruning": False,
        "statistics": {
            **stat_manifest,
            "source": "task_train_observed_only",
            "rows": len(train),
            "cold_start_population": "warm_train"
            if hasattr(task, "online_session_history")
            else None,
            "sha256": hashlib.sha256(
                pd.util.hash_pandas_object(
                    train[["item_id", "rating"]], index=False
                ).values.tobytes()
            ).hexdigest(),
        },
    }
    manifest.setdefault("item_enrichment", {})["movie_summaries"] = {
        "enabled": True,
        "task_column": "item_summary",
        "source_sha256": digest,
        "source": "agent4rec_title_only_generation",
        "metadata_version": ITEM_PROFILE_VERSION,
    }
    return replace(task, **enriched, manifest=manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items",
        type=Path,
        required=True,
        help="Canonical item parquet with item_id and title (or item_title)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_ITEM_PROFILE_PATH)
    parser.add_argument("--client", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    from beyond_click_sim.llm_clients import make_llm_client

    prepare_item_profiles(
        pd.read_parquet(args.items),
        client=make_llm_client(args.client),
        client_name=args.client,
        model=args.model,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
