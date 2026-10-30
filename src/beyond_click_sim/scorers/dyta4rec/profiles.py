"""Explicit training-only statistics and metadata; no borrowed trait algorithm."""

import json
import math
from statistics import mean, median

import pandas as pd

from ..recagent.profiles import AGES, OCCUPATIONS, unix_time


def genres(value):
    if isinstance(value, str):
        values = json.loads(value) if value.startswith("[") else value.split("|")
    else:
        values = value
    if (
        not isinstance(values, (tuple, list))
        or not values
        or any(not isinstance(g, str) or not g.strip() for g in values)
    ):
        raise ValueError("Genres must be a nonempty list or serialized genre string")
    return list(dict.fromkeys(g.strip() for g in values))


def metadata(row, config):
    title = row["item_title"]
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Missing item title")
    item_genres = genres(row["item_genres"])
    if max(len(title), len(json.dumps(item_genres))) > config.metadata_max_chars:
        raise ValueError("Item metadata exceeds declared field limit; no truncation")
    return {"title": title, "genres": item_genres}


def demographic_categories(rows):
    result = {}
    for column, name, lookup in (
        ("user_age", "age_category", AGES),
        ("user_gender", "recorded_gender", {"M": "male", "F": "female"}),
        ("user_occupation", "occupation_category", OCCUPATIONS),
    ):
        if column not in rows:
            raise ValueError(f"Mandatory demographics column missing: {column}")
        values = rows[column].drop_duplicates()
        if len(values) != 1:
            raise ValueError(f"Inconsistent demographics: {column}")
        value = values.iloc[0]
        if pd.isna(value):
            result[name] = "unavailable"
        else:
            if column != "user_gender":
                if float(value) != int(value):
                    raise ValueError(f"Invalid category: {column}")
                value = int(value)
            if value not in lookup:
                raise ValueError(f"Invalid category: {column}")
            result[name] = lookup[value]
    return result


def prepare_evidence(X, y, training_events, config, *, sampled_column):
    required = ["user_id", "item_id", "item_title", "item_genres", "rating"]
    if X.empty or not X.index.is_unique or not X.index.equals(y.index):
        raise ValueError("Training rows require nonempty unique, aligned indices")
    if any(c not in X for c in required) or X[required].isna().any().any():
        raise ValueError("Missing training identity, rating or item metadata")
    if y.isna().any() or not y.eq(1).all():
        raise ValueError("Only observed interactions may initialize DyTA4Rec")
    if sampled_column in X and (
        X[sampled_column].isna().any() or X[sampled_column].astype(bool).any()
    ):
        raise ValueError("Sampled negatives cannot be training experiences")
    if not X.index.equals(training_events.index):
        raise ValueError("Verified training events must have exactly aligned indices")
    if (
        training_events.event_id.isna().any()
        or training_events.event_id.duplicated().any()
    ):
        raise ValueError("Event identities must be unique")
    for column in ("user_id", "item_id"):
        if not X[column].equals(training_events[column]):
            raise ValueError("Verified event user/item identities do not match")
    frame = X[
        required + [c for c in ("user_age", "user_gender", "user_occupation") if c in X]
    ].copy()
    frame["rating"] = pd.to_numeric(frame.rating, errors="raise")
    if not frame.rating.isin(range(1, 6)).all():
        raise ValueError("ML-1M historical ratings must be integers from 1 to 5")
    frame["event_id"] = training_events.event_id.map(str)
    if frame.event_id.duplicated().any():
        raise ValueError("Event identities collide after serialization")
    frame["timestamp"] = training_events.timestamp.map(unix_time)
    frame["user_id"] = frame.user_id.map(str)
    frame["item_id"] = frame.item_id.map(str)
    # Stable source identities, rather than dataframe input order, break time ties.
    frame = frame.sort_values(["timestamp", "event_id"], kind="stable")
    population = {}
    for item, rows in frame.groupby("item_id", sort=False):
        descriptions = [metadata(r, config) for _, r in rows.iterrows()]
        if any(d != descriptions[0] for d in descriptions):
            raise ValueError("Inconsistent item metadata in training")
        population[item] = {
            "training_interaction_count": len(rows),
            "training_mean_rating": float(rows.rating.mean()),
        }
    users = {}
    for user, rows in frame.groupby("user_id", sort=False):
        events = [
            {
                "ref": f"E{i}",
                "event_id": r["event_id"],
                "item_id": r["item_id"],
                "timestamp": float(r["timestamp"]),
                "position": i,
                "rating": int(r["rating"]),
                **metadata(r, config),
            }
            for i, (_, r) in enumerate(rows.iterrows(), 1)
        ]
        users[user] = {
            "events": events,
            "demographics": demographic_categories(rows),
            "statistics": statistics(events, population),
            "cutoff": events[-1]["timestamp"],
        }
    return users, population


def statistics(events, population):
    counts = [population[e["item_id"]]["training_interaction_count"] for e in events]
    qualities = [population[e["item_id"]]["training_mean_rating"] for e in events]
    by_genre = {}
    for event in events:
        for genre in event["genres"]:
            by_genre.setdefault(genre, []).append(event["rating"])
    return {
        "observed_events": len(events),
        "distinct_items": len({e["item_id"] for e in events}),
        "rating_histogram": {
            str(r): sum(e["rating"] == r for e in events) for r in range(1, 6)
        },
        "mean_own_rating": mean(e["rating"] for e in events),
        "genre_statistics": {
            g: {"observed_events": len(rs), "mean_own_rating": mean(rs)}
            for g, rs in sorted(by_genre.items())
        },
        "distinct_genres": len(by_genre),
        "population_count_over_own_events": {
            "mean": mean(counts),
            "median": median(counts),
        },
        "population_mean_rating_over_own_events": {
            "mean": mean(qualities),
            "median": median(qualities),
        },
        "population_associations_by_own_rating": {
            str(r): {
                "mean_training_count": mean(cs) if cs else None,
                "mean_training_rating": mean(qs) if qs else None,
            }
            for r in range(1, 6)
            for cs, qs in [
                (
                    [c for e, c in zip(events, counts) if e["rating"] == r],
                    [q for e, q in zip(events, qualities) if e["rating"] == r],
                )
            ]
        },
    }


def event_view(event):
    """Prompt allowlist: no raw IDs, timestamps, exposures, or synthetic feelings."""
    return {k: event[k] for k in ("ref", "position", "title", "genres", "rating")}


def candidate_view(X, population, config, *, group_column):
    required = ["user_id", "item_id", group_column, "item_title", "item_genres"]
    if any(c not in X for c in required) or X[required].isna().any().any():
        raise ValueError("Missing candidate identity or metadata")
    if not X.index.is_unique or X.empty:
        raise ValueError("Candidates need nonempty unique indices")
    for _, group in X.groupby(group_column, sort=False):
        if (
            group.user_id.nunique() != 1
            or group.item_id.duplicated().any()
            or len(group) > 20
        ):
            raise ValueError("Groups need one user and at most 20 distinct items")
    return [
        {
            "label": f"C{i}",
            **metadata(row, config),
            **population.get(
                str(row.item_id),
                {
                    "training_interaction_count": 0,
                    "training_mean_rating": None,
                },
            ),
        }
        for i, (_, row) in enumerate(X.iterrows(), 1)
    ]


def initialization_size(n, config):
    blocks = math.ceil(n / config.block_size)
    level, merges = blocks, 0
    while level > 1:
        level = math.ceil(level / config.merge_fan_in)
        merges += level
    return {
        "blocks": blocks,
        "merges": merges,
        "summary_records": blocks + merges,
        "llm_calls": 2 * (blocks + merges) + 3,
    }
