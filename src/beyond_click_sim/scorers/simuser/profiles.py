"""Training-only metadata, random history sampling, contrasts and habit arithmetic."""

import random
import numpy as np
import pandas as pd

from .state import digest, identity

STATIC_COLUMNS = ("item_title", "item_genres", "item_summary")


def genres(value):
    values = value.split("|") if isinstance(value, str) else value
    if not isinstance(values, (tuple, list, np.ndarray)) or not len(values):
        raise ValueError("Nonempty canonical genres required")
    if any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError("Invalid canonical genres")
    return tuple(sorted(set(v.strip() for v in values)))


def metadata(catalog, config):
    required = {"item_id", *STATIC_COLUMNS}
    if not required <= set(catalog) or catalog.empty:
        raise ValueError("Catalog requires titles, genres and frozen summaries")
    result = {}
    for row in catalog.to_dict("records"):
        key = identity(row["item_id"])
        if key in result:
            raise ValueError("Duplicate catalog item ID")
        for col in ("item_title", "item_summary"):
            if not isinstance(row[col], str) or not row[col].strip():
                raise ValueError(f"Missing required metadata: {col} for {key}")
        result[key] = {
            "title": row["item_title"].strip()[: config.title_chars],
            "genres": list(genres(row["item_genres"])),
            "summary": row["item_summary"].strip()[: config.summary_chars],
            "training_interaction_count": 0,
            "training_mean_rating": None,
        }
    return dict(sorted(result.items()))


def training_records(X, y, events, catalog):
    if (
        X.empty
        or not X.index.is_unique
        or not X.index.equals(y.index)
        or not y.eq(1).all()
        or y.isna().any()
    ):
        raise ValueError(
            "Fit requires index-aligned observed training interactions only"
        )
    if "is_sampled" in X and X.is_sampled.fillna(False).any():
        raise ValueError("Sampled negatives cannot be historical experiences")
    if not {"user_id", "item_id", "rating"} <= set(X):
        raise ValueError("Observed training user/item IDs and ratings required")
    if (
        events is None
        or not events.index.equals(X.index)
        or not {"event_id", "timestamp", "user_id", "item_id"} <= set(events)
    ):
        raise ValueError("Verified, index-aligned training events required")
    if not events[["user_id", "item_id"]].equals(X[["user_id", "item_id"]]):
        raise ValueError("Training event identity mismatch")
    ratings = pd.to_numeric(X.rating, errors="raise")
    if (
        ratings.isna().any()
        or not ratings.between(1, 5).all()
        or not (ratings % 1 == 0).all()
    ):
        raise ValueError("MovieLens training ratings must be integers from 1 to 5")
    records = []
    for (_, row), (_, event) in zip(X.iterrows(), events.iterrows(), strict=True):
        item, user = identity(row.item_id), identity(row.user_id)
        if item not in catalog:
            raise ValueError(f"Training item absent from catalog: {item}")
        raw_time = event.timestamp
        timestamp = (
            float(raw_time)
            if isinstance(raw_time, (int, float, np.number))
            else pd.Timestamp(raw_time).timestamp()
        )
        if not np.isfinite(timestamp):
            raise ValueError("Finite training timestamps required")
        records.append(
            {
                "user": user,
                "item": item,
                "event": identity(event.event_id),
                "timestamp": timestamp,
                "rating": int(row.rating),
            }
        )
    if len({r["event"] for r in records}) != len(records):
        raise ValueError("Duplicate training event identity")
    records.sort(key=lambda r: (r["user"], r["timestamp"], r["event"]))
    return records


def add_training_statistics(catalog, records):
    values = {}
    for record in records:
        values.setdefault(record["item"], []).append(record["rating"])
    for item, ratings in values.items():
        catalog[item]["training_interaction_count"] = len(ratings)
        catalog[item]["training_mean_rating"] = float(np.mean(ratings))


def preference(rating):
    return "liked" if rating >= 4 else "disliked" if rating <= 2 else "neutral"


def habits(history, catalog):
    mean = float(np.mean([r["rating"] for r in history]))
    return {
        "engagement_interaction_count": len(history),
        "conformity_mean_squared_deviation": float(
            np.mean(
                [
                    (r["rating"] - catalog[r["item"]]["training_mean_rating"]) ** 2
                    for r in history
                ]
            )
        ),
        "variety_distinct_genres": len(
            {g for r in history for g in catalog[r["item"]]["genres"]}
        ),
        "mean_historical_rating": mean,
        "pickiness": "not picky"
        if mean >= 4.5
        else "moderately picky"
        if mean >= 3.5
        else "extremely picky",
    }


def preparation(user, history, population, catalog, config):
    rng_key = digest({"seed": config.seed, "user": user, "stage": "persona_sampling"})
    rng = random.Random(int(rng_key, 16))
    sampled = sorted(
        rng.sample(history, min(len(history), config.history_sample)),
        key=lambda r: (r["timestamp"], r["event"]),
    )
    own = {r["item"]: r for r in history}
    contrast = {}
    for event in population:
        if event["user"] != user and event["item"] not in own:
            contrast.setdefault(event["item"], event)
    rho = min(config.contrast_size, len(own), len(contrast))
    if not rho:
        raise ValueError(f"No eligible training contrast for {user}")
    subsets = []
    for _ in range(config.contrast_subsets):
        target = rng.sample(sorted(own), rho)
        other = rng.sample(sorted(contrast), rho)
        mixed = target + other
        rng.shuffle(mixed)
        subsets.append(
            {
                "items": mixed,
                "target": target,
                "target_events": [own[i]["event"] for i in target],
                "contrast_events": [contrast[i]["event"] for i in other],
            }
        )
    return {
        "rng_identity": rng_key,
        "sampled_event_ids": [r["event"] for r in sampled],
        "sampled_history": [
            {
                "title": catalog[r["item"]]["title"],
                "genres": catalog[r["item"]]["genres"],
                "rating": r["rating"],
                "preference": preference(r["rating"]),
            }
            for r in sampled
        ],
        "rho": rho,
        "subsets": subsets,
        "habits": habits(history, catalog),
    }
