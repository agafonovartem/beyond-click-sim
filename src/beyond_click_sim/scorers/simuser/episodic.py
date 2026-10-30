"""Literal history-initialized memories and fixed-query cosine retrieval."""

import numpy as np

from .state import digest


def memory_records(history, catalog):
    records = []
    for event in history:
        rating = event["rating"]
        feeling = (
            "liked"
            if rating > 4
            else "disliked"
            if rating <= 2
            else "felt neutral about"
        )
        item = catalog[event["item"]]
        records.append(
            {
                "id": "E" + digest(event["event"]),
                "event_id": event["event"],
                "timestamp": event["timestamp"],
                "item_id": event["item"],
                "rating": rating,
                "text": f"I interacted with {item['title']} ({', '.join(item['genres'])}). "
                f"I {feeling} it; my historical rating was {rating}/5.",
            }
        )
    return records


def ranked_queries(records, vectors, query_vectors):
    similarities = np.clip(query_vectors @ vectors.T, -1, 1)
    return [
        sorted(zip(records, row.tolist()), key=lambda pair: (-pair[1], pair[0]["id"]))
        for row in similarities
    ]


def retrieve(ranked, k):
    merged = {}
    for query_index, rows in enumerate(ranked):
        for record, similarity in rows[:k]:
            key = record["id"]
            if key not in merged:
                merged[key] = {**record, "cosine": similarity, "query_indices": []}
            merged[key]["cosine"] = max(merged[key]["cosine"], similarity)
            merged[key]["query_indices"].append(query_index)
    return sorted(merged.values(), key=lambda r: (-r["cosine"], r["id"]))
