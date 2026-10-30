"""simuser_graph_v1: sparse path counts, history projection and real witnesses."""

from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix

from .profiles import preference
from .state import digest


class TrainingGraph:
    def __init__(self, catalog, events):
        self.catalog = catalog
        self.items = tuple(sorted(catalog))
        self.positions = {item: i for i, item in enumerate(self.items)}
        self.users = tuple(sorted({e["user"] for e in events}))
        user_positions = {u: i for i, u in enumerate(self.users)}
        self.histories = {u: set() for u in self.users}
        self.item_users = {i: set() for i in self.items}
        self.edges = {}
        for e in events:
            self.histories[e["user"]].add(e["item"])
            self.item_users[e["item"]].add(e["user"])
            self.edges.setdefault((e["user"], e["item"]), []).append(
                {
                    "event_id": e["event"],
                    "rating": e["rating"],
                    "relation": preference(e["rating"]),
                }
            )
        pairs = sorted(self.edges)
        self.B = csr_matrix(
            (
                np.ones(len(pairs)),
                (
                    [user_positions[u] for u, _ in pairs],
                    [self.positions[i] for _, i in pairs],
                ),
            ),
            shape=(len(self.users), len(self.items)),
        )
        genre_names = sorted({g for item in catalog.values() for g in item["genres"]})
        genre_positions = {g: i for i, g in enumerate(genre_names)}
        pairs = [
            (i, genre_positions[g])
            for i, key in enumerate(self.items)
            for g in catalog[key]["genres"]
        ]
        self.C = csr_matrix(
            (np.ones(len(pairs)), ([p[0] for p in pairs], [p[1] for p in pairs])),
            shape=(len(self.items), len(genre_names)),
        )
        self.ui_diagonal = np.asarray(self.B.sum(axis=0)).ravel()
        self.g_diagonal = np.asarray(self.C.sum(axis=1)).ravel()
        self.vectors = None
        # Per-instance caches must not retain completed runs' large graphs via
        # a class-level lru_cache keyed by self.
        self.item_channels = lru_cache(maxsize=128)(self._item_channels)
        self.user_projection = lru_cache(maxsize=2048)(self._user_projection)

    def _channels(self, indices):
        ui = (self.B[:, indices].T @ self.B).toarray()
        genre = (self.C[indices] @ self.C.T).toarray()
        channels = []
        for matrix, diagonal in ((ui, self.ui_diagonal), (genre, self.g_diagonal)):
            denominator = diagonal[indices, None] + diagonal[None, :]
            channels.append(
                np.divide(
                    2 * matrix,
                    denominator,
                    out=np.zeros_like(matrix),
                    where=denominator != 0,
                )
            )
        return ui, genre, *channels

    def _item_channels(self, item):
        return tuple(row[0] for row in self._channels([self.positions[item]]))

    def _user_projection(self, user):
        indices = [self.positions[i] for i in sorted(self.histories[user])]
        # Bounded batches avoid an all-path enumeration or a dense catalog square.
        total = np.zeros(len(self.items))
        for start in range(0, len(indices), 64):
            _, _, ui, genre = self._channels(indices[start : start + 64])
            total += (0.5 * ui + 0.5 * genre).sum(axis=0)
        return total / len(indices)

    def ranked(self, user, query):
        if self.vectors is None:
            raise ValueError("Graph embeddings are not initialized")
        ui_count, g_count, ui, genre = self.item_channels(query)
        user_term = self.user_projection(user)
        cosine = np.clip(self.vectors @ self.vectors[self.positions[query]], -1, 1)
        sem = (1 + cosine) / 2
        item = 0.5 * ui + 0.5 * genre
        scores = 0.75 * (0.8 * item + 0.2 * user_term) + 0.25 * sem
        return [
            {
                "item_id": self.items[i],
                "query_item": query,
                "score": float(scores[i]),
                "ui_count": float(ui_count[i]),
                "genre_count": float(g_count[i]),
                "ui_pathsim": float(ui[i]),
                "genre_pathsim": float(genre[i]),
                "item_similarity": float(item[i]),
                "user_projection": float(user_term[i]),
                "semantic": float(sem[i]),
            }
            for i in sorted(
                range(len(self.items)), key=lambda i: (-scores[i], self.items[i])
            )
            if self.items[i] != query
        ]

    def witnesses(self, user, query, item):
        ui = [
            [query, u, item]
            for u in sorted(self.item_users[query] & self.item_users[item])[:2]
        ]
        genre = [
            [query, "genre:" + g, item]
            for g in sorted(
                set(self.catalog[query]["genres"]) & set(self.catalog[item]["genres"])
            )[:2]
        ]
        user_ui, user_genre = [], []
        for h in sorted(self.histories[user]):
            for v in sorted(self.item_users[h] & self.item_users[item]):
                if len(user_ui) < 2:
                    user_ui.append([user, h, v, item])
            for g in sorted(
                set(self.catalog[h]["genres"]) & set(self.catalog[item]["genres"])
            ):
                if len(user_genre) < 2:
                    user_genre.append([user, h, "genre:" + g, item])
            if len(user_ui) == len(user_genre) == 2:
                break
        return {
            "item_user_item": ui,
            "item_genre_item": genre,
            "user_history_user_item": user_ui,
            "user_history_genre_item": user_genre,
        }

    def retrieve(self, user, ranked, k):
        merged = {}
        for label, rows in ranked.items():
            for row in rows[:k]:
                item = row["item_id"]
                if item not in merged:
                    merged[item] = {
                        "id": "G" + digest(item),
                        "item_id": item,
                        **self.catalog[item],
                        "target_training_feedback": self.edges.get((user, item), []),
                        "queries": [],
                    }
                paths = self.witnesses(user, row["query_item"], item)
                merged[item]["queries"].append(
                    {
                        "candidate_label": label,
                        **row,
                        "witnesses": paths,
                        "semantic_only": not any(paths.values()),
                    }
                )
        return [merged[k] for k in sorted(merged)]
