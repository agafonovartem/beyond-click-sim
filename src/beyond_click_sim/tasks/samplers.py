from __future__ import annotations

import random
from typing import Any

import pandas as pd

from beyond_click_sim.tasks._sampling import stable_sample_values
from beyond_click_sim.tasks.base import CandidateSampler


class NonInteractionCandidateSampler(CandidateSampler):
    """Add sampled non-interactions to held-out observed positives."""

    def __init__(
        self,
        negative_ratio: int,
        seed: int = 0,
        user_column: str = "user_id",
        item_column: str = "item_id",
        target_column: str = "target",
        sampled_column: str = "sampled",
        candidate_group_column: str = "candidate_group",
    ) -> None:
        """Configure a 1:`negative_ratio` candidate sampler."""

        super().__init__(seed=seed)
        self.negative_ratio = negative_ratio
        self.user_column = user_column
        self.item_column = item_column
        self.target_column = target_column
        self.sampled_column = sampled_column
        self.candidate_group_column = candidate_group_column

    def sample(
        self,
        positives: pd.DataFrame,
        *,
        interactions: pd.DataFrame,
        items: pd.DataFrame,
        excluded_pairs: set[tuple[Any, Any]] | None = None,
    ) -> pd.DataFrame:
        """Return held-out positives with sampled items the user never observed."""

        if self.negative_ratio < 1:
            raise ValueError("negative_ratio must be positive.")

        output_columns = [
            self.user_column,
            self.item_column,
            self.target_column,
            self.sampled_column,
            self.candidate_group_column,
        ]
        if positives.empty:
            return pd.DataFrame(columns=output_columns)

        all_items = items[self.item_column].drop_duplicates().tolist()
        observed_by_user = (
            interactions.groupby(self.user_column, sort=False)[self.item_column]
            .agg(lambda values: set(values))
            .to_dict()
        )

        rows: list[dict[str, Any]] = []
        for position, positive in positives.reset_index(drop=True).iterrows():
            user_id = positive[self.user_column]
            group_id = self._candidate_group_id(positive, position)

            group_rows = [
                {
                    self.user_column: user_id,
                    self.item_column: positive[self.item_column],
                    self.target_column: 1, # Interaction target-only!
                    self.sampled_column: False,
                    self.candidate_group_column: group_id,
                }
            ]

            observed_items = observed_by_user.get(user_id, set())
            if len(all_items) - len(observed_items) < self.negative_ratio:
                raise ValueError(
                    f"User {user_id!r} has only "
                    f"{len(all_items) - len(observed_items)} available negative "
                    f"items, need {self.negative_ratio}."
                )

            # Seed per candidate set so negatives do not depend on row order.
            group_rng = random.Random(f"{self.seed}:{group_id}")
            for item_id in self._sample_negative_items(
                rng=group_rng,
                user_id=user_id,
                all_items=all_items,
                observed_items=observed_items,
                excluded_pairs=excluded_pairs,
            ):
                group_rows.append(
                    {
                        self.user_column: user_id,
                        self.item_column: item_id,
                        self.target_column: 0,
                        self.sampled_column: True,
                        self.candidate_group_column: group_id,
                    }
                )
            rows.extend(_shuffled_group_rows(group_rows, seed=self.seed, group_id=group_id))

        return pd.DataFrame(rows, columns=output_columns)

    def _sample_negative_items(
        self,
        *,
        rng: random.Random,
        user_id: Any,
        all_items: list[Any],
        observed_items: set[Any],
        excluded_pairs: set[tuple[Any, Any]] | None,
    ) -> list[Any]:
        """Sample unseen items without scanning the full item universe."""

        negatives: list[Any] = []
        selected: set[Any] = set()
        attempts = 0
        max_attempts = max(100, self.negative_ratio * 100)
        while len(negatives) < self.negative_ratio:
            attempts += 1
            if attempts > max_attempts:
                return _sample_negative_items_from_scan(
                    rng=rng,
                    user_id=user_id,
                    all_items=all_items,
                    observed_items=observed_items,
                    excluded_pairs=excluded_pairs,
                    negative_count=self.negative_ratio,
                )
            item_id = rng.choice(all_items)
            if (
                item_id in observed_items
                or item_id in selected
                or _is_excluded(
                    excluded_pairs,
                    user_id=user_id,
                    item_id=item_id,
                )
            ):
                continue
            negatives.append(item_id)
            selected.add(item_id)
        return negatives

    def _candidate_group_id(self, positive: pd.Series, position: int) -> str:
        """Create a stable group id for one held-out positive candidate set."""

        user_id = positive[self.user_column]
        if "interaction_id" in positive.index:
            source_id = positive["interaction_id"]
        else:
            source_id = position
        return f"candidate:{user_id}:{source_id}"


class CappedUserInteractionCandidateSampler(CandidateSampler):
    """Build capped Agent4Rec-style candidate lists for held-out users.

    Each candidate group contains up to `total_items` candidates with ratio
    1:`negative_ratio`: `k` observed positives and `k * negative_ratio`
    never-observed negatives, where `k <= total_items // (negative_ratio + 1)`.
    Users with more held-out positives get multiple candidate groups, so the
    sampler does not discard positives.

    `max_eval_users` is an evaluation budget: when provided, sample that many
    held-out users first, then build all candidate groups for their positives.
    """

    def __init__(
        self,
        negative_ratio: int,
        total_items: int = 20,
        max_eval_users: int | None = None,
        seed: int = 0,
        user_column: str = "user_id",
        item_column: str = "item_id",
        target_column: str = "target",
        sampled_column: str = "sampled",
        candidate_group_column: str = "candidate_group",
    ) -> None:
        super().__init__(seed=seed)
        self.negative_ratio = negative_ratio
        self.total_items = total_items
        self.max_eval_users = max_eval_users
        self.user_column = user_column
        self.item_column = item_column
        self.target_column = target_column
        self.sampled_column = sampled_column
        self.candidate_group_column = candidate_group_column

    def sample(
        self,
        positives: pd.DataFrame,
        *,
        interactions: pd.DataFrame,
        items: pd.DataFrame,
        excluded_pairs: set[tuple[Any, Any]] | None = None,
    ) -> pd.DataFrame:
        """Return capped candidate groups containing all held-out positives."""

        if self.negative_ratio < 1:
            raise ValueError("negative_ratio must be positive.")
        if self.total_items < self.negative_ratio + 1:
            raise ValueError("total_items must fit at least one positive group.")
        if self.max_eval_users is not None and self.max_eval_users < 1:
            raise ValueError("max_eval_users must be positive when provided.")

        output_columns = [
            self.user_column,
            self.item_column,
            self.target_column,
            self.sampled_column,
            self.candidate_group_column,
        ]
        if positives.empty:
            return pd.DataFrame(columns=output_columns)

        all_items = items[self.item_column].drop_duplicates().tolist()
        observed_by_user = (
            interactions.groupby(self.user_column, sort=False)[self.item_column]
            .agg(lambda values: set(values))
            .to_dict()
        )

        row_values: dict[str, list[Any]] = {column: [] for column in output_columns}
        max_positive_items = self.total_items // (self.negative_ratio + 1)
        positives_by_user = positives.groupby(self.user_column, sort=False)[
            self.item_column
        ].agg(lambda values: list(dict.fromkeys(values)))
        if self.max_eval_users is not None:
            selected_users = stable_sample_values(
                positives_by_user.index,
                n=self.max_eval_users,
                seed=self.seed,
            )
            positives_by_user = positives_by_user.loc[selected_users]

        for user_id, positive_items in positives_by_user.items():
            stable_positive_items = stable_sample_values(
                positive_items,
                n=None,
                seed=self.seed,
            )
            positive_rng = random.Random(f"{self.seed}:positives:{user_id}")
            shuffled_positives = positive_rng.sample(
                stable_positive_items,
                len(stable_positive_items),
            )

            for chunk_position, selected_positives in enumerate(
                _chunks(shuffled_positives, max_positive_items)
            ):
                group_id = self._candidate_group_id(
                    user_id=user_id,
                    chunk_position=chunk_position,
                )
                group_rng = random.Random(f"{self.seed}:{group_id}")
                negative_count = len(selected_positives) * self.negative_ratio

                observed_items = observed_by_user.get(user_id, set())
                if len(all_items) - len(observed_items) < negative_count:
                    raise ValueError(
                        f"User {user_id!r} has only "
                        f"{len(all_items) - len(observed_items)} available negative "
                        f"items, need {negative_count}."
                    )

                group_rows: list[dict[str, Any]] = []
                self._append_rows(
                    group_rows,
                    user_id=user_id,
                    item_ids=selected_positives,
                    target=1,
                    sampled=False,
                    group_id=group_id,
                )

                negative_items = self._sample_negative_items(
                    rng=group_rng,
                    user_id=user_id,
                    all_items=all_items,
                    observed_items=observed_items,
                    excluded_pairs=excluded_pairs,
                    negative_count=negative_count,
                )
                self._append_rows(
                    group_rows,
                    user_id=user_id,
                    item_ids=negative_items,
                    target=0,
                    sampled=True,
                    group_id=group_id,
                )
                for row in _shuffled_group_rows(
                    group_rows,
                    seed=self.seed,
                    group_id=group_id,
                ):
                    for column in output_columns:
                        row_values[column].append(row[column])

        return pd.DataFrame(row_values, columns=output_columns)

    def _append_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        user_id: Any,
        item_ids: list[Any],
        target: int,
        sampled: bool,
        group_id: str,
    ) -> None:
        """Append candidate rows to a group row list."""

        for item_id in item_ids:
            rows.append(
                {
                    self.user_column: user_id,
                    self.item_column: item_id,
                    self.target_column: target,
                    self.sampled_column: sampled,
                    self.candidate_group_column: group_id,
                }
            )

    @staticmethod
    def _sample_negative_items(
        *,
        rng: random.Random,
        user_id: Any,
        all_items: list[Any],
        observed_items: set[Any],
        excluded_pairs: set[tuple[Any, Any]] | None,
        negative_count: int,
    ) -> list[Any]:
        """Sample unseen items without scanning the full item universe."""

        negatives: list[Any] = []
        selected: set[Any] = set()
        attempts = 0
        max_attempts = max(100, negative_count * 100)
        while len(negatives) < negative_count:
            attempts += 1
            if attempts > max_attempts:
                return _sample_negative_items_from_scan(
                    rng=rng,
                    user_id=user_id,
                    all_items=all_items,
                    observed_items=observed_items,
                    excluded_pairs=excluded_pairs,
                    negative_count=negative_count,
                )
            item_id = rng.choice(all_items)
            if (
                item_id in observed_items
                or item_id in selected
                or _is_excluded(
                    excluded_pairs,
                    user_id=user_id,
                    item_id=item_id,
                )
            ):
                continue
            negatives.append(item_id)
            selected.add(item_id)
        return negatives

    def _candidate_group_id(self, *, user_id: Any, chunk_position: int) -> str:
        """Create one stable candidate group id per user positive chunk."""

        return f"candidate:user:{user_id}:chunk:{chunk_position}"


class PostSplitUserSampler:
    """Limit held-out rows to a deterministic sample of users after splitting.

    Samples users (not candidate items): it draws a random subset of real
    held-out users and keeps their rows, adding no synthetic negatives.
    Intentionally not a ``CandidateSampler`` — it has a different ``sample()``
    contract (``sample(rows, *, train)`` returning subset rows plus a selection
    summary) and must not be used interchangeably with one.
    """

    def __init__(
        self,
        n_users: int | None,
        seed: int = 0,
        user_column: str = "user_id",
        require_train_history: bool = True,
    ) -> None:
        if n_users is not None and n_users < 1:
            raise ValueError("n_users must be positive when provided.")
        self.n_users = n_users
        self.seed = seed
        self.user_column = user_column
        self.require_train_history = require_train_history

    def sample(
        self,
        rows: pd.DataFrame,
        *,
        train: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        if rows.empty:
            return rows.copy(), {
                "eligible_users": 0,
                "selected_users": 0,
                "rows_before": 0,
                "rows_after": 0,
            }
        if self.user_column not in rows.columns:
            raise ValueError(f"Missing user column: {self.user_column!r}")
        if self.require_train_history and self.user_column not in train.columns:
            raise ValueError(f"Missing user column in train: {self.user_column!r}")

        if self.require_train_history:
            train_users = set(train[self.user_column])
            eligible_rows = rows[rows[self.user_column].isin(train_users)].copy()
        else:
            eligible_rows = rows.copy()

        eligible_users = eligible_rows[self.user_column].drop_duplicates()
        selected_users = stable_sample_values(
            eligible_users,
            n=self.n_users,
            seed=self.seed,
        )
        sampled = eligible_rows[
            eligible_rows[self.user_column].isin(selected_users)
        ].copy()
        return sampled, {
            "eligible_users": int(eligible_users.nunique()),
            "selected_users": int(sampled[self.user_column].nunique()),
            "rows_before": int(len(rows)),
            "rows_after": int(len(sampled)),
        }


def _chunks(values: list[Any], size: int) -> list[list[Any]]:
    """Split values into consecutive non-empty chunks."""

    return [values[start : start + size] for start in range(0, len(values), size)]


def _shuffled_group_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    group_id: str,
) -> list[dict[str, Any]]:
    """Return candidate rows in a stable shuffled order within one group."""

    return random.Random(f"{seed}:order:{group_id}").sample(rows, len(rows))


def _is_excluded(
    excluded_pairs: set[tuple[Any, Any]] | None,
    *,
    user_id: Any,
    item_id: Any,
) -> bool:
    """Return whether a sampled negative pair is blocked for this split."""

    return excluded_pairs is not None and (user_id, item_id) in excluded_pairs


def _sample_negative_items_from_scan(
    *,
    rng: random.Random,
    user_id: Any,
    all_items: list[Any],
    observed_items: set[Any],
    excluded_pairs: set[tuple[Any, Any]] | None,
    negative_count: int,
) -> list[Any]:
    """Fallback sampler used when rejection sampling cannot find enough items."""

    available_items = [
        item_id
        for item_id in all_items
        if item_id not in observed_items
        and not _is_excluded(
            excluded_pairs,
            user_id=user_id,
            item_id=item_id,
        )
    ]
    if len(available_items) < negative_count:
        raise ValueError(
            f"User {user_id!r} has only {len(available_items)} available "
            f"negative items, need {negative_count}."
        )
    return rng.sample(available_items, negative_count)
