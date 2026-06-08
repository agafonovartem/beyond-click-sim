from __future__ import annotations

import random
from typing import Any

import pandas as pd

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

            rows.append(
                {
                    self.user_column: user_id,
                    self.item_column: positive[self.item_column],
                    self.target_column: 1, # Interaction target-only!
                    self.sampled_column: False,
                    self.candidate_group_column: group_id,
                }
            )

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
                all_items=all_items,
                observed_items=observed_items,
            ):
                rows.append(
                    {
                        self.user_column: user_id,
                        self.item_column: item_id,
                        self.target_column: 0,
                        self.sampled_column: True,
                        self.candidate_group_column: group_id,
                    }
                )

        return pd.DataFrame(rows, columns=output_columns)

    def _sample_negative_items(
        self,
        *,
        rng: random.Random,
        all_items: list[Any],
        observed_items: set[Any],
    ) -> list[Any]:
        """Sample unseen items without scanning the full item universe."""

        negatives: list[Any] = []
        selected: set[Any] = set()
        while len(negatives) < self.negative_ratio:
            item_id = rng.choice(all_items)
            if item_id in observed_items or item_id in selected:
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


class FixedSizeUserInteractionCandidateSampler(CandidateSampler):
    """Build one Agent4Rec-style candidate list per user.

    For each user in a held-out split, sample up to `total_items` candidates with
    ratio 1:`negative_ratio`: `k` observed positives and `k * negative_ratio`
    never-observed negatives, where `k <= total_items // (negative_ratio + 1)`.
    """

    def __init__(
        self,
        negative_ratio: int,
        total_items: int = 20,
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
    ) -> pd.DataFrame:
        """Return one fixed-size candidate group per user."""

        if self.negative_ratio < 1:
            raise ValueError("negative_ratio must be positive.")
        if self.total_items < self.negative_ratio + 1:
            raise ValueError("total_items must fit at least one positive group.")

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

        for user_id, positive_items in positives_by_user.items():
            group_id = self._candidate_group_id(user_id)
            group_rng = random.Random(f"{self.seed}:{group_id}")
            positive_count = min(len(positive_items), max_positive_items)
            selected_positives = group_rng.sample(positive_items, positive_count)
            negative_count = len(selected_positives) * self.negative_ratio

            observed_items = observed_by_user.get(user_id, set())
            if len(all_items) - len(observed_items) < negative_count:
                raise ValueError(
                    f"User {user_id!r} has only "
                    f"{len(all_items) - len(observed_items)} available negative "
                    f"items, need {negative_count}."
                )

            self._append_rows(
                row_values,
                user_id=user_id,
                item_ids=selected_positives,
                target=1,
                sampled=False,
                group_id=group_id,
            )

            negative_items = self._sample_negative_items(
                rng=group_rng,
                all_items=all_items,
                observed_items=observed_items,
                negative_count=negative_count,
            )
            self._append_rows(
                row_values,
                user_id=user_id,
                item_ids=negative_items,
                target=0,
                sampled=True,
                group_id=group_id,
            )

        return pd.DataFrame(row_values, columns=output_columns)

    def _append_rows(
        self,
        row_values: dict[str, list[Any]],
        *,
        user_id: Any,
        item_ids: list[Any],
        target: int,
        sampled: bool,
        group_id: str,
    ) -> None:
        """Append candidate rows to column lists."""

        count = len(item_ids)
        row_values[self.user_column].extend([user_id] * count)
        row_values[self.item_column].extend(item_ids)
        row_values[self.target_column].extend([target] * count)
        row_values[self.sampled_column].extend([sampled] * count)
        row_values[self.candidate_group_column].extend([group_id] * count)

    @staticmethod
    def _sample_negative_items(
        *,
        rng: random.Random,
        all_items: list[Any],
        observed_items: set[Any],
        negative_count: int,
    ) -> list[Any]:
        """Sample unseen items without scanning the full item universe."""

        negatives: list[Any] = []
        selected: set[Any] = set()
        while len(negatives) < negative_count:
            item_id = rng.choice(all_items)
            if item_id in observed_items or item_id in selected:
                continue
            negatives.append(item_id)
            selected.add(item_id)
        return negatives

    def _candidate_group_id(self, user_id: Any) -> str:
        """Create one stable candidate group id per user."""

        return f"candidate:user:{user_id}"
