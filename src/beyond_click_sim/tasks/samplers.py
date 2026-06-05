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
