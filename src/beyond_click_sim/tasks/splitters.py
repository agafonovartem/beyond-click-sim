from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from beyond_click_sim.tasks.base import SplitFrames, Splitter


class RandomFractionSplitter(Splitter):
    """Randomly split rows globally or inside each group."""

    def __init__(
        self,
        train_fraction: float = 0.7,
        val_fraction: float = 0.0,
        test_fraction: float = 0.3,
        seed: int = 0,
        group_column: str | None = "user_id", # per-user interaction split, not user-level split
        stratify_column: str | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.group_column = group_column
        self.stratify_column = stratify_column

    def split(self, interactions: pd.DataFrame) -> SplitFrames:
        self._validate_fractions()
        if interactions.empty:
            empty = interactions.copy()
            return SplitFrames(train=empty, val=empty.copy(), test=empty.copy())

        if self.group_column is None:
            return self._split_frame(interactions, random_state=self.seed)

        train_parts: list[pd.DataFrame] = []
        val_parts: list[pd.DataFrame] = []
        test_parts: list[pd.DataFrame] = []

        for group_position, (group_key, group) in enumerate(
            interactions.groupby(self.group_column, sort=True)
        ):
            # Keep grouped splitting simple and explicit: sklearn handles each group.
            split = self._split_frame(
                group,
                random_state=self.seed + group_position,
                group_key=group_key,
            )
            train_parts.append(split.train)
            val_parts.append(split.val)
            test_parts.append(split.test)

        return SplitFrames(
            train=self._concat_parts(train_parts, interactions),
            val=self._concat_parts(val_parts, interactions),
            test=self._concat_parts(test_parts, interactions),
        )

    def _validate_fractions(self) -> None:
        fractions = (self.train_fraction, self.val_fraction, self.test_fraction)
        if any(fraction < 0 for fraction in fractions):
            raise ValueError("Split fractions must be non-negative.")
        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError("Split fractions must sum to 1.")

    def _split_frame(
        self,
        frame: pd.DataFrame,
        *,
        random_state: int,
        group_key: Any | None = None,
    ) -> SplitFrames:
        train_val = frame
        test = self._empty_like(frame)
        if self.test_fraction > 0:
            train_val, test = self._train_test_split(
                frame,
                test_size=self.test_fraction,
                random_state=random_state,
                group_key=group_key,
            )

        train = train_val
        val = self._empty_like(frame)
        if self.val_fraction > 0:
            val_size = self.val_fraction / (self.train_fraction + self.val_fraction)
            train, val = self._train_test_split(
                train_val,
                test_size=val_size,
                random_state=random_state,
                group_key=group_key,
            )

        return SplitFrames(
            train=train.reset_index(drop=True),
            val=val.reset_index(drop=True),
            test=test.reset_index(drop=True),
        )

    def _train_test_split(
        self,
        frame: pd.DataFrame,
        *,
        test_size: float,
        random_state: int,
        group_key: Any | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        stratify = None
        if self.stratify_column is not None:
            stratify = frame[self.stratify_column]

        try:
            return train_test_split(
                frame,
                test_size=test_size,
                random_state=random_state,
                shuffle=True,
                stratify=stratify,
            )
        except ValueError as error:
            location = (
                "global split"
                if group_key is None
                else f"group {self.group_column}={group_key!r}"
            )
            raise ValueError(f"Cannot split {location}: {error}") from error

    @staticmethod
    def _empty_like(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.iloc[0:0].copy()

    @staticmethod
    def _concat_parts(
        parts: list[pd.DataFrame],
        template: pd.DataFrame,
    ) -> pd.DataFrame:
        if not parts:
            return template.iloc[0:0].copy().reset_index(drop=True)
        return pd.concat(parts, ignore_index=True)
