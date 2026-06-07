from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import pandas as pd


class Scorer(ABC):
    """Base contract for models that assign one numeric score per task row."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Fit scorer state from train inputs and targets."""
        ...

    @abstractmethod
    def score(self, X: pd.DataFrame) -> pd.Series:
        """Return one score per input row."""
        ...
