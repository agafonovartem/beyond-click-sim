from __future__ import annotations

from collections.abc import Iterable
import random
from typing import Any


def stable_sample_values(
    values: Iterable[Any],
    *,
    n: int | None,
    seed: int,
) -> list[Any]:
    """Sample unique values deterministically from an order-independent pool."""

    unique_values = sorted(set(values), key=repr)
    if n is None or len(unique_values) <= n:
        return unique_values
    if n < 1:
        raise ValueError("n must be positive when provided.")
    return random.Random(seed).sample(unique_values, n)
