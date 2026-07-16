"""Per-user 80/20 splits with guaranteed size parity across split types.

The memorization / split-sensitivity experiment (revisiting Di Palma et al. 2025)
compares two ways of holding out each user's interactions:

- ``file_order``: the last ``n_test`` rows in *ratings-file order* are the test
  set (equivalently, the first ``n_train`` rows are train). This reproduces the
  released LLM-MemoryInspector protocol, which splits by row order without a
  per-user permutation.
- ``random``: a seeded per-user permutation, then the same ``n_train`` / ``n_test``
  counts.

The single load-bearing property of this experiment is **size parity**: for every
user the pair ``(n_train, n_test)`` is *identical* between the two splits, so the
only thing that changes is *which* items are held out, never how many. We
therefore compute the cut ``n_train`` exactly once per user and reuse it for both
orderings, then assert parity as a sanity check.

Rounding rule: ``n_train = floor(train_fraction * n)``, clamped so both train and
test have at least one row. This is a documented convention; the exact rounding
used by Di Palma et al. is not verified here, but it does not affect the relative
comparison because both arms use the same ``n_train``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

import numpy as np
import pandas as pd


TRAIN_FRACTION = 0.8


def _user_seed(base_seed: int, user_id: object) -> int:
    """Stable per-user RNG seed that does not depend on evaluation order."""
    try:
        user_int = int(user_id)
    except (TypeError, ValueError):
        user_int = int.from_bytes(str(user_id).encode("utf-8"), "little", signed=False)
    return (base_seed * 1_000_003 + user_int) % (2**32)


def _n_train(n: int, train_fraction: float) -> int:
    """Number of train rows for a user with ``n`` interactions (both sides non-empty)."""
    n_train = floor(train_fraction * n)
    n_train = max(1, min(n_train, n - 1))
    return n_train


@dataclass(frozen=True)
class SplitResult:
    """Two per-user splits sharing identical per-user train/test sizes."""

    file_order: pd.DataFrame  # interactions + 'split' column in {'train','test'}
    random: pd.DataFrame
    sizes: pd.DataFrame  # per-user [user_id, n, n_train, n_test]


def make_parity_splits(
    interactions: pd.DataFrame,
    *,
    train_fraction: float = TRAIN_FRACTION,
    seed: int = 0,
    user_column: str = "user_id",
) -> SplitResult:
    """Build file-order and random per-user splits with identical per-user sizes.

    ``interactions`` must already be in ratings-file order (the order in which
    rows appear in ratings.dat). The canonical parquet produced by
    ``MovieLens1MAdapter`` satisfies this (row i == ``ml-1m:row:{i+1}``).

    Returns full interaction frames with an added ``split`` column for each
    ordering, plus a per-user size table.
    """
    if interactions.empty:
        raise ValueError("interactions is empty")

    work = interactions.reset_index(drop=True)
    work["_file_pos"] = np.arange(len(work))

    file_split = np.empty(len(work), dtype=object)
    rand_split = np.empty(len(work), dtype=object)
    size_rows: list[dict[str, object]] = []

    # groupby(sort=False) preserves the (contiguous) file order of each user block.
    for user_id, group in work.groupby(user_column, sort=False):
        positions = group["_file_pos"].to_numpy()  # already in file order
        n = len(positions)
        n_train = _n_train(n, train_fraction)
        n_test = n - n_train

        # file-order: first n_train rows -> train, remaining -> test
        file_split[positions[:n_train]] = "train"
        file_split[positions[n_train:]] = "test"

        # random: seeded per-user permutation, SAME n_train / n_test
        rng = np.random.default_rng(_user_seed(seed, user_id))
        perm = rng.permutation(n)
        rand_split[positions[perm[:n_train]]] = "train"
        rand_split[positions[perm[n_train:]]] = "test"

        size_rows.append(
            {user_column: user_id, "n": n, "n_train": n_train, "n_test": n_test}
        )

    file_frame = interactions.reset_index(drop=True).copy()
    file_frame["split"] = file_split
    random_frame = interactions.reset_index(drop=True).copy()
    random_frame["split"] = rand_split

    sizes = pd.DataFrame(size_rows)

    _assert_parity(file_frame, random_frame, sizes, user_column=user_column)

    return SplitResult(file_order=file_frame, random=random_frame, sizes=sizes)


def _assert_parity(
    file_frame: pd.DataFrame,
    random_frame: pd.DataFrame,
    sizes: pd.DataFrame,
    *,
    user_column: str,
) -> None:
    """Hard check that per-user train/test sizes match across the two splits."""
    for name, frame in (("file_order", file_frame), ("random", random_frame)):
        counts = (
            frame.groupby([user_column, "split"], sort=False)
            .size()
            .unstack(fill_value=0)
        )
        merged = sizes.set_index(user_column).join(counts)
        if not (merged["train"] == merged["n_train"]).all():
            raise AssertionError(f"{name}: per-user train size mismatch")
        if not (merged["test"] == merged["n_test"]).all():
            raise AssertionError(f"{name}: per-user test size mismatch")
