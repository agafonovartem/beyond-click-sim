from __future__ import annotations

from collections.abc import Iterable
from math import comb
from typing import Literal

import numpy as np
import pandas as pd


RankingTiePolicy = Literal["average"]


def grouped_ranking_metrics(
    y_true: pd.Series,
    scores: pd.Series,
    groups: pd.Series,
    *,
    ks: Iterable[int] = (1, 3, 5, 10),
    tie_policy: RankingTiePolicy = "average",
) -> dict[str, float | int | str]:
    """Compute ranking metrics per candidate group, then average groups equally."""

    normalized_ks = _validate_inputs(
        y_true,
        scores,
        groups,
        ks=ks,
        tie_policy=tie_policy,
    )
    group_codes = _factorize_group_keys(pd.MultiIndex.from_arrays([groups]))
    per_group = _per_group_ranking_metrics(
        y_true=y_true,
        scores=scores,
        group_codes=group_codes,
        ks=normalized_ks,
    )
    return _aggregate_per_group_metrics(
        per_group,
        ks=normalized_ks,
        tie_policy=tie_policy,
    )


def user_grouped_ranking_metrics(
    y_true: pd.Series,
    scores: pd.Series,
    groups: pd.Series,
    users: pd.Series,
    *,
    ks: Iterable[int] = (1, 3, 5, 10),
    tie_policy: RankingTiePolicy = "average",
) -> dict[str, float | int | str]:
    """Compute per-group ranking metrics, average groups per user, then average users."""

    normalized_ks = _validate_inputs(
        y_true,
        scores,
        groups,
        users,
        ks=ks,
        tie_policy=tie_policy,
    )
    group_codes = _factorize_group_keys(pd.MultiIndex.from_arrays([users, groups]))
    user_codes, _ = pd.factorize(users, sort=False)
    if (user_codes < 0).any():
        raise ValueError("groups and users must not contain NA values")
    n_groups = int(group_codes.max()) + 1
    group_user_codes = np.empty(n_groups, dtype=int)
    group_user_codes[group_codes] = user_codes
    per_group = _per_group_ranking_metrics(
        y_true=y_true,
        scores=scores,
        group_codes=group_codes,
        ks=normalized_ks,
    )
    return _aggregate_per_group_metrics(
        per_group,
        ks=normalized_ks,
        tie_policy=tie_policy,
        group_user_codes=group_user_codes,
    )


def _per_group_ranking_metrics(
    *,
    y_true: pd.Series,
    scores: pd.Series,
    group_codes: np.ndarray,
    ks: tuple[int, ...],
) -> pd.DataFrame:
    true_values = y_true.astype(bool).to_numpy()
    score_values = scores.astype(float).to_numpy()

    records: list[dict[str, float | int | bool]] = []
    for group_code in range(int(group_codes.max()) + 1):
        positions = np.flatnonzero(group_codes == group_code)
        group_true = true_values[positions]
        group_scores = score_values[positions]
        values, has_ties = _ranking_values_for_group(group_true, group_scores, ks=ks)
        records.append(
            {
                "group_code": group_code,
                "n": int(len(group_true)),
                "n_positive": int(group_true.sum()),
                "has_score_ties": has_ties,
                **values,
            }
        )

    return pd.DataFrame.from_records(records)


def _ranking_values_for_group(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    ks: tuple[int, ...],
) -> tuple[dict[str, float], bool]:
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_true = y_true[order].astype(int)
    sorted_scores = scores[order]
    group_size = len(sorted_true)
    total_positive = int(sorted_true.sum())
    has_ties = bool(len(np.unique(sorted_scores)) < group_size)

    block_starts = np.flatnonzero(
        np.concatenate([[True], sorted_scores[1:] != sorted_scores[:-1]])
    )
    block_ends = np.concatenate([block_starts[1:], [group_size]])
    discounts = 1.0 / np.log2(np.arange(2, group_size + 2))

    values: dict[str, float] = {}
    for k in ks:
        k_eff = min(k, group_size)
        values[f"hit_rate@{k}"] = _expected_hit_rate_at_k(
            sorted_true=sorted_true,
            block_starts=block_starts,
            block_ends=block_ends,
            k=k_eff,
        )
        values[f"ndcg@{k}"] = _expected_ndcg_at_k(
            sorted_true=sorted_true,
            block_starts=block_starts,
            block_ends=block_ends,
            discounts=discounts,
            total_positive=total_positive,
            k=k_eff,
        )

    return values, has_ties


def _expected_hit_rate_at_k(
    *,
    sorted_true: np.ndarray,
    block_starts: np.ndarray,
    block_ends: np.ndarray,
    k: int,
) -> float:
    if int(sorted_true.sum()) == 0 or k <= 0:
        return 0.0
    if k >= len(sorted_true):
        return 1.0

    for start, end in zip(block_starts, block_ends, strict=True):
        block_positive = int(sorted_true[start:end].sum())
        if end <= k:
            if block_positive > 0:
                return 1.0
            continue
        if start >= k:
            return 0.0

        selected_from_block = k - start
        return _probability_at_least_one_positive(
            total=end - start,
            positives=block_positive,
            selected=selected_from_block,
        )

    return 0.0


def _expected_ndcg_at_k(
    *,
    sorted_true: np.ndarray,
    block_starts: np.ndarray,
    block_ends: np.ndarray,
    discounts: np.ndarray,
    total_positive: int,
    k: int,
) -> float:
    if total_positive == 0 or k <= 0:
        return 0.0

    expected_dcg = 0.0
    for start, end in zip(block_starts, block_ends, strict=True):
        if start >= k:
            break
        selected_end = min(end, k)
        block_size = end - start
        block_positive = int(sorted_true[start:end].sum())
        expected_relevance = block_positive / block_size
        expected_dcg += float(expected_relevance * discounts[start:selected_end].sum())

    ideal_positive = min(total_positive, k)
    ideal_dcg = float(discounts[:ideal_positive].sum())
    return expected_dcg / ideal_dcg


def _probability_at_least_one_positive(*, total: int, positives: int, selected: int) -> float:
    if positives <= 0 or selected <= 0:
        return 0.0
    if selected >= total or selected > total - positives:
        return 1.0
    return 1.0 - (comb(total - positives, selected) / comb(total, selected))


def _aggregate_per_group_metrics(
    per_group: pd.DataFrame,
    *,
    ks: tuple[int, ...],
    tie_policy: str,
    group_user_codes: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    metric_keys = [f"{name}@{k}" for k in ks for name in ("hit_rate", "ndcg")]
    if group_user_codes is None:
        metrics = {key: float(per_group[key].mean()) for key in metric_keys}
        n_users = None
    else:
        n_users = int(group_user_codes.max()) + 1
        groups_per_user = np.bincount(group_user_codes, minlength=n_users)
        metrics = {}
        for key in metric_keys:
            user_sums = np.bincount(
                group_user_codes,
                weights=per_group[key].to_numpy(dtype=float),
                minlength=n_users,
            )
            metrics[key] = float((user_sums / groups_per_user).mean())

    n_groups = int(len(per_group))
    diagnostics: dict[str, float | int | str] = {
        "n_groups": n_groups,
        "n": int(per_group["n"].sum()),
        "n_positive": int(per_group["n_positive"].sum()),
        "groups_without_positive": int(per_group["n_positive"].eq(0).sum()),
        "groups_with_score_ties": int(per_group["has_score_ties"].sum()),
        "groups_with_score_ties_fraction": float(per_group["has_score_ties"].mean()),
        "tie_policy": tie_policy,
    }
    for k in ks:
        diagnostics[f"groups_with_size_lte@{k}"] = int(per_group["n"].le(k).sum())
    if n_users is not None:
        diagnostics["n_users"] = n_users

    return {**metrics, **diagnostics}


def _factorize_group_keys(group_keys: pd.MultiIndex) -> np.ndarray:
    group_codes, _ = pd.factorize(group_keys, sort=False)
    if (group_codes < 0).any():
        raise ValueError("groups and users must not contain NA values")
    return group_codes


def _validate_inputs(
    y_true: pd.Series,
    scores: pd.Series,
    groups: pd.Series,
    users: pd.Series | None = None,
    *,
    ks: Iterable[int],
    tie_policy: str,
) -> tuple[int, ...]:
    _require_same_length(y_true, scores, left_name="y_true", right_name="scores")
    _require_same_length(y_true, groups, left_name="y_true", right_name="groups")
    if users is not None:
        _require_same_length(y_true, users, left_name="y_true", right_name="users")
    if len(y_true) == 0:
        raise ValueError("Cannot compute ranking metrics on empty inputs")
    if y_true.isna().any():
        raise ValueError("y_true contains NA values")
    if scores.isna().any():
        raise ValueError("scores contains NaN values")
    if groups.isna().any() or (users is not None and users.isna().any()):
        raise ValueError("groups and users must not contain NA values")
    if tie_policy != "average":
        raise ValueError(f"Unsupported tie_policy: {tie_policy!r}")

    normalized = tuple(dict.fromkeys(int(k) for k in ks))
    if not normalized:
        raise ValueError("ks must be non-empty")
    if any(k <= 0 for k in normalized):
        raise ValueError("ks must contain positive integers")
    return normalized


def _require_same_length(
    left: pd.Series,
    right: pd.Series,
    *,
    left_name: str,
    right_name: str,
) -> None:
    if len(left) != len(right):
        raise ValueError(f"{left_name} and {right_name} must have the same length")
