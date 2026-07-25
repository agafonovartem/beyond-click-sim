from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

import pandas as pd

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks._sampling import stable_sample_values
from beyond_click_sim.tasks.base import (
    CandidateSampler,
    DatasetFilter,
    ItemFeatureBuilder,
    Task,
    TaskBuilder,
    TaskSchema,
)
from beyond_click_sim.tasks.item_enrichment import item_enrichment_manifest


@dataclass(frozen=True)
class ColdStartSplitFrames:
    """User-partitioned cold-start split with temporal online session history.

    `val` and `test` contain raw post-k positive interaction rows — no negatives.
    Candidate sampling is the task builder's responsibility.
    `online_session_history` contains the first k temporal interactions per cold
    user; these rows are not in train, val, or test.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    online_session_history: pd.DataFrame
    warm_user_ids: frozenset
    val_cold_user_ids: frozenset
    test_cold_user_ids: frozenset
    dropped_val_cold_count: int = 0
    dropped_test_cold_count: int = 0


class ColdUserHoldoutSplitter:
    """Partition users into warm and cold sets; temporally split cold users at position k.

    No cold user or any of their interactions appears in train. The first k
    temporal interactions per cold user (sorted by timestamp, ties broken by
    item_id) form `online_session_history`. Post-k interactions are the
    evaluation target in val/test. Cold users with zero post-k interactions are
    dropped from evaluation entirely.

    Requires a timestamp column — raises ValueError for datasets without one
    (e.g. Steam, which has no wall-clock timestamps in iteration 1).
    """

    def __init__(
        self,
        k: int,
        train_fraction: float = 0.7,
        val_fraction: float = 0.1,
        test_fraction: float = 0.2,
        seed: int = 0,
        timestamp_column: str = "timestamp",
        user_column: str = "user_id",
        item_column: str = "item_id",
    ) -> None:
        if abs(train_fraction + val_fraction + test_fraction - 1.0) > 1e-9:
            raise ValueError(
                "train_fraction + val_fraction + test_fraction must sum to 1.0, "
                f"got {train_fraction + val_fraction + test_fraction}."
            )
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}.")
        self.k = k
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.timestamp_column = timestamp_column
        self.user_column = user_column
        self.item_column = item_column

    def split(self, interactions: pd.DataFrame) -> ColdStartSplitFrames:
        if self.timestamp_column not in interactions.columns:
            raise ValueError(
                f"ColdUserHoldoutSplitter requires a timestamp column "
                f"({self.timestamp_column!r}) for temporal online_session_history "
                f"ordering. Steam interactions have no timestamps; this splitter "
                f"is not supported for Steam in iteration 1."
            )

        # Step 1: partition users (random, seed-controlled)
        all_users = sorted(interactions[self.user_column].unique(), key=repr)
        n_users = len(all_users)
        n_cold = round((self.val_fraction + self.test_fraction) * n_users)
        cold_users = stable_sample_values(all_users, n=n_cold, seed=self.seed)

        n_val_cold = round(
            self.val_fraction / (self.val_fraction + self.test_fraction) * n_cold
        )
        val_cold_users = stable_sample_values(
            cold_users, n=n_val_cold, seed=self.seed + 1
        )
        val_cold_set = set(val_cold_users)
        cold_set = set(cold_users)
        test_cold_set = cold_set - val_cold_set
        warm_set = set(all_users) - cold_set

        train = (
            interactions[interactions[self.user_column].isin(warm_set)]
            .copy()
            .reset_index(drop=True)
        )

        # Step 2: temporal split within each cold user (mandatory, no random ordering)
        cold_interactions = (
            interactions[interactions[self.user_column].isin(cold_set)]
            .copy()
            .sort_values(
                [self.user_column, self.timestamp_column, self.item_column],
                ascending=True,
            )
        )
        rank = cold_interactions.groupby(self.user_column, sort=False).cumcount()

        history_mask = rank < self.k
        online_session_history = (
            cold_interactions[history_mask].copy().reset_index(drop=True)
        )

        post_k = cold_interactions[~history_mask].copy()
        val_mask = post_k[self.user_column].isin(val_cold_set)
        val_raw = post_k[val_mask].reset_index(drop=True)
        test_raw = post_k[~val_mask].reset_index(drop=True)

        # Drop cold users with zero post-k rows; record counts for the manifest
        val_users_with_postk = set(val_raw[self.user_column].unique())
        test_users_with_postk = set(test_raw[self.user_column].unique())
        dropped_val = len(val_cold_set - val_users_with_postk)
        dropped_test = len(test_cold_set - test_users_with_postk)

        active_cold = val_users_with_postk | test_users_with_postk
        online_session_history = (
            online_session_history[
                online_session_history[self.user_column].isin(active_cold)
            ]
            .copy()
            .reset_index(drop=True)
        )

        return ColdStartSplitFrames(
            train=train,
            val=val_raw,
            test=test_raw,
            online_session_history=online_session_history,
            warm_user_ids=frozenset(warm_set),
            val_cold_user_ids=frozenset(val_users_with_postk),
            test_cold_user_ids=frozenset(test_users_with_postk),
            dropped_val_cold_count=dropped_val,
            dropped_test_cold_count=dropped_test,
        )


def select_cold_start_targets(
    post_k: pd.DataFrame,
    *,
    target_source_column: str,
    targets_per_user: int,
    user_column: str = "user_id",
    item_column: str = "item_id",
) -> tuple[pd.DataFrame, int]:
    """Select the fixed evaluation target frame from raw post-k interactions.

    For each user, keeps the first `targets_per_user` rows with
    `target_source_column == 1` in the input (temporal) order. Rows that fail
    the target condition are excluded from the frame entirely — they are
    neither targets nor negative candidates. Users with zero qualifying rows
    are dropped here and counted separately from the splitter's own drops.

    Returns `(frame, n_dropped_zero_qualifying)` where frame has columns
    `[user_column, item_column]`. The frame depends only on its inputs — it
    never sees the negative ratio, the negative strategy, or the sampling
    seed — so tasks sharing `(dataset, k, split_seed)` share an identical
    frame by construction.
    """

    if targets_per_user < 1:
        raise ValueError(
            f"targets_per_user must be positive, got {targets_per_user}."
        )
    missing = [
        column
        for column in (user_column, item_column, target_source_column)
        if column not in post_k.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    output_columns = [user_column, item_column]
    if post_k.empty:
        return pd.DataFrame(columns=output_columns), 0

    qualifying = post_k[post_k[target_source_column].astype(bool)]
    frame = (
        qualifying.groupby(user_column, sort=False)
        .head(targets_per_user)[output_columns]
        .reset_index(drop=True)
    )
    users_with_targets = set(frame[user_column].unique())
    n_dropped = len(set(post_k[user_column].unique()) - users_with_targets)
    return frame, n_dropped


def target_frame_sha256(
    frame: pd.DataFrame,
    *,
    user_column: str = "user_id",
    item_column: str = "item_id",
) -> str:
    """Content hash of a target frame — the cross-task frame-identity check."""

    pairs = sorted(
        (repr(user_id), repr(item_id))
        for user_id, item_id in zip(
            frame[user_column], frame[item_column], strict=True
        )
    )
    return hashlib.sha256(repr(pairs).encode("utf-8")).hexdigest()


def _targets_per_user_distribution(
    frame: pd.DataFrame, *, user_column: str
) -> dict[str, float | int]:
    if frame.empty:
        return {"min": 0, "mean": 0.0, "max": 0}
    counts = frame.groupby(user_column, sort=False).size()
    return {
        "min": int(counts.min()),
        "mean": float(counts.mean()),
        "max": int(counts.max()),
    }


@dataclass
class ColdStartTask(Task):
    """Task variant that carries cold users' online session history.

    `online_session_history` holds the first k temporal interactions per cold
    user, enriched with user/item features and history context columns — same
    column schema as `train`. Pass it to the LLM scorer's `fit()` instead of
    `train`; pass `train` to Popularity/ItemKNN scorers as usual.
    """

    online_session_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    k: int = 0


class ColdStartTaskBuilder:
    """Build a cold-start candidate-set task over a fixed target frame.

    Cold users are entirely withheld from training. Their first k temporal
    interactions form `task.online_session_history` (for LLM/ItemKNN context).

    The evaluation target frame is built before any negative sampling: per
    split, the first `targets_per_user` qualifying (`target_source_column == 1`)
    post-k items per user, in temporal order. The frame depends only on
    `(dataset, k, split_seed)` — the sampler receives it as given, so the
    negative ratio, negative strategy, and sampling seed cannot shift the
    evaluated users or targets. `target_frame_sha256` in the manifest makes
    frame identity across tasks checkable without rebuilding them.

    Evaluated users are held fixed across k as well: the eval-user pool is
    selected from users eligible at `pool_k` (>= 1 qualifying target in the
    post-`pool_k` eval-pool). Eligibility at `pool_k` implies eligibility at
    every smaller k because eval-pools are nested. The selection itself uses
    `split_seed`, never the sampling seed.

    Strict warm-item coverage is enforced: history profiles, evaluation
    targets, and sampled negatives may only use items that appear in warm
    `train`. Users whose k-item history contains any cold-only item are
    dropped; post-k rows on cold-only items are excluded before target
    selection; the negative sampler receives the warm-item catalog only.

    Item features are enriched from warm `task.train` interactions only, so
    item popularity statistics remain uncontaminated by cold user activity.
    """

    def __init__(
        self,
        name: str,
        dataset_filter: DatasetFilter,
        splitter: ColdUserHoldoutSplitter,
        sampler: CandidateSampler,
        *,
        targets_per_user: int,
        target_source_column: str = "target_like_ge4",
        max_eval_users: int | None = None,
        pool_k: int,
        split_seed: int,
        target_column: str = "target",
        user_column: str = "user_id",
        item_column: str = "item_id",
        sampled_column: str = "sampled",
        candidate_group_column: str = "candidate_group",
        history_context_columns: tuple[str, ...] = (),
        item_feature_builder: ItemFeatureBuilder | None = None,
    ) -> None:
        if pool_k < splitter.k:
            raise ValueError(
                f"pool_k ({pool_k}) must be >= splitter k ({splitter.k}); "
                "eligibility is defined at the largest k so the same users are "
                "eligible at every smaller k."
            )
        self.name = name
        self.dataset_filter = dataset_filter
        self.splitter = splitter
        self.sampler = sampler
        self.targets_per_user = targets_per_user
        self.target_source_column = target_source_column
        self.max_eval_users = max_eval_users
        self.pool_k = pool_k
        self.split_seed = split_seed
        self.target_column = target_column
        self.user_column = user_column
        self.item_column = item_column
        self.sampled_column = sampled_column
        self.candidate_group_column = candidate_group_column
        self.history_context_columns = history_context_columns
        self.item_feature_builder = item_feature_builder

    def build(self, dataset: CanonicalDataset) -> ColdStartTask:
        canonical_manifest = dataset.load_manifest()
        users = dataset.load_users()
        items = dataset.load_items()
        interactions = dataset.load_interactions()
        task_item_enrichment = item_enrichment_manifest(canonical_manifest, items)

        users, items, interactions = self.dataset_filter.filter(
            users=users, items=items, interactions=interactions
        )
        split = self.splitter.split(interactions)
        warm_item_ids = set(split.train[self.item_column].unique())

        history_warm_users = self._users_with_fully_warm_history(
            split.online_session_history,
            warm_item_ids,
            k=self.splitter.k,
        )
        dropped_nonwarm_history_val = int(
            len(split.val_cold_user_ids - history_warm_users)
        )
        dropped_nonwarm_history_test = int(
            len(split.test_cold_user_ids - history_warm_users)
        )

        # Eval-user pool: selected once from users eligible at pool_k, so the
        # same users are evaluated for every k, m, strategy, and sampling seed.
        # Eligibility requires warm history and at least one warm qualifying
        # target at pool_k.
        selected_val_users, selected_test_users = self._select_eval_users(
            interactions,
            split,
            warm_item_ids=warm_item_ids,
            history_warm_users=history_warm_users,
        )
        val_post_k_raw = split.val[
            split.val[self.user_column].isin(selected_val_users)
        ]
        test_post_k_raw = split.test[
            split.test[self.user_column].isin(selected_test_users)
        ]
        selected_users = selected_val_users | selected_test_users
        online_history = split.online_session_history[
            split.online_session_history[self.user_column].isin(selected_users)
        ]

        dropped_nonwarm_post_k_val = int(
            (~val_post_k_raw[self.item_column].isin(warm_item_ids)).sum()
        )
        dropped_nonwarm_post_k_test = int(
            (~test_post_k_raw[self.item_column].isin(warm_item_ids)).sum()
        )
        val_post_k = val_post_k_raw[
            val_post_k_raw[self.item_column].isin(warm_item_ids)
        ]
        test_post_k = test_post_k_raw[
            test_post_k_raw[self.item_column].isin(warm_item_ids)
        ]

        # Fixed target frames — built before and independently of negatives.
        val_frame, dropped_val_zero_qualifying = select_cold_start_targets(
            val_post_k,
            target_source_column=self.target_source_column,
            targets_per_user=self.targets_per_user,
            user_column=self.user_column,
            item_column=self.item_column,
        )
        test_frame, dropped_test_zero_qualifying = select_cold_start_targets(
            test_post_k,
            target_source_column=self.target_source_column,
            targets_per_user=self.targets_per_user,
            user_column=self.user_column,
            item_column=self.item_column,
        )

        # All cold interactions are the observed exclusion set for negative sampling
        # so profile items and post-k positives are never drawn as negatives.
        all_cold_interactions = pd.concat(
            [split.online_session_history, split.val, split.test],
            ignore_index=True,
        )
        warm_items = items[items[self.item_column].isin(warm_item_ids)].copy()

        val_rows = self.sampler.sample(
            val_frame,
            interactions=all_cold_interactions,
            items=warm_items,
            train_interactions=split.train,
        )
        val_sampler_diagnostics = dict(getattr(self.sampler, "diagnostics_", {}))
        val_negative_pairs = self._sampled_pairs(val_rows)
        test_rows = self.sampler.sample(
            test_frame,
            interactions=all_cold_interactions,
            items=warm_items,
            train_interactions=split.train,
            excluded_pairs=val_negative_pairs,
        )
        test_sampler_diagnostics = dict(getattr(self.sampler, "diagnostics_", {}))

        items, item_feature_manifest = self._enrich_items_from_train(
            items=items, train_interactions=split.train
        )
        user_features = TaskBuilder._prefixed_features(users, self.user_column, "user")
        item_features = TaskBuilder._prefixed_features(items, self.item_column, "item")
        feature_columns = TaskBuilder._feature_columns(
            user_features,
            item_features,
            user_column=self.user_column,
            item_column=self.item_column,
        )
        task_columns = self._task_columns(feature_columns)

        train = TaskBuilder._with_features(
            rows=self._positive_rows(split.train, candidate_group=pd.NA),
            user_features=user_features,
            item_features=item_features,
            columns=task_columns,
            user_column=self.user_column,
            item_column=self.item_column,
        )
        val = TaskBuilder._with_features(
            rows=self._without_history_context(val_rows),
            user_features=user_features,
            item_features=item_features,
            columns=task_columns,
            user_column=self.user_column,
            item_column=self.item_column,
        )
        test = TaskBuilder._with_features(
            rows=self._without_history_context(test_rows),
            user_features=user_features,
            item_features=item_features,
            columns=task_columns,
            user_column=self.user_column,
            item_column=self.item_column,
        )
        # online_session_history shares the same column schema as train so
        # scorers can call fit() on either frame interchangeably.
        online_session_history = TaskBuilder._with_features(
            rows=self._positive_rows(online_history, candidate_group=pd.NA),
            user_features=user_features,
            item_features=item_features,
            columns=task_columns,
            user_column=self.user_column,
            item_column=self.item_column,
        )

        schema = TaskSchema(
            target_column=self.target_column,
            feature_columns=feature_columns,
            id_columns=(self.user_column, self.item_column),
            candidate_group_column=self.candidate_group_column,
            sampled_column=self.sampled_column,
            history_context_columns=self.history_context_columns,
        )

        return ColdStartTask(
            name=self.name,
            train=train,
            val=val,
            test=test,
            online_session_history=online_session_history,
            k=self.splitter.k,
            schema=schema,
            manifest={
                "dataset": dataset.name,
                "dataset_version": dataset.version,
                "target_source_column": self.target_source_column,
                "target_column": self.target_column,
                "targets_per_user": self.targets_per_user,
                "split_seed": self.split_seed,
                "pool_k": self.pool_k,
                "sampling_seed": self.sampler.seed,
                "negative_strategy": getattr(self.sampler, "strategy", None),
                "warm_item_coverage": {
                    "enforced": True,
                    "policy": "strict",
                    "warm_items": int(len(warm_item_ids)),
                    "dropped_users_nonwarm_history_val": (
                        dropped_nonwarm_history_val
                    ),
                    "dropped_users_nonwarm_history_test": (
                        dropped_nonwarm_history_test
                    ),
                    "dropped_post_k_rows_nonwarm_val": dropped_nonwarm_post_k_val,
                    "dropped_post_k_rows_nonwarm_test": (
                        dropped_nonwarm_post_k_test
                    ),
                },
                "target_frame_sha256": {
                    "val": target_frame_sha256(
                        val_frame, user_column=self.user_column,
                        item_column=self.item_column,
                    ),
                    "test": target_frame_sha256(
                        test_frame, user_column=self.user_column,
                        item_column=self.item_column,
                    ),
                },
                "feature_columns": list(feature_columns),
                "history_context_columns": list(self.history_context_columns),
                "item_enrichment": task_item_enrichment,
                "item_feature_builder": item_feature_manifest,
                "sampled_column": self.sampled_column,
                "candidate_group_column": self.candidate_group_column,
                "k": self.splitter.k,
                "filter": TaskBuilder._component_manifest(self.dataset_filter),
                "splitter": TaskBuilder._component_manifest(self.splitter),
                "sampler": TaskBuilder._component_manifest(self.sampler),
                "sampler_diagnostics": {
                    "val": val_sampler_diagnostics,
                    "test": test_sampler_diagnostics,
                },
                "rows": {
                    "train": int(len(train)),
                    "val": int(len(val)),
                    "test": int(len(test)),
                    "online_session_history": int(len(online_session_history)),
                },
                "users": {
                    "warm": int(len(split.warm_user_ids)),
                    "val_cold": int(len(split.val_cold_user_ids)),
                    "test_cold": int(len(split.test_cold_user_ids)),
                    "evaluated_val": int(len(selected_val_users)),
                    "evaluated_test": int(len(selected_test_users)),
                    "dropped_val_cold": split.dropped_val_cold_count,
                    "dropped_test_cold": split.dropped_test_cold_count,
                    "dropped_zero_qualifying_targets_val": int(
                        dropped_val_zero_qualifying
                    ),
                    "dropped_zero_qualifying_targets_test": int(
                        dropped_test_zero_qualifying
                    ),
                    "dropped_nonwarm_history_val": dropped_nonwarm_history_val,
                    "dropped_nonwarm_history_test": dropped_nonwarm_history_test,
                },
                "targets_per_user_distribution": {
                    "val": _targets_per_user_distribution(
                        val_frame, user_column=self.user_column
                    ),
                    "test": _targets_per_user_distribution(
                        test_frame, user_column=self.user_column
                    ),
                },
                "items": int(items[self.item_column].nunique()),
            },
        )

    def _select_eval_users(
        self,
        interactions: pd.DataFrame,
        split: ColdStartSplitFrames,
        *,
        warm_item_ids: set[Any],
        history_warm_users: set[Any],
    ) -> tuple[set[Any], set[Any]]:
        """Select eval users from the pool_k eligibility pool (split_seed only).

        A user is eligible if:
        - their pool_k history items are all present in warm train, and
        - their post-pool_k eval-pool contains at least one qualifying
          (`target_source_column == 1`) interaction on a warm item.

        Because eval-pools are nested in k, users eligible at pool_k are
        eligible at every k <= pool_k, which keeps the evaluated-user set
        identical across the k-grid. Selection uses `split_seed` only.
        """

        val_with_warm_history = split.val_cold_user_ids & history_warm_users
        test_with_warm_history = split.test_cold_user_ids & history_warm_users
        if self.max_eval_users is None:
            return val_with_warm_history, test_with_warm_history

        pool_splitter = ColdUserHoldoutSplitter(
            k=self.pool_k,
            train_fraction=self.splitter.train_fraction,
            val_fraction=self.splitter.val_fraction,
            test_fraction=self.splitter.test_fraction,
            seed=self.splitter.seed,
            timestamp_column=self.splitter.timestamp_column,
            user_column=self.user_column,
            item_column=self.item_column,
        )
        pool_split = pool_splitter.split(interactions)
        pool_warm_item_ids = set(pool_split.train[self.item_column].unique())
        pool_history_warm_users = self._users_with_fully_warm_history(
            pool_split.online_session_history,
            pool_warm_item_ids,
            k=self.pool_k,
        )
        eligible_val = (
            self._eligible_users(pool_split.val, warm_item_ids=pool_warm_item_ids)
            & pool_history_warm_users
        )
        eligible_test = (
            self._eligible_users(pool_split.test, warm_item_ids=pool_warm_item_ids)
            & pool_history_warm_users
        )
        selected_val = set(
            stable_sample_values(
                sorted(eligible_val, key=repr),
                n=self.max_eval_users,
                seed=self.split_seed,
            )
        )
        # Val and test pools are drawn independently but with different seeds so
        # the val selection cannot shift the test selection (val size varies
        # with the eligible pool).
        selected_test = set(
            stable_sample_values(
                sorted(eligible_test, key=repr),
                n=self.max_eval_users,
                seed=self.split_seed + 1,
            )
        )
        return selected_val, selected_test

    def _users_with_fully_warm_history(
        self,
        history: pd.DataFrame,
        warm_item_ids: set[Any],
        *,
        k: int,
    ) -> set[Any]:
        """Return users whose history has exactly k rows, all on warm items."""

        if history.empty or k < 1:
            return set()
        warm_users: set[Any] = set()
        for user_id, group in history.groupby(self.user_column, sort=False):
            if len(group) != k:
                continue
            if set(group[self.item_column]).issubset(warm_item_ids):
                warm_users.add(user_id)
        return warm_users

    def _eligible_users(
        self,
        post_k: pd.DataFrame,
        *,
        warm_item_ids: set[Any] | None = None,
    ) -> set[Any]:
        frame = post_k
        if warm_item_ids is not None:
            frame = frame[frame[self.item_column].isin(warm_item_ids)]
        if frame.empty:
            return set()
        qualifying = frame[frame[self.target_source_column].astype(bool)]
        return set(qualifying[self.user_column].unique())

    def _positive_rows(
        self,
        interactions: pd.DataFrame,
        *,
        candidate_group: Any,
    ) -> pd.DataFrame:
        rows = interactions[
            [self.user_column, self.item_column, *self.history_context_columns]
        ].copy()
        rows[self.target_column] = interactions[self.target_source_column]
        rows[self.sampled_column] = False
        rows[self.candidate_group_column] = candidate_group
        return rows

    def _without_history_context(self, rows: pd.DataFrame) -> pd.DataFrame:
        rows = rows.copy()
        for column in self.history_context_columns:
            rows[column] = pd.NA
        return rows

    def _sampled_pairs(self, rows: pd.DataFrame) -> set[tuple[Any, Any]]:
        if rows.empty:
            return set()
        sampled_rows = rows[rows[self.sampled_column].astype(bool)]
        return set(
            zip(
                sampled_rows[self.user_column],
                sampled_rows[self.item_column],
                strict=True,
            )
        )

    def _task_columns(self, feature_columns: tuple[str, ...]) -> list[str]:
        return [
            self.user_column,
            self.item_column,
            *feature_columns,
            *self.history_context_columns,
            self.target_column,
            self.sampled_column,
            self.candidate_group_column,
        ]

    def _enrich_items_from_train(
        self,
        *,
        items: pd.DataFrame,
        train_interactions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any] | None]:
        if self.item_feature_builder is None:
            return items, None
        return self.item_feature_builder.enrich_items(
            items=items,
            train_interactions=train_interactions,
            item_column=self.item_column,
        )
