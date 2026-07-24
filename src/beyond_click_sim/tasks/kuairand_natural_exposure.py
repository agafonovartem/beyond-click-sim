from __future__ import annotations

from typing import Any

import pandas as pd

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks.base import TaskSchema
from beyond_click_sim.tasks.kuairand_policy_shift import (
    POLICIES,
    SOURCE_END,
    SOURCE_START,
    TAB_VALUE,
    TEST_END,
    TEST_START,
    VAL_END,
    VAL_START,
    KuaiRandPolicyShiftTask,
    KuaiRandPolicyShiftTaskBuilder,
    _canonical_content_identity,
    _frame_sha256,
    _json_sha256,
    _stable_int,
    _values_sha256,
)


DEFAULT_HISTORY_SIZE = 5
DEFAULT_CANDIDATE_SIZE = 5
DEFAULT_MIN_EXPOSURE_USERS_PER_POLICY = 8
DEFAULT_MIN_TRAINING_EXPOSURE_USERS_PER_POLICY = 5
DEFAULT_CANDIDATE_SEED = 0


class KuaiRandNaturalExposureTaskBuilder:
    """Build the label-blind natural-exposure policy-shift task.

    The builder deliberately separates:

    - the H visible rows for each evaluated user;
    - the policy-specific collaborative training graph, which retains all
      source rows from non-evaluation users but only those same H rows from
      evaluation users;
    - m natural target exposures selected without consulting click labels.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        history_size: int = DEFAULT_HISTORY_SIZE,
        candidate_size: int = DEFAULT_CANDIDATE_SIZE,
        min_exposure_users_per_policy: int = (
            DEFAULT_MIN_EXPOSURE_USERS_PER_POLICY
        ),
        min_training_exposure_users_per_policy: int | None = None,
        candidate_seed: int = DEFAULT_CANDIDATE_SEED,
    ) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        if candidate_size <= 1:
            raise ValueError("candidate_size must be at least two")
        if min_exposure_users_per_policy <= 0:
            raise ValueError("min_exposure_users_per_policy must be positive")
        resolved_training_support = (
            min(
                DEFAULT_MIN_TRAINING_EXPOSURE_USERS_PER_POLICY,
                min_exposure_users_per_policy,
            )
            if min_training_exposure_users_per_policy is None
            else min_training_exposure_users_per_policy
        )
        if resolved_training_support <= 0:
            raise ValueError(
                "min_training_exposure_users_per_policy must be positive"
            )
        if resolved_training_support > min_exposure_users_per_policy:
            raise ValueError(
                "training exposure support cannot exceed full-source support"
            )
        self.history_size = history_size
        self.candidate_size = candidate_size
        self.min_exposure_users_per_policy = min_exposure_users_per_policy
        self.min_training_exposure_users_per_policy = (
            resolved_training_support
        )
        self.candidate_seed = candidate_seed
        self.name = name or (
            "kuairand_tab1_policy_shift_natural"
            f"_h{history_size}_m{candidate_size}"
            f"_expdeg{min_exposure_users_per_policy}"
            f"_trainexpdeg{resolved_training_support}"
            f"_candidate{candidate_seed}"
        )

    def build(self, dataset: CanonicalDataset) -> KuaiRandPolicyShiftTask:
        helper = KuaiRandPolicyShiftTaskBuilder(
            name=f"{self.name}_frame_helper",
            candidate_seed=self.candidate_seed,
        )
        canonical_manifest = dataset.load_manifest()
        users = dataset.load_users(columns=["user_id"])
        items = dataset.load_items(
            columns=["item_id", "description", "category_path"]
        )
        interactions = dataset.load_interactions(
            columns=[
                "interaction_id",
                "user_id",
                "item_id",
                "exposure_policy",
                "tab",
                "date",
                "timestamp_ms",
                "target_click",
            ]
        )
        helper._require_columns(users, ["user_id"], table="users")
        helper._validate_unique_items(items)
        prepared = helper._prepare_interactions(interactions)
        semantic_items, semantic_audit = helper._semantic_items(items)
        semantic_interactions = prepared.merge(
            semantic_items,
            on="item_id",
            how="inner",
            validate="many_to_one",
        )
        tab1 = semantic_interactions.loc[
            semantic_interactions["tab"].eq(TAB_VALUE)
        ].copy()

        source_by_policy = {
            policy: helper._first_pair_rows(
                tab1.loc[
                    tab1["_date_key"].between(SOURCE_START, SOURCE_END)
                    & tab1["_policy"].eq(policy)
                ].copy()
            )
            for policy in POLICIES
        }
        source_counts = {
            policy: frame.groupby("user_id", sort=False).size()
            for policy, frame in source_by_policy.items()
        }
        source_history_users = set(
            source_counts["standard"]
            .loc[source_counts["standard"].ge(self.history_size)]
            .index
        ) & set(
            source_counts["random"]
            .loc[source_counts["random"].ge(self.history_size)]
            .index
        )
        exposure_degrees = {
            policy: frame.groupby("item_id", sort=False)["user_id"].nunique()
            for policy, frame in source_by_policy.items()
        }
        positive_degrees = {
            policy: (
                frame.loc[frame["target_click"].eq(1)]
                .groupby("item_id", sort=False)["user_id"]
                .nunique()
            )
            for policy, frame in source_by_policy.items()
        }
        supported_items = {
            policy: set(
                degree.loc[
                    degree.ge(self.min_exposure_users_per_policy)
                ].index
            )
            for policy, degree in exposure_degrees.items()
        }
        common_supported_items = (
            supported_items["standard"] & supported_items["random"]
        )
        if not common_supported_items:
            raise ValueError(
                "No items meet the common exposure-support threshold"
            )

        target_frames: dict[tuple[str, str], pd.DataFrame] = {}
        target_audit: dict[str, Any] = {}
        evaluation_users: set[object] = set()
        for split, start, end in (
            ("val", VAL_START, VAL_END),
            ("test", TEST_START, TEST_END),
        ):
            pools, pool_audit = self._target_pools(
                helper=helper,
                tab1=tab1,
                all_interactions=prepared,
                start=start,
                end=end,
                allowed_items=common_supported_items,
            )
            target_users = self._target_users(
                pools,
                source_history_users=source_history_users,
            )
            if not target_users:
                raise ValueError(
                    f"No label-blind natural target users for split={split}"
                )
            evaluation_users.update(target_users)
            built = {
                policy: self._materialize_targets(
                    helper=helper,
                    rows=pools[policy],
                    users=target_users,
                    split=split,
                    policy=policy,
                )
                for policy in POLICIES
            }
            self._validate_targets(
                built,
                split=split,
                expected_users=target_users,
            )
            target_frames.update(
                {(split, policy): frame for policy, frame in built.items()}
            )
            target_audit[split] = {
                "window": [_date_iso(start), _date_iso(end)],
                "label_blind_users": len(target_users),
                "user_ids_sha256": _values_sha256(target_users),
                "pool": pool_audit,
                "policies": {
                    policy: _target_summary(
                        built[policy],
                        candidate_size=self.candidate_size,
                    )
                    for policy in POLICIES
                },
            }

        for frame in target_frames.values():
            for policy in POLICIES:
                frame[f"{policy}_source_exposure_user_degree"] = (
                    frame["item_id"]
                    .map(exposure_degrees[policy])
                    .fillna(0)
                    .astype(int)
                )
                frame[f"{policy}_source_positive_user_degree"] = (
                    frame["item_id"]
                    .map(positive_degrees[policy])
                    .fillna(0)
                    .astype(int)
                )
            frame["minimum_source_exposure_user_degree"] = frame[
                [
                    f"{policy}_source_exposure_user_degree"
                    for policy in POLICIES
                ]
            ].min(axis=1)
            frame["minimum_source_positive_user_degree"] = frame[
                [
                    f"{policy}_source_positive_user_degree"
                    for policy in POLICIES
                ]
            ].min(axis=1)
            frame["source_positive_support_stratum"] = (
                frame["minimum_source_positive_user_degree"]
                .map(_positive_support_stratum)
                .astype("string")
            )

        sorted_evaluation_users = sorted(evaluation_users, key=repr)
        visible_raw = {
            policy: self._latest_history_rows(
                source_by_policy[policy],
                users=sorted_evaluation_users,
            )
            for policy in POLICIES
        }
        training_raw = {
            policy: pd.concat(
                [
                    source_by_policy[policy].loc[
                        ~source_by_policy[policy]["user_id"].isin(
                            evaluation_users
                        )
                    ],
                    visible_raw[policy],
                ],
                ignore_index=True,
            )
            .sort_values(
                ["user_id", "timestamp_ms", "item_id", "interaction_id"],
                kind="mergesort",
            )
            .reset_index(drop=True)
            for policy in POLICIES
        }
        visible = {
            policy: helper._history_frame(rows, policy=policy)
            for policy, rows in visible_raw.items()
        }
        training = {
            policy: helper._history_frame(rows, policy=policy)
            for policy, rows in training_raw.items()
        }
        training_exposure_degrees = {
            policy: (
                frame.groupby("item_id", sort=False)["user_id"].nunique()
            )
            for policy, frame in training.items()
        }
        for frame in target_frames.values():
            for policy in POLICIES:
                frame[f"{policy}_training_exposure_user_degree"] = (
                    frame["item_id"]
                    .map(training_exposure_degrees[policy])
                    .fillna(0)
                    .astype(int)
                )
            frame["minimum_training_exposure_user_degree"] = frame[
                [
                    f"{policy}_training_exposure_user_degree"
                    for policy in POLICIES
                ]
            ].min(axis=1)
        permuted = {
            policy: _maximum_disagreement_permutation(visible[policy])
            for policy in POLICIES
        }
        self._require_training_warmth(
            training,
            target_frames=target_frames,
        )

        schema = TaskSchema(
            target_column="target",
            feature_columns=("item_description", "item_category_path"),
            id_columns=("user_id", "item_id"),
            candidate_group_column="candidate_group",
            sampled_column=None,
            history_context_columns=("history_click",),
        )
        frames = {
            "standard_history": visible["standard"],
            "random_history": visible["random"],
            "standard_history_permuted": permuted["standard"],
            "random_history_permuted": permuted["random"],
            "standard_training_history": training["standard"],
            "random_training_history": training["random"],
            "val_standard_targets": target_frames[("val", "standard")],
            "val_random_targets": target_frames[("val", "random")],
            "test_standard_targets": target_frames[("test", "standard")],
            "test_random_targets": target_frames[("test", "random")],
        }
        frame_hashes = {
            name: _frame_sha256(frame) for name, frame in frames.items()
        }
        canonical_manifest_sha256 = _json_sha256(canonical_manifest)
        canonical_content_identity = _canonical_content_identity(
            canonical_manifest,
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            table_rows={
                "users": len(users),
                "items": len(items),
                "interactions": len(interactions),
            },
        )
        canonical_content_sha256 = _json_sha256(
            canonical_content_identity
        )
        task_identity = _json_sha256(
            {
                "protocol": "natural-exposure-policy-shift-v1",
                "history_size": self.history_size,
                "candidate_size": self.candidate_size,
                "min_exposure_users_per_policy": (
                    self.min_exposure_users_per_policy
                ),
                "min_training_exposure_users_per_policy": (
                    self.min_training_exposure_users_per_policy
                ),
                "candidate_seed": self.candidate_seed,
                "canonical_content_sha256": canonical_content_sha256,
                "frames": frame_hashes,
            }
        )
        manifest = {
            "dataset": dataset.name,
            "dataset_version": dataset.version,
            "task_type": "exposure_policy_shift",
            "protocol": "natural-exposure-policy-shift-v1",
            "candidate_seed": self.candidate_seed,
            "windows": {
                "source": [_date_iso(SOURCE_START), _date_iso(SOURCE_END)],
                "val": [_date_iso(VAL_START), _date_iso(VAL_END)],
                "test": [_date_iso(TEST_START), _date_iso(TEST_END)],
            },
            "scenario": {"column": "tab", "value": TAB_VALUE},
            "history": {
                "visible_rows_per_evaluation_user_per_policy": (
                    self.history_size
                ),
                "selection": (
                    "latest H unique source-window user-item pairs; labels "
                    "do not affect selection"
                ),
                "evaluation_users": len(sorted_evaluation_users),
                "evaluation_user_ids_sha256": _values_sha256(
                    sorted_evaluation_users
                ),
                "training_graph": (
                    "all source pairs from non-evaluation users plus the same "
                    "visible H pairs from evaluation users"
                ),
                "permutation_control": (
                    "within-user maximum-disagreement binary-label "
                    "permutation preserving every click multiset"
                ),
                "policies": {
                    policy: {
                        "visible": _history_summary(visible[policy]),
                        "training": _history_summary(training[policy]),
                        "permutation": _permutation_summary(
                            visible[policy],
                            permuted[policy],
                        ),
                    }
                    for policy in POLICIES
                },
            },
            "candidate_construction": {
                "kind": "label_blind_natural_exposure_group",
                "candidates_per_group": self.candidate_size,
                "selection": (
                    "stable seeded hash over eligible exposure identity; "
                    "independent of target_click"
                ),
                "candidate_seed": self.candidate_seed,
                "missing_as_negative": False,
                "target_user_selection_uses_labels": False,
                "candidate_selection_uses_labels": False,
                "simultaneous_logged_slate": False,
            },
            "support": {
                "filter": "distinct source exposure users under each policy",
                "uses_click_labels": False,
                "minimum_exposure_users_per_policy": (
                    self.min_exposure_users_per_policy
                ),
                "minimum_training_exposure_users_per_policy": (
                    self.min_training_exposure_users_per_policy
                ),
                "common_supported_items": len(common_supported_items),
                "common_supported_item_ids_sha256": _values_sha256(
                    sorted(common_supported_items, key=repr)
                ),
                "positive_user_degree_is_analysis_stratum_only": True,
                "source_degree_summaries": {
                    policy: {
                        "exposure": _numeric_summary(
                            exposure_degrees[policy]
                        ),
                        "positive": _numeric_summary(
                            positive_degrees[policy].reindex(
                                exposure_degrees[policy].index,
                                fill_value=0,
                            )
                        ),
                    }
                    for policy in POLICIES
                },
            },
            "pair_novelty": {
                "required": True,
                "definition": (
                    "target user-item pair absent before target-window start "
                    "under both policies and every released tab"
                ),
            },
            "targets": target_audit,
            "metadata_visibility": {
                "history": [
                    "item_description",
                    "item_category_path",
                    "history_click",
                ],
                "candidate": [
                    "item_description",
                    "item_category_path",
                ],
                "hidden_from_llm": [
                    "user_id",
                    "item_id",
                    "candidate_group",
                    "history_policy",
                    "target_policy",
                    "event_date",
                    "event_timestamp_ms",
                    "interaction_id",
                    "target",
                    "standard_source_exposure_user_degree",
                    "random_source_exposure_user_degree",
                    "standard_source_positive_user_degree",
                    "random_source_positive_user_degree",
                    "minimum_source_exposure_user_degree",
                    "minimum_source_positive_user_degree",
                    "source_positive_support_stratum",
                    "standard_training_exposure_user_degree",
                    "random_training_exposure_user_degree",
                    "minimum_training_exposure_user_degree",
                ],
                "semantic_coverage": semantic_audit,
            },
            "rows": {
                name: int(len(frame)) for name, frame in frames.items()
            },
            "users": {
                name: int(frame["user_id"].nunique())
                for name, frame in frames.items()
            },
            "hashes": {
                "canonical_manifest_sha256": canonical_manifest_sha256,
                "canonical_content_sha256": canonical_content_sha256,
                "frames": frame_hashes,
                "task_identity_sha256": task_identity,
            },
            "canonical_content_identity": canonical_content_identity,
            "canonical_provenance": {
                "raw_sources": canonical_manifest.get("raw_sources", []),
                "semantic_supplement": canonical_manifest.get(
                    "semantic_supplement",
                    canonical_manifest.get("kuairand_semantic_supplement"),
                ),
            },
            "known_caveats": [
                (
                    "Natural candidate rows are separately logged exposures, "
                    "not one simultaneous recommendation slate."
                ),
                (
                    "Standard and random target columns have different natural "
                    "candidate and click distributions; compare histories only "
                    "within one target-policy column."
                ),
                (
                    "All-zero and all-one groups remain materialized; ranking "
                    "metrics must report mixed-label coverage."
                ),
                (
                    "Only probability-producing scorers support natural-rate "
                    "log-loss, Brier, and calibration; ranking-only scorers "
                    "remain conditional ranking baselines."
                ),
                (
                    "KuaiRand-Pure omits standard-feed items outside the "
                    "released intervention pool."
                ),
                (
                    "The native standard and random source logs have different "
                    "volumes. Matched-versus-transferred collaborative gaps "
                    "therefore describe robustness to the released regimes, "
                    "not a source-size-controlled causal policy effect."
                ),
            ],
        }
        return KuaiRandPolicyShiftTask(
            name=self.name,
            standard_history=visible["standard"],
            random_history=visible["random"],
            standard_history_permuted=permuted["standard"],
            random_history_permuted=permuted["random"],
            standard_training_history=training["standard"],
            random_training_history=training["random"],
            val_standard_targets=target_frames[("val", "standard")],
            val_random_targets=target_frames[("val", "random")],
            test_standard_targets=target_frames[("test", "standard")],
            test_random_targets=target_frames[("test", "random")],
            schema=schema,
            manifest=manifest,
        )

    def _target_pools(
        self,
        *,
        helper: KuaiRandPolicyShiftTaskBuilder,
        tab1: pd.DataFrame,
        all_interactions: pd.DataFrame,
        start: int,
        end: int,
        allowed_items: set[object],
    ) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
        prior_pairs = (
            all_interactions.loc[
                all_interactions["_date_key"].lt(start),
                ["user_id", "item_id"],
            ]
            .drop_duplicates()
            .assign(_seen_before_target=True)
        )
        target_window = tab1.loc[tab1["_date_key"].between(start, end)]
        pools: dict[str, pd.DataFrame] = {}
        audit: dict[str, Any] = {}
        for policy in POLICIES:
            rows = helper._first_pair_rows(
                target_window.loc[target_window["_policy"].eq(policy)].copy()
            )
            before_support = len(rows)
            rows = rows.loc[rows["item_id"].isin(allowed_items)].copy()
            after_support = len(rows)
            rows = rows.merge(
                prior_pairs,
                on=["user_id", "item_id"],
                how="left",
                validate="many_to_one",
            )
            rows = rows.loc[rows["_seen_before_target"].isna()].drop(
                columns="_seen_before_target"
            )
            pools[policy] = rows.reset_index(drop=True)
            audit[policy] = {
                "rows_before_item_support": int(before_support),
                "rows_after_item_support": int(after_support),
                "rows_after_pair_novelty": int(len(rows)),
                "click_rate_descriptive_only": (
                    float(rows["target_click"].mean()) if len(rows) else 0.0
                ),
            }
        return pools, audit

    def _target_users(
        self,
        pools: dict[str, pd.DataFrame],
        *,
        source_history_users: set[object],
    ) -> list[object]:
        target_users = set(source_history_users)
        for policy in POLICIES:
            counts = pools[policy].groupby("user_id", sort=False).size()
            target_users &= set(
                counts.loc[counts.ge(self.candidate_size)].index
            )
        return sorted(target_users, key=repr)

    def _materialize_targets(
        self,
        *,
        helper: KuaiRandPolicyShiftTaskBuilder,
        rows: pd.DataFrame,
        users: list[object],
        split: str,
        policy: str,
    ) -> pd.DataFrame:
        eligible = rows.loc[rows["user_id"].isin(users)].copy()
        eligible["_candidate_key"] = [
            _stable_int(
                self.candidate_seed,
                "natural-target",
                split,
                policy,
                user_id,
                item_id,
                interaction_id,
            )
            for user_id, item_id, interaction_id in zip(
                eligible["user_id"],
                eligible["item_id"],
                eligible["interaction_id"],
                strict=True,
            )
        ]
        selected = (
            eligible.sort_values(
                [
                    "user_id",
                    "_candidate_key",
                    "timestamp_ms",
                    "interaction_id",
                ],
                kind="mergesort",
            )
            .groupby("user_id", sort=False)
            .head(self.candidate_size)
            .sort_values(
                ["user_id", "_candidate_key", "interaction_id"],
                kind="mergesort",
            )
            .drop(columns="_candidate_key")
            .reset_index(drop=True)
        )
        selected["candidate_group"] = [
            f"{split}_{policy}_u{user_id}_m{self.candidate_size}"
            for user_id in selected["user_id"]
        ]
        selected["target_policy"] = policy
        return helper._target_frame(selected)

    def _latest_history_rows(
        self,
        rows: pd.DataFrame,
        *,
        users: list[object],
    ) -> pd.DataFrame:
        selected = (
            rows.loc[rows["user_id"].isin(users)]
            .sort_values(
                ["user_id", "timestamp_ms", "item_id", "interaction_id"],
                kind="mergesort",
            )
            .groupby("user_id", sort=False)
            .tail(self.history_size)
            .sort_values(
                ["user_id", "timestamp_ms", "item_id", "interaction_id"],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
        counts = selected.groupby("user_id").size()
        if len(counts) != len(users) or not counts.eq(self.history_size).all():
            raise RuntimeError("Visible history does not contain exactly H rows")
        return selected

    def _validate_targets(
        self,
        targets: dict[str, pd.DataFrame],
        *,
        split: str,
        expected_users: list[object],
    ) -> None:
        expected = set(expected_users)
        for policy in POLICIES:
            frame = targets[policy]
            if set(frame["user_id"]) != expected:
                raise RuntimeError(
                    f"{split}/{policy} target user set changed during selection"
                )
            group_sizes = frame.groupby("candidate_group").size()
            if not group_sizes.eq(self.candidate_size).all():
                raise RuntimeError(
                    f"{split}/{policy} groups are not fixed-size m"
                )
            if frame.groupby("candidate_group")["user_id"].nunique().ne(1).any():
                raise RuntimeError(
                    f"{split}/{policy} group contains multiple users"
                )

    def _require_training_warmth(
        self,
        training: dict[str, pd.DataFrame],
        *,
        target_frames: dict[tuple[str, str], pd.DataFrame],
    ) -> None:
        candidate_items = set().union(
            *(set(frame["item_id"]) for frame in target_frames.values())
        )
        for policy in POLICIES:
            degree = training[policy].groupby("item_id")["user_id"].nunique()
            insufficient = {
                item_id
                for item_id in candidate_items
                if int(degree.get(item_id, 0))
                < self.min_training_exposure_users_per_policy
            }
            if insufficient:
                raise ValueError(
                    f"{len(insufficient)} target items have fewer than "
                    f"{self.min_training_exposure_users_per_policy} users in "
                    f"the {policy} training graph"
                )


def _maximum_disagreement_permutation(history: pd.DataFrame) -> pd.DataFrame:
    """Preserve each click multiset while changing as many assignments as possible."""

    permuted = history.copy()
    result = permuted["history_click"].astype(int).to_numpy(copy=True)
    for _, positions in permuted.groupby("user_id", sort=False).indices.items():
        position_list = list(positions)
        positive = [
            position for position in position_list if result[position] == 1
        ]
        negative = [
            position for position in position_list if result[position] == 0
        ]
        replacement = result[position_list].copy()
        if len(positive) <= len(negative):
            replacement[:] = 0
            negative_offsets = [
                position_list.index(position) for position in negative
            ]
            replacement[negative_offsets[: len(positive)]] = 1
        else:
            replacement[:] = 1
            positive_offsets = [
                position_list.index(position) for position in positive
            ]
            replacement[positive_offsets[: len(negative)]] = 0
        result[position_list] = replacement
    permuted["history_click"] = result
    permuted["target"] = result
    return permuted


def _target_summary(
    rows: pd.DataFrame,
    *,
    candidate_size: int,
) -> dict[str, Any]:
    positives = rows.groupby("candidate_group", sort=False)["target"].sum()
    return {
        "rows": int(len(rows)),
        "users": int(rows["user_id"].nunique()),
        "items": int(rows["item_id"].nunique()),
        "groups": int(rows["candidate_group"].nunique()),
        "click_rate": float(rows["target"].mean()),
        "mixed_label_groups": int(
            (positives.gt(0) & positives.lt(candidate_size)).sum()
        ),
        "mixed_label_group_rate": float(
            (positives.gt(0) & positives.lt(candidate_size)).mean()
        ),
        "all_zero_groups": int(positives.eq(0).sum()),
        "all_one_groups": int(positives.eq(candidate_size).sum()),
    }


def _history_summary(rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(rows)),
        "users": int(rows["user_id"].nunique()),
        "items": int(rows["item_id"].nunique()),
        "click_rate": float(rows["history_click"].mean()),
    }


def _permutation_summary(
    observed: pd.DataFrame,
    permuted: pd.DataFrame,
) -> dict[str, Any]:
    identity = ["user_id", "item_id", "interaction_id"]
    if not observed[identity].equals(permuted[identity]):
        raise RuntimeError("Permutation changed history row identity")
    observed_sum = observed.groupby("user_id")["history_click"].sum()
    permuted_sum = permuted.groupby("user_id")["history_click"].sum()
    if not observed_sum.equals(permuted_sum):
        raise RuntimeError("Permutation changed a user's click multiset")
    changed = observed["history_click"].ne(permuted["history_click"])
    mixed_users = (
        observed.groupby("user_id")["history_click"].nunique().gt(1)
    )
    return {
        "rows": int(len(observed)),
        "changed_rows": int(changed.sum()),
        "changed_fraction": float(changed.mean()),
        "constant_label_users": int((~mixed_users).sum()),
        "mixed_label_users": int(mixed_users.sum()),
    }


def _numeric_summary(values: pd.Series) -> dict[str, int | float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {
            "count": 0,
            "min": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0,
        }
    return {
        "count": int(len(numeric)),
        "min": int(numeric.min()),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "p95": float(numeric.quantile(0.95)),
        "max": int(numeric.max()),
    }


def _date_iso(value: int) -> str:
    text = f"{int(value):08d}"
    return f"{text[:4]}-{text[4:6]}-{text[6:]}"


def _positive_support_stratum(value: object) -> str:
    degree = int(value)
    if degree == 0:
        return "zero"
    if degree == 1:
        return "one"
    if degree <= 4:
        return "two_to_four"
    return "five_or_more"
