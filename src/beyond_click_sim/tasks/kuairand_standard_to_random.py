from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.data.kuairand_semantics import (
    load_kuairand_english_semantics,
)
from beyond_click_sim.tasks.base import TaskSchema


TargetPolicy = Literal["standard", "random"]
TargetPool = Literal["common_warm", "full_pool"]
HistoryControl = Literal["observed", "none"]
EvaluationSplit = Literal["val", "test"]

POLICIES: tuple[TargetPolicy, ...] = ("standard", "random")
SOURCE_START = 20220408
SOURCE_END = 20220421
BUFFER_DATE = 20220422
VAL_START = 20220423
VAL_END = 20220430
TEST_START = 20220501
TEST_END = 20220508
TAB_VALUE = 1


@dataclass(frozen=True)
class KuaiRandStandardToRandomCell:
    """One source-standard history condition and future target-policy cell."""

    task_name: str
    split: EvaluationSplit
    target_policy: TargetPolicy
    history_control: HistoryControl
    history: pd.DataFrame
    training_source: pd.DataFrame
    item_features: pd.DataFrame
    targets: pd.DataFrame
    schema: TaskSchema
    manifest: dict[str, Any]


@dataclass
class KuaiRandStandardToRandomTask:
    """Materialized standard-source to future standard/random KuaiRand task."""

    name: str
    target_pool: TargetPool
    visible_history: pd.DataFrame
    training_source: pd.DataFrame
    item_features: pd.DataFrame
    val_standard_targets: pd.DataFrame
    val_random_targets: pd.DataFrame
    test_standard_targets: pd.DataFrame
    test_random_targets: pd.DataFrame
    schema: TaskSchema
    manifest: dict[str, Any]

    def cell(
        self,
        *,
        split: EvaluationSplit,
        target_policy: TargetPolicy,
        history_control: HistoryControl = "observed",
    ) -> KuaiRandStandardToRandomCell:
        if split not in {"val", "test"}:
            raise ValueError(f"Unsupported split: {split!r}")
        if target_policy not in POLICIES:
            raise ValueError(f"Unsupported target policy: {target_policy!r}")
        if history_control not in {"observed", "none"}:
            raise ValueError(f"Unsupported history control: {history_control!r}")
        targets = self._targets(split, target_policy)
        target_users = set(targets["user_id"])
        history = self.visible_history.loc[
            self.visible_history["user_id"].isin(target_users)
        ].copy()
        if history_control == "none":
            history = history.iloc[0:0].copy()
        return KuaiRandStandardToRandomCell(
            task_name=self.name,
            split=split,
            target_policy=target_policy,
            history_control=history_control,
            history=history,
            training_source=self.training_source.copy(),
            item_features=self.item_features.copy(),
            targets=targets.copy(),
            schema=self.schema,
            manifest={
                "parent_task": self.name,
                "target_pool": self.target_pool,
                "split": split,
                "target_policy": target_policy,
                "history_control": history_control,
                "history_sha256": _frame_sha256(history),
                "training_source_sha256": _frame_sha256(self.training_source),
                "item_features_sha256": _frame_sha256(self.item_features),
                "targets_sha256": _frame_sha256(targets),
            },
        )

    def _targets(
        self,
        split: EvaluationSplit,
        policy: TargetPolicy,
    ) -> pd.DataFrame:
        if split == "val":
            return (
                self.val_standard_targets
                if policy == "standard"
                else self.val_random_targets
            )
        return (
            self.test_standard_targets
            if policy == "standard"
            else self.test_random_targets
        )


class KuaiRandStandardToRandomTaskBuilder:
    """Build the final pre-intervention-standard to future-exposure task.

    Cohorts are always selected from the common-warm pools, including when
    ``target_pool="full_pool"``. Thus primary and secondary tasks use identical
    users and differ only in candidate-pool eligibility.
    """

    def __init__(
        self,
        *,
        translation_dir: Path | str,
        target_pool: TargetPool = "common_warm",
        candidate_seed: int = 0,
        cohort_seed: int = 0,
        history_size: int = 20,
        candidate_size: int = 10,
        validation_users: int = 200,
        test_users: int = 500,
        min_source_exposures: int = 20,
        min_source_users: int = 5,
        name: str | None = None,
    ) -> None:
        if target_pool not in {"common_warm", "full_pool"}:
            raise ValueError(f"Unsupported target_pool: {target_pool!r}")
        positive_parameters = {
            "history_size": history_size,
            "candidate_size": candidate_size,
            "validation_users": validation_users,
            "test_users": test_users,
            "min_source_exposures": min_source_exposures,
            "min_source_users": min_source_users,
        }
        invalid = [
            key for key, value in positive_parameters.items() if value <= 0
        ]
        if invalid:
            raise ValueError(f"Parameters must be positive: {invalid}")
        self.translation_dir = Path(translation_dir).expanduser().resolve()
        self.target_pool = target_pool
        self.candidate_seed = int(candidate_seed)
        self.cohort_seed = int(cohort_seed)
        self.history_size = int(history_size)
        self.candidate_size = int(candidate_size)
        self.validation_users = int(validation_users)
        self.test_users = int(test_users)
        self.min_source_exposures = int(min_source_exposures)
        self.min_source_users = int(min_source_users)
        self.name = name or (
            "kuairand_standard_source_to_future"
            f"_{target_pool}_h{history_size}_m{candidate_size}"
            f"_cohort{cohort_seed}_candidate{candidate_seed}"
        )

    def build(self, dataset: CanonicalDataset) -> KuaiRandStandardToRandomTask:
        canonical_manifest = dataset.load_manifest()
        semantics = load_kuairand_english_semantics(
            self.translation_dir,
            canonical_items_path=dataset.items_path,
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
        prepared = self._prepare_interactions(interactions)
        prepared = self._mark_first_released_pairs(prepared)
        semantic_item_ids = set(semantics.items["item_id"])

        source = prepared.loc[
            prepared["item_id"].isin(semantic_item_ids)
            & prepared["tab"].eq(TAB_VALUE)
            & prepared["exposure_policy"].eq("standard")
            & prepared["_date_key"].between(SOURCE_START, SOURCE_END)
        ].copy()
        if source.empty:
            raise ValueError("The fixed standard source window is empty")
        if source["exposure_policy"].ne("standard").any():
            raise AssertionError("Random rows entered the standard source")
        if source["_date_key"].eq(BUFFER_DATE).any():
            raise AssertionError("The April 22 buffer entered the source")

        source_degrees = self._source_degrees(source)
        structural_warm = source_degrees.loc[
            source_degrees["source_exposure_count"].ge(
                self.min_source_exposures
            )
            & source_degrees["source_distinct_user_count"].ge(
                self.min_source_users
            )
        ]
        common_warm = structural_warm
        common_warm_items = set(common_warm.index)
        if not common_warm_items:
            raise ValueError("No items meet the common-warm source gate")

        latest_source_pairs = self._latest_source_pairs(source)
        history_counts = latest_source_pairs.groupby(
            "user_id", sort=False
        ).size()
        history_eligible_users = set(
            history_counts.loc[history_counts.ge(self.history_size)].index
        )

        windows = {
            "val": (VAL_START, VAL_END),
            "test": (TEST_START, TEST_END),
        }
        all_pools: dict[tuple[str, str], pd.DataFrame] = {}
        pool_audit: dict[str, Any] = {}
        warm_eligible: dict[str, set[Any]] = {}
        for split, (start, end) in windows.items():
            split_audit: dict[str, Any] = {}
            warm_policy_users: dict[str, set[Any]] = {}
            for policy in POLICIES:
                base = prepared.loc[
                    prepared["item_id"].isin(semantic_item_ids)
                    & prepared["tab"].eq(TAB_VALUE)
                    & prepared["exposure_policy"].eq(policy)
                    & prepared["_date_key"].between(start, end)
                ]
                first = base.loc[base["_pair_rank"].eq(0)].copy()
                warm = first.loc[
                    first["item_id"].isin(common_warm_items)
                ].copy()
                all_pools[(split, policy)] = first
                warm_counts = warm.groupby("user_id", sort=False).size()
                warm_policy_users[policy] = set(
                    warm_counts.loc[
                        warm_counts.ge(self.candidate_size)
                    ].index
                )
                split_audit[policy] = {
                    "target_window_rows": int(len(base)),
                    "first_released_pair_rows": int(len(first)),
                    "rows_removed_by_pair_novelty": int(len(base) - len(first)),
                    "common_warm_rows": int(len(warm)),
                }
            warm_eligible[split] = (
                history_eligible_users
                & warm_policy_users["standard"]
                & warm_policy_users["random"]
            )
            split_audit["common_warm_eligible_users"] = len(
                warm_eligible[split]
            )
            pool_audit[split] = split_audit

        validation_cohort = self._select_users(
            warm_eligible["val"],
            n=self.validation_users,
            namespace="validation-cohort",
        )
        test_available = warm_eligible["test"] - set(validation_cohort)
        test_cohort = self._select_users(
            test_available,
            n=self.test_users,
            namespace="test-cohort",
        )
        if set(validation_cohort) & set(test_cohort):
            raise AssertionError("Validation and test cohorts overlap")
        evaluation_users = sorted(
            {*validation_cohort, *test_cohort},
            key=repr,
        )

        visible_raw = (
            latest_source_pairs.loc[
                latest_source_pairs["user_id"].isin(evaluation_users)
            ]
            .sort_values(
                [
                    "user_id",
                    "_date_key",
                    "timestamp_ms",
                    "interaction_id",
                ],
                kind="mergesort",
            )
            .groupby("user_id", sort=False)
            .tail(self.history_size)
            .sort_values(
                [
                    "user_id",
                    "_date_key",
                    "timestamp_ms",
                    "interaction_id",
                ],
                kind="mergesort",
            )
            .reset_index(drop=True)
            .merge(
                semantics.items,
                on="item_id",
                how="left",
                validate="many_to_one",
            )
        )
        visible_counts = visible_raw.groupby("user_id").size()
        if (
            len(visible_counts) != len(evaluation_users)
            or not visible_counts.eq(self.history_size).all()
        ):
            raise AssertionError("Visible histories do not contain exactly H rows")

        training_source = self._history_frame(source, include_metadata=False)
        visible_history = self._history_frame(visible_raw)
        cohorts = {"val": validation_cohort, "test": test_cohort}
        targets: dict[tuple[str, str], pd.DataFrame] = {}
        for split in ("val", "test"):
            for policy in POLICIES:
                selected = self._select_candidates(
                    (
                        all_pools[(split, policy)]
                        if self.target_pool == "full_pool"
                        else all_pools[(split, policy)].loc[
                            all_pools[(split, policy)]["item_id"].isin(
                                common_warm_items
                            )
                        ]
                    ),
                    users=cohorts[split],
                    split=split,
                    policy=policy,
                )
                selected = selected.merge(
                    semantics.items,
                    on="item_id",
                    how="left",
                    validate="many_to_one",
                )
                targets[(split, policy)] = self._target_frame(
                    selected,
                    source_degrees=source_degrees,
                )
                self._validate_target_frame(
                    targets[(split, policy)],
                    expected_users=cohorts[split],
                )

        schema = TaskSchema(
            target_column="target",
            feature_columns=(
                "item_caption_en",
                "item_cover_text_en",
                "item_category_path_en",
            ),
            id_columns=("user_id", "item_id"),
            candidate_group_column="candidate_group",
            sampled_column=None,
            history_context_columns=("history_click",),
        )
        frame_map = {
            "training_source": training_source,
            "item_features": semantics.items,
            "visible_history": visible_history,
            "val_standard_targets": targets[("val", "standard")],
            "val_random_targets": targets[("val", "random")],
            "test_standard_targets": targets[("test", "standard")],
            "test_random_targets": targets[("test", "random")],
        }
        frame_hashes = {
            name: _frame_sha256(frame) for name, frame in frame_map.items()
        }
        manifest = {
            "dataset": dataset.name,
            "dataset_version": dataset.version,
            "task_type": "standard_source_to_future_exposure",
            "protocol": "kuairand-standard-to-random-v2",
            "target_pool": self.target_pool,
            "target_source_column": "target_click",
            "target_column": "target",
            "scenario": {"column": "tab", "value": TAB_VALUE},
            "windows": {
                "source": [_date_iso(SOURCE_START), _date_iso(SOURCE_END)],
                "buffer": [_date_iso(BUFFER_DATE), _date_iso(BUFFER_DATE)],
                "val": [_date_iso(VAL_START), _date_iso(VAL_END)],
                "test": [_date_iso(TEST_START), _date_iso(TEST_END)],
            },
            "source": {
                "policy": "standard",
                "rows": int(len(training_source)),
                "users": int(training_source["user_id"].nunique()),
                "items": int(training_source["item_id"].nunique()),
                "complete_source_used_by_classical_models": True,
                "item_metadata_stored_once": True,
                "visible_history_size": self.history_size,
                "visible_history_selection": (
                    "latest exposure per user-item pair, then latest H pairs"
                ),
                "history_eligible_users": len(history_eligible_users),
            },
            "support": {
                "minimum_source_exposures": self.min_source_exposures,
                "minimum_distinct_source_users": self.min_source_users,
                "structural_warm_items": int(len(structural_warm)),
                "common_warm_items": int(len(common_warm)),
                "positive_source_user_degree_is_audit_only": True,
                "common_warm_item_ids_sha256": _values_sha256(
                    sorted(common_warm_items, key=repr)
                ),
                "full_pool_degree_strata": [
                    "0",
                    "1--4",
                    "5--19",
                    "20--99",
                    "100+",
                ],
            },
            "pair_novelty": {
                "required": True,
                "definition": (
                    "candidate is the first released exposure of its user-item "
                    "pair across all tabs and both policies"
                ),
                "total_order": [
                    "date",
                    "timestamp_ms",
                    "interaction_id",
                ],
                "buffer_participates": True,
                "audit": pool_audit,
            },
            "cohorts": {
                "selection_uses_target_labels": False,
                "selection_order": "validation first, then disjoint test",
                "cohort_seed": self.cohort_seed,
                "common_warm_eligible": {
                    split: len(users)
                    for split, users in warm_eligible.items()
                },
                "validation": {
                    "users": len(validation_cohort),
                    "user_ids_sha256": _values_sha256(validation_cohort),
                },
                "test": {
                    "users": len(test_cohort),
                    "user_ids_sha256": _values_sha256(test_cohort),
                },
                "identical_for_common_warm_and_full_pool": True,
            },
            "candidate_construction": {
                "kind": "label_blind_natural_exposure_group",
                "candidate_seed": self.candidate_seed,
                "candidates_per_group": self.candidate_size,
                "missing_as_negative": False,
                "labels_balanced": False,
                "selection_uses_target_labels": False,
                "simultaneous_logged_slate": False,
            },
            "targets": {
                f"{split}_{policy}": _target_summary(
                    frame,
                    candidate_size=self.candidate_size,
                )
                for (split, policy), frame in targets.items()
            },
            "metadata_visibility": {
                "language": "English",
                "translation_model": semantics.manifest.get("model"),
                "history": [
                    "item_caption_en",
                    "item_cover_text_en",
                    "item_category_path_en",
                    "history_click",
                ],
                "candidate": [
                    "item_caption_en",
                    "item_cover_text_en",
                    "item_category_path_en",
                ],
                "hidden_from_llm": [
                    "user_id",
                    "item_id",
                    "interaction_id",
                    "event_date",
                    "event_timestamp_ms",
                    "target_policy",
                    "target_pool",
                    "candidate_group",
                    "source_exposure_count",
                    "source_distinct_user_count",
                    "source_positive_user_count",
                    "source_exposure_degree_stratum",
                    "source_exposure_cold",
                    "source_positive_graph_cold",
                    "target",
                ],
            },
            "rows": {
                name: int(len(frame)) for name, frame in frame_map.items()
            },
            "hashes": {
                "canonical_manifest_sha256": _json_sha256(
                    canonical_manifest
                ),
                "translation_manifest_sha256": semantics.manifest_sha256,
                "translation_items_sha256": semantics.items_sha256,
                "frames": frame_hashes,
            },
            "known_caveats": [
                (
                    "Natural candidate rows are separate logged exposures, "
                    "not a simultaneous production slate."
                ),
                (
                    "The random target differs both temporally and by exposure "
                    "regime from the pre-intervention standard source."
                ),
                (
                    "All-zero and all-positive groups remain materialized and "
                    "must be excluded only from conditional ranking metrics."
                ),
            ],
        }
        manifest["hashes"]["task_identity_sha256"] = _json_sha256(
            {
                "protocol": manifest["protocol"],
                "target_pool": self.target_pool,
                "candidate_seed": self.candidate_seed,
                "cohort_seed": self.cohort_seed,
                "translation_items_sha256": semantics.items_sha256,
                "frames": frame_hashes,
            }
        )
        return KuaiRandStandardToRandomTask(
            name=self.name,
            target_pool=self.target_pool,
            visible_history=visible_history,
            training_source=training_source,
            item_features=semantics.items.copy(),
            val_standard_targets=targets[("val", "standard")],
            val_random_targets=targets[("val", "random")],
            test_standard_targets=targets[("test", "standard")],
            test_random_targets=targets[("test", "random")],
            schema=schema,
            manifest=manifest,
        )

    @staticmethod
    def _prepare_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
        required = [
            "interaction_id",
            "user_id",
            "item_id",
            "exposure_policy",
            "tab",
            "date",
            "timestamp_ms",
            "target_click",
        ]
        missing = [column for column in required if column not in interactions]
        if missing:
            raise ValueError(f"Interactions are missing columns: {missing}")
        # The canonical parquet is already typed. Avoid a deep copy of all
        # 2.6M rows here; subsequent assignments replace only columns that
        # actually need normalization.
        rows = interactions.loc[:, required].copy(deep=False)
        for column in ("interaction_id", "user_id", "item_id"):
            if rows[column].isna().any():
                raise ValueError(f"{column} contains missing values")
            if not (
                pd.api.types.is_object_dtype(rows[column])
                or pd.api.types.is_string_dtype(rows[column])
            ):
                rows[column] = rows[column].astype(str)
        policy = rows["exposure_policy"]
        if policy.isna().any():
            raise ValueError("exposure_policy contains missing values")
        normalized_policy = policy.str.strip().str.lower()
        if not normalized_policy.equals(policy):
            rows["exposure_policy"] = normalized_policy
        unknown = set(rows["exposure_policy"].dropna().unique()) - set(POLICIES)
        if unknown:
            raise ValueError(f"Unsupported exposure policies: {sorted(unknown)}")
        rows["_date_key"] = _normalize_date_key(rows["date"])
        if not pd.api.types.is_integer_dtype(rows["tab"]):
            rows["tab"] = pd.to_numeric(
                rows["tab"], errors="raise"
            ).astype(int)
        if not pd.api.types.is_integer_dtype(rows["timestamp_ms"]):
            rows["timestamp_ms"] = pd.to_numeric(
                rows["timestamp_ms"], errors="raise"
            ).astype("int64")
        click = rows["target_click"]
        if not pd.api.types.is_numeric_dtype(click):
            click = pd.to_numeric(click, errors="coerce")
        if click.isna().any() or not set(click.unique()) <= {0, 1}:
            raise ValueError("target_click must contain only binary 0/1 values")
        if str(click.dtype) != "int8":
            rows["target_click"] = click.astype("int8")
        if rows["interaction_id"].duplicated().any():
            raise ValueError("interaction_id must be unique")
        return rows

    @staticmethod
    def _mark_first_released_pairs(rows: pd.DataFrame) -> pd.DataFrame:
        rows.sort_values(
            [
                "user_id",
                "item_id",
                "_date_key",
                "timestamp_ms",
                "interaction_id",
            ],
            kind="mergesort",
            inplace=True,
        )
        rows["_pair_rank"] = rows.groupby(
            ["user_id", "item_id"], sort=False
        ).cumcount()
        return rows

    @staticmethod
    def _latest_source_pairs(source: pd.DataFrame) -> pd.DataFrame:
        return (
            source.sort_values(
                [
                    "user_id",
                    "item_id",
                    "_date_key",
                    "timestamp_ms",
                    "interaction_id",
                ],
                kind="mergesort",
            )
            .drop_duplicates(["user_id", "item_id"], keep="last")
            .reset_index(drop=True)
        )

    @staticmethod
    def _source_degrees(source: pd.DataFrame) -> pd.DataFrame:
        degrees = source.groupby("item_id", sort=False).agg(
            source_exposure_count=("interaction_id", "size"),
            source_distinct_user_count=("user_id", "nunique"),
        )
        positive = (
            source.loc[source["target_click"].eq(1)]
            .groupby("item_id", sort=False)["user_id"]
            .nunique()
        )
        degrees["source_positive_user_count"] = positive.reindex(
            degrees.index, fill_value=0
        ).astype(int)
        return degrees

    def _select_users(
        self,
        users: set[Any],
        *,
        n: int,
        namespace: str,
    ) -> list[Any]:
        if len(users) < n:
            raise ValueError(
                f"Only {len(users)} users are eligible for {namespace}; need {n}"
            )
        return sorted(
            users,
            key=lambda user_id: (
                _stable_int(self.cohort_seed, namespace, user_id),
                repr(user_id),
            ),
        )[:n]

    def _select_candidates(
        self,
        rows: pd.DataFrame,
        *,
        users: list[Any],
        split: str,
        policy: str,
    ) -> pd.DataFrame:
        selected_pool = rows.loc[rows["user_id"].isin(users)].copy()
        selected_pool["_candidate_key"] = [
            _stable_int(
                self.candidate_seed,
                "candidate",
                self.target_pool,
                split,
                policy,
                user_id,
                item_id,
                interaction_id,
            )
            for user_id, item_id, interaction_id in zip(
                selected_pool["user_id"],
                selected_pool["item_id"],
                selected_pool["interaction_id"],
                strict=True,
            )
        ]
        selected = (
            selected_pool.sort_values(
                ["user_id", "_candidate_key", "interaction_id"],
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
            (
                f"{split}_{policy}_{self.target_pool}_seed"
                f"{self.candidate_seed}_u{user_id}"
            )
            for user_id in selected["user_id"]
        ]
        selected["target_policy"] = policy
        selected["target_pool"] = self.target_pool
        return selected

    @staticmethod
    def _base_output_frame(
        rows: pd.DataFrame,
        *,
        include_metadata: bool = True,
    ) -> pd.DataFrame:
        payload: dict[str, Any] = {
            "user_id": rows["user_id"].to_numpy(),
            "item_id": rows["item_id"].to_numpy(),
        }
        if include_metadata:
            payload.update(
                {
                    "item_caption_en": rows["item_caption_en"].to_numpy(),
                    "item_cover_text_en": rows[
                        "item_cover_text_en"
                    ].to_numpy(),
                    "item_category_path_en": rows[
                        "item_category_path_en"
                    ].to_numpy(),
                }
            )
        payload.update(
            {
                "event_date": rows["_date_key"].map(_date_iso).to_numpy(),
                "event_timestamp_ms": rows["timestamp_ms"].to_numpy(),
                "interaction_id": rows["interaction_id"].to_numpy(),
            }
        )
        return pd.DataFrame(payload)

    @classmethod
    def _history_frame(
        cls,
        rows: pd.DataFrame,
        *,
        include_metadata: bool = True,
    ) -> pd.DataFrame:
        frame = cls._base_output_frame(
            rows,
            include_metadata=include_metadata,
        )
        labels = rows["target_click"].to_numpy(dtype=int)
        frame["history_click"] = labels
        frame["target"] = labels
        frame["candidate_group"] = pd.NA
        frame["target_policy"] = pd.NA
        frame["target_pool"] = pd.NA
        return frame

    @classmethod
    def _target_frame(
        cls,
        rows: pd.DataFrame,
        *,
        source_degrees: pd.DataFrame,
    ) -> pd.DataFrame:
        frame = cls._base_output_frame(rows)
        frame["history_click"] = pd.NA
        frame["target"] = rows["target_click"].to_numpy(dtype=int)
        frame["candidate_group"] = rows["candidate_group"].to_numpy()
        frame["target_policy"] = rows["target_policy"].to_numpy()
        frame["target_pool"] = rows["target_pool"].to_numpy()
        for column in (
            "source_exposure_count",
            "source_distinct_user_count",
            "source_positive_user_count",
        ):
            frame[column] = (
                frame["item_id"]
                .map(source_degrees[column])
                .fillna(0)
                .astype(int)
            )
        frame["source_exposure_degree_stratum"] = (
            frame["source_exposure_count"]
            .map(_exposure_degree_stratum)
            .astype("string")
        )
        frame["source_exposure_cold"] = frame[
            "source_exposure_count"
        ].eq(0)
        frame["source_positive_graph_cold"] = frame[
            "source_positive_user_count"
        ].eq(0)
        return frame

    def _validate_target_frame(
        self,
        frame: pd.DataFrame,
        *,
        expected_users: list[Any],
    ) -> None:
        if set(frame["user_id"]) != set(expected_users):
            raise AssertionError("Target user cohort changed during sampling")
        sizes = frame.groupby("candidate_group", sort=False).size()
        if len(sizes) != len(expected_users) or not sizes.eq(
            self.candidate_size
        ).all():
            raise AssertionError("Target groups do not contain exactly m rows")
        if frame.groupby("candidate_group")["user_id"].nunique().ne(1).any():
            raise AssertionError("A candidate group contains multiple users")
        if frame["history_click"].notna().any():
            raise AssertionError("Target candidates leak history feedback")
        if self.target_pool == "common_warm":
            if (
                frame["source_exposure_count"]
                .lt(self.min_source_exposures)
                .any()
                or frame["source_distinct_user_count"]
                .lt(self.min_source_users)
                .any()
            ):
                raise AssertionError("A common-warm target violates its gate")


def _target_summary(
    rows: pd.DataFrame,
    *,
    candidate_size: int,
) -> dict[str, Any]:
    positives = rows.groupby("candidate_group", sort=False)["target"].sum()
    mixed = positives.gt(0) & positives.lt(candidate_size)
    return {
        "rows": int(len(rows)),
        "users": int(rows["user_id"].nunique()),
        "items": int(rows["item_id"].nunique()),
        "groups": int(len(positives)),
        "positive_rate": float(rows["target"].mean()),
        "all_zero_groups": int(positives.eq(0).sum()),
        "all_positive_groups": int(positives.eq(candidate_size).sum()),
        "mixed_label_groups": int(mixed.sum()),
        "mixed_label_group_rate": float(mixed.mean()),
        "source_exposure_cold_rows": int(rows["source_exposure_cold"].sum()),
        "source_positive_graph_cold_rows": int(
            rows["source_positive_graph_cold"].sum()
        ),
    }


def _normalize_date_key(values: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(values):
        integer = values
        if integer.between(19000101, 29991231).all():
            return integer
    text = values.astype("string").str.strip()
    compact = text.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    parsed.loc[compact] = pd.to_datetime(
        text.loc[compact],
        format="%Y%m%d",
        errors="coerce",
    )
    parsed.loc[~compact] = pd.to_datetime(
        text.loc[~compact],
        errors="coerce",
    )
    if parsed.isna().any():
        examples = values.loc[parsed.isna()].head(3).tolist()
        raise ValueError(f"Could not parse interaction dates: {examples}")
    return parsed.dt.strftime("%Y%m%d").astype(int)


def _date_iso(value: int) -> str:
    text = f"{int(value):08d}"
    return f"{text[:4]}-{text[4:6]}-{text[6:]}"


def _exposure_degree_stratum(value: object) -> str:
    degree = int(value)
    if degree == 0:
        return "0"
    if degree <= 4:
        return "1--4"
    if degree <= 19:
        return "5--19"
    if degree <= 99:
        return "20--99"
    return "100+"


def _stable_int(seed: int, *parts: object) -> int:
    payload = "\0".join([str(seed), *(repr(part) for part in parts)])
    return int.from_bytes(
        hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest(),
        "big",
    )


def _values_sha256(values: list[Any]) -> str:
    payload = "\n".join(repr(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(
        index=False,
        lineterminator="\n",
        na_rep="<NA>",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
