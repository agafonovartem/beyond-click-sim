from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.tasks.base import Task, TaskSchema


PolicyRegime = Literal["standard", "random"]
PolicyShiftSplit = Literal["val", "test"]
HistoryControl = Literal["observed", "permuted", "none"]

POLICIES: tuple[PolicyRegime, ...] = ("standard", "random")
SOURCE_START = 20220422
SOURCE_END = 20220430
VAL_START = 20220501
VAL_END = 20220503
TEST_START = 20220504
TEST_END = 20220508
TAB_VALUE = 1
CANDIDATE_SIZE = 3
POSITIVES_PER_GROUP = 1


@dataclass(frozen=True)
class KuaiRandPolicyShiftCell:
    """One fixed history-policy -> target-policy evaluation cell."""

    task_name: str
    history_policy: PolicyRegime
    target_policy: PolicyRegime
    split: PolicyShiftSplit
    history_control: HistoryControl
    history: pd.DataFrame
    training_history: pd.DataFrame
    targets: pd.DataFrame
    schema: TaskSchema
    manifest: dict[str, Any]


@dataclass
class KuaiRandPolicyShiftTask:
    """Materialized 2x2 standard/random KuaiRand policy-shift task.

    The task stores both source-policy histories and both target-policy tables.
    This makes matched and transferred methods consume byte-identical target
    rows within each target-policy column. The generic ``Task`` views returned
    by :meth:`to_task` let existing scorers run on one cell without learning
    about the specialized container.
    """

    name: str
    standard_history: pd.DataFrame
    random_history: pd.DataFrame
    standard_history_permuted: pd.DataFrame
    random_history_permuted: pd.DataFrame
    val_standard_targets: pd.DataFrame
    val_random_targets: pd.DataFrame
    test_standard_targets: pd.DataFrame
    test_random_targets: pd.DataFrame
    schema: TaskSchema
    manifest: dict[str, Any]
    standard_training_history: pd.DataFrame | None = None
    random_training_history: pd.DataFrame | None = None

    def cell(
        self,
        *,
        history_policy: PolicyRegime,
        target_policy: PolicyRegime,
        split: PolicyShiftSplit,
        history_control: HistoryControl = "observed",
    ) -> KuaiRandPolicyShiftCell:
        """Return an isolated view of one history/target policy cell."""

        history = self._history(history_policy, history_control)
        training_history = self._training_history(
            history_policy,
            history_control,
        )
        targets = self._targets(target_policy, split)
        cell_manifest = {
            "parent_task": self.name,
            "history_policy": history_policy,
            "target_policy": target_policy,
            "split": split,
            "history_control": history_control,
            "matched": (
                history_control == "observed"
                and history_policy == target_policy
            ),
            "history_sha256": _frame_sha256(history),
            "training_history_sha256": _frame_sha256(training_history),
            "targets_sha256": _frame_sha256(targets),
        }
        return KuaiRandPolicyShiftCell(
            task_name=self.name,
            history_policy=history_policy,
            target_policy=target_policy,
            split=split,
            history_control=history_control,
            history=history.copy(),
            training_history=training_history.copy(),
            targets=targets.copy(),
            schema=self.schema,
            manifest=cell_manifest,
        )

    def to_task(
        self,
        *,
        history_policy: PolicyRegime,
        target_policy: PolicyRegime,
        history_control: HistoryControl = "observed",
    ) -> Task:
        """Expose one source->target policy pair through the generic Task API."""

        history = self._history(history_policy, history_control)
        training_history = self._training_history(
            history_policy,
            history_control,
        )
        view_name = (
            f"{self.name}_{history_policy}_to_{target_policy}_{history_control}"
        )
        return Task(
            name=view_name,
            train=training_history.copy(),
            val=self._targets(target_policy, "val").copy(),
            test=self._targets(target_policy, "test").copy(),
            schema=self.schema,
            manifest={
                **self.manifest,
                "view": {
                    "history_policy": history_policy,
                    "target_policy": target_policy,
                    "history_control": history_control,
                    "matched": (
                        history_control == "observed"
                        and history_policy == target_policy
                    ),
                    "history_sha256": _frame_sha256(history),
                    "val_targets_sha256": _frame_sha256(
                        self._targets(target_policy, "val")
                    ),
                    "test_targets_sha256": _frame_sha256(
                        self._targets(target_policy, "test")
                    ),
                },
            },
        )

    def _history(
        self,
        policy: PolicyRegime,
        control: HistoryControl,
    ) -> pd.DataFrame:
        _validate_choice("history_policy", policy, POLICIES)
        _validate_choice(
            "history_control",
            control,
            ("observed", "permuted", "none"),
        )
        observed = (
            self.standard_history if policy == "standard" else self.random_history
        )
        if control == "observed":
            return observed
        if control == "permuted":
            return (
                self.standard_history_permuted
                if policy == "standard"
                else self.random_history_permuted
            )
        return observed.iloc[0:0].copy()

    def _training_history(
        self,
        policy: PolicyRegime,
        control: HistoryControl,
    ) -> pd.DataFrame:
        """Return collaborative source data with the visible control applied.

        The original c3 task has no separate training graph, so it falls back
        to the visible history frame. Natural-exposure tasks can retain full
        source evidence from non-evaluation users while exposing exactly H
        rows for every evaluated user.
        """

        observed_visible = (
            self.standard_history if policy == "standard" else self.random_history
        )
        observed_training = (
            self.standard_training_history
            if policy == "standard"
            else self.random_training_history
        )
        if observed_training is None:
            return self._history(policy, control)
        if control == "observed":
            return observed_training
        if control == "none":
            return observed_training.iloc[0:0].copy()

        controlled_visible = self._history(policy, control)
        evaluation_users = set(observed_visible["user_id"])
        non_evaluation_training = observed_training.loc[
            ~observed_training["user_id"].isin(evaluation_users)
        ]
        return pd.concat(
            [non_evaluation_training, controlled_visible],
            ignore_index=True,
        ).sort_values(
            ["user_id", "event_timestamp_ms", "interaction_id"],
            kind="mergesort",
        ).reset_index(drop=True)

    def _targets(
        self,
        policy: PolicyRegime,
        split: PolicyShiftSplit,
    ) -> pd.DataFrame:
        _validate_choice("target_policy", policy, POLICIES)
        _validate_choice("split", split, ("val", "test"))
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


class KuaiRandPolicyShiftTaskBuilder:
    """Build the locked tab-1 KuaiRand standard/random policy-shift task.

    Protocol:
    - tab=1 only for model histories and target construction;
    - 2022-04-22..30 source, 2022-05-01..03 validation,
      2022-05-04..08 test;
    - standard/random source histories balanced to the same number of first
      observed user-item pairs separately for every shared source user;
    - the full outcome-blind balanced source is retained for aggregate/CF
      training; only target users must have both click labels under both
      policies;
    - target items must have a positive observation under both policies in
      that exact full balanced source;
    - exact target user-item pairs must be absent under either policy and every
      tab before the target window;
    - each target group is an observed c3 pseudo-list with one click and two
      non-clicks, and both target policies use the same feasible users;
    - every visible history/candidate item has non-empty released description
      and category text.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        seed: int | None = None,
        source_seed: int = 42,
        candidate_seed: int | None = None,
        max_history_items_per_policy: int = 20,
        min_positive_source_users_per_policy: int = 1,
        user_column: str = "user_id",
        item_column: str = "item_id",
        interaction_id_column: str = "interaction_id",
        policy_column: str = "exposure_policy",
        tab_column: str = "tab",
        date_column: str = "date",
        timestamp_column: str = "timestamp_ms",
        click_column: str = "target_click",
        description_column: str = "description",
        category_column: str = "category_path",
        target_column: str = "target",
        history_click_column: str = "history_click",
        candidate_group_column: str = "candidate_group",
    ) -> None:
        if (
            seed is not None
            and candidate_seed is not None
            and seed != candidate_seed
        ):
            raise ValueError(
                "seed is a backward-compatible alias for candidate_seed; "
                "provide only one value or make them equal"
            )
        resolved_candidate_seed = (
            candidate_seed
            if candidate_seed is not None
            else seed
            if seed is not None
            else 42
        )
        if max_history_items_per_policy <= 0:
            raise ValueError(
                "max_history_items_per_policy must be a positive integer"
            )
        if min_positive_source_users_per_policy <= 0:
            raise ValueError(
                "min_positive_source_users_per_policy must be a positive integer"
            )
        self.source_seed = source_seed
        self.candidate_seed = resolved_candidate_seed
        # Keep the old attribute for callers that still inspect ``seed``.
        # Its narrower meaning is now candidate sampling only.
        self.seed = resolved_candidate_seed
        self.max_history_items_per_policy = max_history_items_per_policy
        self.min_positive_source_users_per_policy = (
            min_positive_source_users_per_policy
        )
        self.name = name or (
            "kuairand_tab1_policy_shift_c3_9d3d5d_"
            f"hcap{max_history_items_per_policy}_"
            f"posdeg{min_positive_source_users_per_policy}_source{source_seed}_"
            f"candidates{resolved_candidate_seed}"
        )
        self.user_column = user_column
        self.item_column = item_column
        self.interaction_id_column = interaction_id_column
        self.policy_column = policy_column
        self.tab_column = tab_column
        self.date_column = date_column
        self.timestamp_column = timestamp_column
        self.click_column = click_column
        self.description_column = description_column
        self.category_column = category_column
        self.target_column = target_column
        self.history_click_column = history_click_column
        self.candidate_group_column = candidate_group_column

    def build(self, dataset: CanonicalDataset) -> KuaiRandPolicyShiftTask:
        canonical_manifest = dataset.load_manifest()
        users = dataset.load_users(columns=[self.user_column])
        items = dataset.load_items(
            columns=[
                self.item_column,
                self.description_column,
                self.category_column,
            ]
        )
        interactions = dataset.load_interactions(
            columns=[
                self.interaction_id_column,
                self.user_column,
                self.item_column,
                self.policy_column,
                self.tab_column,
                self.date_column,
                self.timestamp_column,
                self.click_column,
            ]
        )
        self._require_columns(users, [self.user_column], table="users")
        self._require_columns(
            items,
            [
                self.item_column,
                self.description_column,
                self.category_column,
            ],
            table="items",
        )
        self._require_columns(
            interactions,
            [
                self.interaction_id_column,
                self.user_column,
                self.item_column,
                self.policy_column,
                self.tab_column,
                self.date_column,
                self.timestamp_column,
                self.click_column,
            ],
            table="interactions",
        )
        self._validate_unique_items(items)

        prepared_interactions = self._prepare_interactions(interactions)
        semantic_items, semantic_audit = self._semantic_items(items)
        semantic_interactions = prepared_interactions.merge(
            semantic_items,
            on=self.item_column,
            how="inner",
            validate="many_to_one",
        )
        tab1 = semantic_interactions[
            semantic_interactions[self.tab_column].eq(TAB_VALUE)
        ].copy()

        source = tab1[
            tab1["_date_key"].between(SOURCE_START, SOURCE_END)
        ].copy()
        source_by_policy = {
            policy: self._first_pair_rows(
                source[source["_policy"].eq(policy)].copy()
            )
            for policy in POLICIES
        }
        balanced_source, balance_audit = self._balance_source(source_by_policy)
        target_eligible_users = self._target_eligible_users(balanced_source)
        if not target_eligible_users:
            raise ValueError(
                "No users have positive and negative balanced source labels "
                "under both policies"
            )
        # Do not filter the training graph to the eventual target-user cohort.
        # That cohort restriction is needed for personalized LLM histories,
        # but would discard outcome-blind source evidence from aggregate/CF
        # baselines and artificially weaken their item support.
        self._validate_balanced_histories(balanced_source)

        positive_user_degrees = {
            policy: (
                frame.loc[frame[self.click_column].eq(1)]
                .groupby(self.item_column, sort=False)[self.user_column]
                .nunique()
            )
            for policy, frame in balanced_source.items()
        }
        supported_items = {
            policy: set(
                degree[
                    degree.ge(self.min_positive_source_users_per_policy)
                ].index
            )
            for policy, degree in positive_user_degrees.items()
        }
        common_warm_items = (
            supported_items["standard"] & supported_items["random"]
        )
        if not common_warm_items:
            raise ValueError(
                "No semantically covered items have positive source support "
                "under both policies"
            )

        histories = {
            policy: self._history_frame(frame, policy=policy)
            for policy, frame in balanced_source.items()
        }
        source_degree_audit = {
            policy: self._source_degree_summary(histories[policy])
            for policy in POLICIES
        }
        permuted_histories = {
            policy: self._permuted_history(
                frame,
                namespace=f"{policy}-history-click-permutation",
            )
            for policy, frame in histories.items()
        }

        target_frames: dict[
            tuple[PolicyShiftSplit, PolicyRegime], pd.DataFrame
        ] = {}
        target_audit: dict[str, Any] = {}
        shared_users_by_split: dict[PolicyShiftSplit, list[Any]] = {}
        for split_name, start, end in (
            ("val", VAL_START, VAL_END),
            ("test", TEST_START, TEST_END),
        ):
            built, audit, shared_users = self._build_target_split(
                tab1=tab1,
                all_interactions=prepared_interactions,
                split_name=split_name,
                start=start,
                end=end,
                allowed_users=target_eligible_users,
                allowed_items=common_warm_items,
            )
            target_frames.update(
                {
                    (split_name, policy): frame
                    for policy, frame in built.items()
                }
            )
            target_audit[split_name] = audit
            shared_users_by_split[split_name] = shared_users

        schema = TaskSchema(
            target_column=self.target_column,
            feature_columns=("item_description", "item_category_path"),
            id_columns=(self.user_column, self.item_column),
            candidate_group_column=self.candidate_group_column,
            sampled_column=None,
            history_context_columns=(self.history_click_column,),
        )
        frames_for_hash = {
            "standard_history": histories["standard"],
            "random_history": histories["random"],
            "standard_history_permuted": permuted_histories["standard"],
            "random_history_permuted": permuted_histories["random"],
            "val_standard_targets": target_frames[("val", "standard")],
            "val_random_targets": target_frames[("val", "random")],
            "test_standard_targets": target_frames[("test", "standard")],
            "test_random_targets": target_frames[("test", "random")],
        }
        frame_hashes = {
            name: _frame_sha256(frame)
            for name, frame in frames_for_hash.items()
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
        task_identity_sha256 = _json_sha256(
            {
                "name": self.name,
                "source_seed": self.source_seed,
                "candidate_seed": self.candidate_seed,
                "max_history_items_per_policy": (
                    self.max_history_items_per_policy
                ),
                "min_positive_source_users_per_policy": (
                    self.min_positive_source_users_per_policy
                ),
                "protocol": "tab1-policy-shift-c3-9d3d5d",
                "canonical_content_sha256": canonical_content_sha256,
                "frames": frame_hashes,
            }
        )

        manifest = {
            "dataset": dataset.name,
            "dataset_version": dataset.version,
            "task_type": "exposure_policy_shift",
            "protocol": "tab1-standard-random-c3-9d-source-3d-val-5d-test",
            # ``seed`` remains for older result consumers. New code should
            # read the two explicit seed fields.
            "seed": self.candidate_seed,
            "source_seed": self.source_seed,
            "candidate_seed": self.candidate_seed,
            "min_positive_source_users_per_policy": (
                self.min_positive_source_users_per_policy
            ),
            "target_source_column": self.click_column,
            "target_column": self.target_column,
            "scenario": {
                "column": self.tab_column,
                "value": TAB_VALUE,
            },
            "policy": {
                "column": self.policy_column,
                "regimes": list(POLICIES),
                "meaning": (
                    "simultaneous standard-feed versus inserted random exposures; "
                    "not a before/after deployment switch"
                ),
            },
            "windows": {
                "source": ["2022-04-22", "2022-04-30"],
                "val": ["2022-05-01", "2022-05-03"],
                "test": ["2022-05-04", "2022-05-08"],
            },
            "history": {
                "construction": (
                    "chronologically first source-window user-item pair, then "
                    "outcome-blind deterministic per-user policy balancing"
                ),
                "scope": (
                    "full balanced source over every user observed under both "
                    "policies; not filtered to the target-user cohort"
                ),
                "target_user_filter_applied": False,
                "source_seed": self.source_seed,
                "max_items_per_user_per_policy": (
                    self.max_history_items_per_policy
                ),
                "feedback_column": self.history_click_column,
                "policies": {
                    policy: self._frame_summary(histories[policy])
                    for policy in POLICIES
                },
                "source_degree_audit": source_degree_audit,
                "balance": balance_audit,
                "permuted_control": {
                    "kind": (
                        "deterministic within-user cyclic item-feedback "
                        "permutation"
                    ),
                    "preserves": [
                        "history rows",
                        "history order",
                        "per-user click multiset",
                    ],
                    "assignment_change_audit": {
                        policy: self._permutation_audit(
                            histories[policy],
                            permuted_histories[policy],
                        )
                        for policy in POLICIES
                    },
                    "frame_sha256": {
                        policy: frame_hashes[f"{policy}_history_permuted"]
                        for policy in POLICIES
                    },
                },
                "no_history_control": "empty history frame with identical schema",
            },
            "support": {
                "balanced_source_users": int(
                    histories["standard"][self.user_column].nunique()
                ),
                "balanced_source_user_ids_sha256": _values_sha256(
                    sorted(
                        histories["standard"][self.user_column].unique(),
                        key=repr,
                    )
                ),
                "target_eligible_users": len(target_eligible_users),
                "target_eligible_user_ids_sha256": _values_sha256(
                    sorted(target_eligible_users, key=repr)
                ),
                "candidate_common_positive_items": len(common_warm_items),
                "candidate_common_positive_item_ids_sha256": _values_sha256(
                    sorted(common_warm_items, key=repr)
                ),
                "candidate_min_positive_user_degree_per_policy": (
                    self.min_positive_source_users_per_policy
                ),
                "candidate_support_source": (
                    "exact full balanced source histories stored in this task"
                ),
                "target_user_requirement": (
                    "positive and negative balanced tab-1 source labels under "
                    "both policies"
                ),
                "candidate_item_requirement": (
                    "minimum positive-user degree under both policies in the "
                    "exact full balanced source histories"
                ),
            },
            "candidate_construction": {
                "kind": "observed_c3_pseudo_list",
                "candidates_per_group": CANDIDATE_SIZE,
                "positives_per_group": POSITIVES_PER_GROUP,
                "negatives_per_group": CANDIDATE_SIZE - POSITIVES_PER_GROUP,
                "negative_definition": "observed non-click exposure",
                "missing_as_negative": False,
                "one_group_per_user_and_target_policy": True,
                "candidate_order": (
                    "deterministic seeded shuffle after label-stratified selection"
                ),
                "candidate_seed": self.candidate_seed,
                "simultaneous_logged_slate": False,
            },
            "pair_novelty": {
                "required": True,
                "definition": (
                    "target user-item pair absent before target-window start "
                    "under both exposure policies and every tab"
                ),
            },
            "metadata_visibility": {
                "history": [
                    "item_description",
                    "item_category_path",
                    self.history_click_column,
                ],
                "candidate": [
                    "item_description",
                    "item_category_path",
                ],
                "hidden_from_llm": [
                    self.user_column,
                    self.item_column,
                    self.candidate_group_column,
                    "history_policy",
                    "target_policy",
                    "event_date",
                    "event_timestamp_ms",
                    "interaction_id",
                    self.target_column,
                ],
                "semantic_coverage": semantic_audit,
            },
            "targets": target_audit,
            "shared_target_users": {
                split: {
                    "count": len(user_ids),
                    "user_ids_sha256": _values_sha256(user_ids),
                }
                for split, user_ids in shared_users_by_split.items()
            },
            "rows": {
                name: int(len(frame))
                for name, frame in frames_for_hash.items()
            },
            "users": {
                "canonical": int(users[self.user_column].nunique()),
                **{
                    name: int(frame[self.user_column].nunique())
                    for name, frame in frames_for_hash.items()
                },
            },
            "items": {
                "canonical": int(items[self.item_column].nunique()),
                "semantic_eligible": int(
                    semantic_items[self.item_column].nunique()
                ),
            },
            "hashes": {
                "canonical_manifest_sha256": canonical_manifest_sha256,
                "canonical_content_sha256": canonical_content_sha256,
                "frames": frame_hashes,
                "task_identity_sha256": task_identity_sha256,
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
                    "The c3 rows are separately observed exposures rather than "
                    "a simultaneous logged recommendation slate."
                ),
                (
                    "Conditioning every candidate group to one click evaluates "
                    "ranking, not natural policy CTR or calibration."
                ),
                (
                    "Standard and random target policies use the same users but "
                    "generally different observed candidate items."
                ),
                (
                    "The shared target cohort is selected using future target "
                    "outcomes: every retained user must have enough observed "
                    "clicks and non-clicks to form c3 under both policies."
                ),
                (
                    "Raw metric levels across standard and random target "
                    "columns are not comparable because their observed "
                    "candidate distributions differ; contrasts must remain "
                    "within one target-policy column."
                ),
                (
                    "KuaiRand-Pure omits standard-feed items outside the "
                    "intervention pool, so standard histories are incomplete."
                ),
                (
                    "The primary candidate-support threshold is one positive "
                    "source user per policy; a threshold of two is the "
                    "predeclared near-cold robustness condition."
                ),
            ],
        }

        return KuaiRandPolicyShiftTask(
            name=self.name,
            standard_history=histories["standard"],
            random_history=histories["random"],
            standard_history_permuted=permuted_histories["standard"],
            random_history_permuted=permuted_histories["random"],
            val_standard_targets=target_frames[("val", "standard")],
            val_random_targets=target_frames[("val", "random")],
            test_standard_targets=target_frames[("test", "standard")],
            test_random_targets=target_frames[("test", "random")],
            schema=schema,
            manifest=manifest,
        )

    def _prepare_interactions(self, interactions: pd.DataFrame) -> pd.DataFrame:
        rows = interactions.copy()
        rows["_date_key"] = _normalize_date_key(rows[self.date_column])
        rows["_policy"] = (
            rows[self.policy_column].astype(str).str.strip().str.lower()
        )
        unknown_policies = sorted(
            set(rows["_policy"].dropna().unique()) - set(POLICIES)
        )
        if unknown_policies:
            raise ValueError(
                f"Unsupported exposure policies: {unknown_policies}"
            )
        numeric_click = pd.to_numeric(rows[self.click_column], errors="coerce")
        if numeric_click.isna().any() or not set(numeric_click.unique()).issubset(
            {0, 1}
        ):
            raise ValueError(
                f"{self.click_column!r} must contain only binary 0/1 values"
            )
        rows[self.click_column] = numeric_click.astype("int8")
        numeric_tab = pd.to_numeric(rows[self.tab_column], errors="coerce")
        if numeric_tab.isna().any():
            raise ValueError(f"{self.tab_column!r} contains non-numeric values")
        rows[self.tab_column] = numeric_tab.astype(int)
        rows[self.timestamp_column] = pd.to_numeric(
            rows[self.timestamp_column],
            errors="raise",
        )
        return rows

    def _semantic_items(
        self,
        items: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        description = items[self.description_column].astype("string").str.strip()
        category = items[self.category_column].astype("string").str.strip()
        eligible = description.notna() & description.ne("")
        eligible &= category.notna() & category.ne("")
        semantic_items = items.loc[
            eligible,
            [self.item_column, self.description_column, self.category_column],
        ].copy()
        semantic_items = semantic_items.rename(
            columns={
                self.description_column: "item_description",
                self.category_column: "item_category_path",
            }
        )
        return semantic_items, {
            "require_description": True,
            "require_category_path": True,
            "items_total": int(len(items)),
            "items_eligible": int(eligible.sum()),
            "items_dropped": int((~eligible).sum()),
            "description_source_column": self.description_column,
            "category_source_column": self.category_column,
        }

    def _balance_source(
        self,
        source_by_policy: dict[PolicyRegime, pd.DataFrame],
    ) -> tuple[dict[PolicyRegime, pd.DataFrame], dict[str, Any]]:
        counts = pd.concat(
            {
                policy: frame.groupby(self.user_column, sort=False).size()
                for policy, frame in source_by_policy.items()
            },
            axis=1,
            join="inner",
        )
        per_user_counts = (
            counts.loc[:, list(POLICIES)]
            .min(axis=1)
            .clip(upper=self.max_history_items_per_policy)
            .astype(int)
        )
        per_user_counts = per_user_counts[per_user_counts.gt(0)]

        balanced: dict[PolicyRegime, pd.DataFrame] = {}
        for policy in POLICIES:
            namespace = f"source-balance-{policy}"
            rows = source_by_policy[policy][
                source_by_policy[policy][self.user_column].isin(
                    per_user_counts.index
                )
            ].copy()
            # The seeded key is independent of click outcome, so both cohort
            # formation and capping remain outcome-blind.
            rows["_source_sample_key"] = [
                _stable_int(
                    self.source_seed,
                    namespace,
                    user_id,
                    item_id,
                    interaction_id,
                )
                for user_id, item_id, interaction_id in zip(
                    rows[self.user_column],
                    rows[self.item_column],
                    rows[self.interaction_id_column],
                    strict=True,
                )
            ]
            rows = rows.sort_values(
                [
                    self.user_column,
                    "_source_sample_key",
                    self.item_column,
                    self.timestamp_column,
                    self.interaction_id_column,
                ],
                kind="mergesort",
            )
            sample_rank = rows.groupby(
                self.user_column,
                sort=False,
            ).cumcount()
            keep_limit = rows[self.user_column].map(per_user_counts)
            rows = rows[sample_rank.lt(keep_limit)].drop(
                columns=["_source_sample_key"]
            )
            # Sampling order is an implementation detail. The model must see
            # each user's selected events in chronological order.
            balanced[policy] = rows.sort_values(
                [
                    self.user_column,
                    self.timestamp_column,
                    self.item_column,
                    self.interaction_id_column,
                ],
                kind="mergesort",
            ).reset_index(drop=True)

        count_values = per_user_counts.tolist()
        return balanced, {
            "shared_source_users": int(len(counts)),
            "balanced_source_users": len(per_user_counts),
            "rows_per_policy": {
                policy: int(len(frame))
                for policy, frame in balanced.items()
            },
            "method": (
                "first user-item pair, deterministic seeded per-user sampling "
                "to min(standard_count, random_count, history_cap), without "
                "using outcome labels"
            ),
            "source_seed": self.source_seed,
            "max_history_items_per_policy": (
                self.max_history_items_per_policy
            ),
            "per_user_count_min": (
                int(min(count_values)) if count_values else 0
            ),
            "per_user_count_mean": (
                float(sum(count_values) / len(count_values))
                if count_values
                else 0.0
            ),
            "per_user_count_max": (
                int(max(count_values)) if count_values else 0
            ),
        }

    def _target_eligible_users(
        self,
        balanced: dict[PolicyRegime, pd.DataFrame],
    ) -> set[Any]:
        trainable: dict[PolicyRegime, set[Any]] = {}
        for policy, frame in balanced.items():
            label_range = frame.groupby(self.user_column)[self.click_column].agg(
                ["min", "max"]
            )
            trainable[policy] = set(
                label_range[
                    label_range["min"].eq(0) & label_range["max"].eq(1)
                ].index
            )
        return trainable["standard"] & trainable["random"]

    def _source_degree_summary(
        self,
        history: pd.DataFrame,
    ) -> dict[str, Any]:
        """Summarize degrees on the exact source graph given to baselines."""

        exposure_degree = history.groupby(self.item_column, sort=False).size()
        positive_degree = (
            history.loc[history[self.history_click_column].eq(1)]
            .groupby(self.item_column, sort=False)
            .size()
            .reindex(exposure_degree.index, fill_value=0)
        )
        return {
            "degree_unit": (
                "unique user-item source pairs; repeated pairs were reduced "
                "to their first source-window exposure"
            ),
            "source_exposures": int(len(history)),
            "source_positive_exposures": int(
                history[self.history_click_column].sum()
            ),
            "items_with_exposure": int(len(exposure_degree)),
            "items_with_positive_support": int(positive_degree.gt(0).sum()),
            "item_exposure_degree": _numeric_summary(exposure_degree),
            "item_positive_degree_including_zeros": _numeric_summary(
                positive_degree
            ),
        }

    def _build_target_split(
        self,
        *,
        tab1: pd.DataFrame,
        all_interactions: pd.DataFrame,
        split_name: PolicyShiftSplit,
        start: int,
        end: int,
        allowed_users: set[Any],
        allowed_items: set[Any],
    ) -> tuple[
        dict[PolicyRegime, pd.DataFrame],
        dict[str, Any],
        list[Any],
    ]:
        prior_pairs = (
            all_interactions.loc[
                all_interactions["_date_key"].lt(start),
                [self.user_column, self.item_column],
            ]
            .drop_duplicates()
            .assign(_seen_before_target=True)
        )
        target_window = tab1[tab1["_date_key"].between(start, end)].copy()
        pools: dict[
            PolicyRegime, dict[Any, dict[int, pd.DataFrame]]
        ] = {}
        pool_audit: dict[str, Any] = {}
        feasible_users: dict[PolicyRegime, set[Any]] = {}
        for policy in POLICIES:
            policy_rows = self._first_pair_rows(
                target_window[target_window["_policy"].eq(policy)].copy()
            )
            before_filters = len(policy_rows)
            policy_rows = policy_rows[
                policy_rows[self.user_column].isin(allowed_users)
                & policy_rows[self.item_column].isin(allowed_items)
            ].copy()
            warm_rows = len(policy_rows)
            policy_rows = policy_rows.merge(
                prior_pairs,
                on=[self.user_column, self.item_column],
                how="left",
                validate="many_to_one",
            )
            policy_rows = policy_rows[
                policy_rows["_seen_before_target"].isna()
            ].drop(columns=["_seen_before_target"])
            by_user: dict[Any, dict[int, pd.DataFrame]] = {}
            for user_id, user_rows in policy_rows.groupby(
                self.user_column,
                sort=False,
            ):
                by_user[user_id] = {
                    label: user_rows[user_rows[self.click_column].eq(label)].copy()
                    for label in (0, 1)
                }
            pools[policy] = by_user
            feasible_users[policy] = {
                user_id
                for user_id, labels in by_user.items()
                if len(labels[1]) >= POSITIVES_PER_GROUP
                and len(labels[0])
                >= CANDIDATE_SIZE - POSITIVES_PER_GROUP
            }
            pool_audit[policy] = {
                "first_pair_rows_before_support_filters": int(before_filters),
                "rows_after_user_item_warm_filter": int(warm_rows),
                "rows_after_pair_novelty": int(len(policy_rows)),
                "feasible_users_before_policy_intersection": int(
                    len(feasible_users[policy])
                ),
            }

        shared_users = sorted(
            feasible_users["standard"] & feasible_users["random"],
            key=repr,
        )
        if not shared_users:
            raise ValueError(
                f"No shared users can form observed c3 targets for split={split_name}"
            )
        targets = {
            policy: self._materialize_target_policy(
                pools[policy],
                shared_users=shared_users,
                split_name=split_name,
                policy=policy,
            )
            for policy in POLICIES
        }
        self._validate_target_pair(targets, split_name=split_name)
        audit = {
            "window": [
                _date_key_to_iso(start),
                _date_key_to_iso(end),
            ],
            "shared_users": len(shared_users),
            "shared_user_ids_sha256": _values_sha256(shared_users),
            "policies": {
                policy: {
                    **pool_audit[policy],
                    "groups": int(
                        targets[policy][self.candidate_group_column].nunique()
                    ),
                    "rows": int(len(targets[policy])),
                    "frame_sha256": _frame_sha256(targets[policy]),
                }
                for policy in POLICIES
            },
        }
        return targets, audit, shared_users

    def _materialize_target_policy(
        self,
        pool: dict[Any, dict[int, pd.DataFrame]],
        *,
        shared_users: list[Any],
        split_name: PolicyShiftSplit,
        policy: PolicyRegime,
    ) -> pd.DataFrame:
        groups: list[pd.DataFrame] = []
        for user_id in shared_users:
            positive = self._choose_rows(
                pool[user_id][1],
                n=POSITIVES_PER_GROUP,
                namespace=f"{split_name}-{policy}-c3-positive",
                user_id=user_id,
                seed=self.candidate_seed,
            )
            negative = self._choose_rows(
                pool[user_id][0],
                n=CANDIDATE_SIZE - POSITIVES_PER_GROUP,
                namespace=f"{split_name}-{policy}-c3-negative",
                user_id=user_id,
                seed=self.candidate_seed,
            )
            selected = pd.concat([positive, negative], ignore_index=True)
            positions = list(range(len(selected)))
            random.Random(
                _stable_int(
                    self.candidate_seed,
                    f"{split_name}-{policy}-c3-candidate-order",
                    user_id,
                )
            ).shuffle(positions)
            selected = selected.iloc[positions].copy().reset_index(drop=True)
            selected[self.candidate_group_column] = (
                f"{split_name}_{policy}_u{user_id}_c3"
            )
            selected["target_policy"] = policy
            groups.append(selected)
        selected_rows = pd.concat(groups, ignore_index=True)
        return self._target_frame(selected_rows)

    def _history_frame(
        self,
        rows: pd.DataFrame,
        *,
        policy: PolicyRegime,
    ) -> pd.DataFrame:
        frame = self._base_output_frame(rows)
        frame[self.history_click_column] = rows[self.click_column].to_numpy(
            dtype=int
        )
        frame[self.target_column] = frame[self.history_click_column].astype(int)
        frame[self.candidate_group_column] = pd.NA
        frame["history_policy"] = policy
        frame["target_policy"] = pd.NA
        return frame.loc[:, self._task_columns()].reset_index(drop=True)

    def _target_frame(self, rows: pd.DataFrame) -> pd.DataFrame:
        frame = self._base_output_frame(rows)
        frame[self.history_click_column] = pd.NA
        frame[self.target_column] = rows[self.click_column].to_numpy(dtype=int)
        frame[self.candidate_group_column] = rows[
            self.candidate_group_column
        ].to_numpy()
        frame["history_policy"] = pd.NA
        frame["target_policy"] = rows["target_policy"].to_numpy()
        return frame.loc[:, self._task_columns()].reset_index(drop=True)

    def _base_output_frame(self, rows: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                self.user_column: rows[self.user_column].to_numpy(),
                self.item_column: rows[self.item_column].to_numpy(),
                "item_description": rows["item_description"].to_numpy(),
                "item_category_path": rows["item_category_path"].to_numpy(),
                "event_date": rows["_date_key"].map(_date_key_to_iso).to_numpy(),
                "event_timestamp_ms": rows[self.timestamp_column].to_numpy(),
                "interaction_id": rows[self.interaction_id_column].to_numpy(),
            }
        )

    def _permuted_history(
        self,
        history: pd.DataFrame,
        *,
        namespace: str,
    ) -> pd.DataFrame:
        permuted = history.copy()
        label_values = permuted[self.history_click_column].to_numpy(
            dtype=int,
            copy=True,
        )
        for user_id, positions in permuted.groupby(
            self.user_column,
            sort=False,
        ).indices.items():
            labels = label_values[positions].tolist()
            if len(labels) < 2 or len(set(labels)) < 2:
                continue
            shifts = list(range(1, len(labels)))
            random.Random(
                _stable_int(self.source_seed, namespace, user_id)
            ).shuffle(shifts)
            shifted: list[int] | None = None
            for shift in shifts:
                candidate = labels[-shift:] + labels[:-shift]
                if candidate != labels:
                    shifted = candidate
                    break
            if shifted is None:
                raise AssertionError(
                    f"Could not permute mixed click labels for user {user_id!r}"
                )
            label_values[positions] = shifted
        permuted[self.history_click_column] = label_values
        permuted[self.target_column] = label_values
        return permuted

    def _permutation_audit(
        self,
        observed: pd.DataFrame,
        permuted: pd.DataFrame,
    ) -> dict[str, Any]:
        """Report how much a binary-label permutation actually changed.

        A cyclic shift preserves every user's label multiset, but repeated
        binary labels mean that some item-feedback associations can remain
        unchanged. Recording that retention makes the grounding control
        interpretable without changing the materialized task.
        """

        identity_columns = [
            self.user_column,
            self.item_column,
            self.interaction_id_column,
        ]
        if not observed[identity_columns].equals(permuted[identity_columns]):
            raise AssertionError(
                "Permutation control changed history row identities or order"
            )
        observed_labels = observed[self.history_click_column].astype(int)
        permuted_labels = permuted[self.history_click_column].astype(int)
        changed = observed_labels.ne(permuted_labels)
        mixed_users = (
            observed.groupby(self.user_column)[self.history_click_column]
            .nunique()
            .gt(1)
        )
        mixed_mask = observed[self.user_column].isin(
            mixed_users[mixed_users].index
        )
        mixed_rows = int(mixed_mask.sum())
        changed_mixed_rows = int((changed & mixed_mask).sum())
        return {
            "rows": int(len(observed)),
            "changed_rows": int(changed.sum()),
            "changed_fraction": (
                float(changed.mean()) if len(observed) else 0.0
            ),
            "mixed_label_users": int(mixed_users.sum()),
            "mixed_label_rows": mixed_rows,
            "changed_mixed_label_rows": changed_mixed_rows,
            "changed_mixed_label_fraction": (
                changed_mixed_rows / mixed_rows if mixed_rows else 0.0
            ),
        }

    def _first_pair_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        if rows.empty:
            return rows.copy()
        return (
            rows.sort_values(
                [
                    self.user_column,
                    self.timestamp_column,
                    self.item_column,
                    self.interaction_id_column,
                ],
                kind="mergesort",
            )
            .drop_duplicates(
                [self.user_column, self.item_column],
                keep="first",
            )
            .reset_index(drop=True)
        )

    def _choose_rows(
        self,
        rows: pd.DataFrame,
        *,
        n: int,
        namespace: str,
        user_id: Any,
        seed: int,
    ) -> pd.DataFrame:
        if len(rows) < n:
            raise ValueError(
                f"Cannot select {n} rows from {len(rows)} for user {user_id!r}"
            )
        stable = rows.sort_values(
            [
                self.item_column,
                self.timestamp_column,
                self.interaction_id_column,
            ],
            kind="mergesort",
        )
        if len(stable) == n:
            return stable.copy()
        positions = random.Random(
            _stable_int(seed, namespace, user_id)
        ).sample(range(len(stable)), n)
        return stable.iloc[sorted(positions)].copy()

    def _validate_balanced_histories(
        self,
        balanced: dict[PolicyRegime, pd.DataFrame],
    ) -> None:
        counts = {
            policy: frame.groupby(self.user_column).size().sort_index()
            for policy, frame in balanced.items()
        }
        if not counts["standard"].equals(counts["random"]):
            raise AssertionError(
                "Standard and random histories are not balanced per user"
            )

    def _validate_target_pair(
        self,
        targets: dict[PolicyRegime, pd.DataFrame],
        *,
        split_name: PolicyShiftSplit,
    ) -> None:
        users = {
            policy: set(frame[self.user_column].unique())
            for policy, frame in targets.items()
        }
        if users["standard"] != users["random"]:
            raise AssertionError(
                f"Target policy users differ for split={split_name}"
            )
        for policy, frame in targets.items():
            grouped = frame.groupby(self.candidate_group_column, sort=False)
            sizes = grouped.size()
            positives = grouped[self.target_column].sum()
            users_per_group = grouped[self.user_column].nunique()
            if not sizes.eq(CANDIDATE_SIZE).all():
                raise AssertionError(
                    f"{split_name}/{policy} contains non-c3 groups"
                )
            if not positives.eq(POSITIVES_PER_GROUP).all():
                raise AssertionError(
                    f"{split_name}/{policy} groups must contain one click"
                )
            if not users_per_group.eq(1).all():
                raise AssertionError(
                    f"{split_name}/{policy} group contains multiple users"
                )
            if frame[self.history_click_column].notna().any():
                raise AssertionError(
                    "Target rows expose history_click and would leak labels"
                )

    def _frame_summary(self, frame: pd.DataFrame) -> dict[str, Any]:
        per_user = frame.groupby(self.user_column).size()
        return {
            "rows": int(len(frame)),
            "users": int(frame[self.user_column].nunique()),
            "items": int(frame[self.item_column].nunique()),
            "click_rate": float(frame[self.history_click_column].mean()),
            "rows_per_user_min": int(per_user.min()),
            "rows_per_user_mean": float(per_user.mean()),
            "rows_per_user_max": int(per_user.max()),
            "frame_sha256": _frame_sha256(frame),
        }

    def _task_columns(self) -> list[str]:
        return [
            self.user_column,
            self.item_column,
            "item_description",
            "item_category_path",
            self.history_click_column,
            self.target_column,
            self.candidate_group_column,
            "history_policy",
            "target_policy",
            "event_date",
            "event_timestamp_ms",
            "interaction_id",
        ]

    def _validate_unique_items(self, items: pd.DataFrame) -> None:
        if items[self.item_column].duplicated().any():
            raise ValueError(
                f"items.{self.item_column} must be unique for semantic joins"
            )

    @staticmethod
    def _require_columns(
        frame: pd.DataFrame,
        columns: list[str],
        *,
        table: str,
    ) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{table} is missing required columns: {missing}")


def _canonical_content_identity(
    manifest: dict[str, Any],
    *,
    dataset_name: str,
    dataset_version: str,
    table_rows: dict[str, int],
) -> dict[str, Any]:
    """Return a rebuild-stable fingerprint payload for canonical content.

    Canonical manifests also contain instance metadata such as ``created_at``
    and absolute local paths. Those remain auditable in the full manifest hash
    but must not change scientific task identity after an equivalent rebuild.
    """

    raw_sha256 = sorted(
        {
            str(source["sha256"])
            for source in manifest.get("raw_sources", [])
            if isinstance(source, dict) and source.get("sha256")
        }
    )
    semantic_sha256 = sorted(
        _collect_sha256_values(manifest.get("semantic_supplement"))
    )
    return {
        "dataset": dataset_name,
        "version": dataset_version,
        "schema_version": manifest.get("schema_version"),
        "table_rows": {
            name: int(count)
            for name, count in sorted(table_rows.items())
        },
        "raw_source_sha256": raw_sha256,
        "semantic_source_sha256": semantic_sha256,
    }


def _collect_sha256_values(payload: Any) -> set[str]:
    if isinstance(payload, dict):
        values: set[str] = set()
        for key, value in payload.items():
            if (
                isinstance(value, str)
                and (key == "sha256" or key.endswith("_sha256"))
            ):
                values.add(value)
            else:
                values.update(_collect_sha256_values(value))
        return values
    if isinstance(payload, list):
        values = set()
        for value in payload:
            values.update(_collect_sha256_values(value))
        return values
    return set()


def _normalize_date_key(values: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(values.dtype):
        keys = values.astype("int64")
        unique_keys = pd.Series(keys.unique())
        unique_text = unique_keys.astype("string")
        valid_shape = unique_text.str.fullmatch(r"\d{8}", na=False)
        parsed_unique = pd.to_datetime(
            unique_text,
            format="%Y%m%d",
            errors="coerce",
        )
        if valid_shape.all() and parsed_unique.notna().all():
            return keys.astype(int)
    text = values.astype("string").str.strip()
    compact = text.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(
            text.loc[compact],
            format="%Y%m%d",
            errors="coerce",
        )
    if (~compact).any():
        parsed.loc[~compact] = pd.to_datetime(
            text.loc[~compact],
            errors="coerce",
        )
    if parsed.isna().any():
        examples = values.loc[parsed.isna()].head(3).tolist()
        raise ValueError(f"Could not parse date values: {examples}")
    return parsed.dt.strftime("%Y%m%d").astype(int)


def _date_key_to_iso(value: int) -> str:
    text = f"{int(value):08d}"
    return f"{text[:4]}-{text[4:6]}-{text[6:]}"


def _stable_int(seed: int, *parts: object) -> int:
    payload = "::".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(
        hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest(),
        "big",
    )


def _values_sha256(values: list[Any]) -> str:
    payload = "\n".join(repr(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _numeric_summary(values: pd.Series) -> dict[str, int | float]:
    if values.empty:
        return {
            "count": 0,
            "min": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0,
        }
    return {
        "count": int(len(values)),
        "min": int(values.min()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "max": int(values.max()),
    }


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    csv_payload = frame.to_csv(
        index=False,
        lineterminator="\n",
        na_rep="<NA>",
    ).encode("utf-8")
    return hashlib.sha256(csv_payload).hexdigest()


def _validate_choice(
    name: str,
    value: str,
    choices: tuple[str, ...],
) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
