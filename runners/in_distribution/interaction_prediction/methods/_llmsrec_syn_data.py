"""Recover LLMSRec-Syn's event timestamps without changing materialized Tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from beyond_click_sim.data.canonical import CanonicalDataset, sha256_file
from beyond_click_sim.scorers.history.llmsrec_syn_retrieval import digest
from beyond_click_sim.tasks import (
    MinUserInteractionsFilter,
    RandomFractionSplitter,
    Task,
    TrainItemRatingStatistics,
)
from beyond_click_sim.tasks.base import TaskBuilder
from beyond_click_sim.tasks.filters import SampleUsersFilter, SequentialDatasetFilter


@dataclass(frozen=True)
class PreparedTraining:
    events: pd.DataFrame
    catalog: pd.DataFrame
    provenance: dict[str, Any]


def require_temporal_task(task: Task) -> None:
    if task.manifest.get("dataset") not in ("ml-1m", "steam"):
        raise ValueError(
            "LLMSRec-Syn supports temporal ml-1m and steam interaction tasks"
        )
    split = task.manifest.get("splitter", {})
    if (
        split.get("class") != "RandomFractionSplitter"
        or split.get("shuffle") is not False
        or split.get("group_column") != "user_id"
        or not isinstance(split.get("timestamp_column"), str)
        or not split["timestamp_column"]
    ):
        raise ValueError(
            "LLMSRec-Syn requires an explicit per-user temporal splitter manifest"
        )
    if task.manifest.get("target_source_column") != "target_interact":
        raise ValueError("LLMSRec-Syn runner requires the target_interact task")
    if task.schema.id_columns != ("user_id", "item_id"):
        raise ValueError("LLMSRec-Syn requires user_id/item_id task identities")
    if not task.schema.candidate_group_column or not task.schema.sampled_column:
        raise ValueError(
            "LLMSRec-Syn requires candidate-group and sampled-column metadata"
        )
    if task.train.empty or not task.train.index.is_unique:
        raise ValueError("Training rows must be nonempty with unique indices")


def _filter(config: dict[str, Any]):
    options = dict(config)
    name = options.pop("class", None)
    if name == "SequentialDatasetFilter":
        return SequentialDatasetFilter([_filter(c) for c in options["filters"]])
    classes = {
        "MinUserInteractionsFilter": MinUserInteractionsFilter,
        "SampleUsersFilter": SampleUsersFilter,
    }
    if name not in classes:
        raise ValueError(f"Unsupported filter for timestamp reconstruction: {name}")
    return classes[name](**options)


def prepare_training(task: Task, dataset: CanonicalDataset) -> PreparedTraining:
    """Verify the entire train projection; preserve occurrence order for repeats.

    Read source interactions only to recreate the recorded split. Only its train
    events leave this helper. Never reconstruct or replace evaluation candidates.
    """
    require_temporal_task(task)
    if dataset.name != task.manifest["dataset"] or dataset.version != task.manifest.get(
        "dataset_version"
    ):
        raise ValueError("Canonical dataset name/version does not match the task")
    paths = {
        name: getattr(dataset, f"{name}_path")
        for name in ("users", "items", "interactions", "manifest")
    }
    stats_before = {
        name: (path.stat().st_size, path.stat().st_mtime_ns)
        for name, path in paths.items()
    }
    fingerprints = {name: sha256_file(path) for name, path in paths.items()}
    users, items, interactions = (
        dataset.load_users(),
        dataset.load_items(),
        dataset.load_interactions(),
    )
    for frame, column in ((users, "user_id"), (items, "item_id")):
        if frame[column].isna().any() or frame[column].duplicated().any():
            raise ValueError(f"Canonical {column} must be unique and non-null")
    if "_llmsrec_event_id" in interactions:
        raise ValueError("Reserved LLMSRec-Syn provenance column in canonical data")
    interactions = interactions.copy()
    if "interaction_id" in interactions:
        if (
            interactions.interaction_id.isna().any()
            or interactions.interaction_id.duplicated().any()
        ):
            raise ValueError("Canonical interaction_id must be unique and non-null")
        interactions["_llmsrec_event_id"] = interactions.interaction_id
        event_identity = "canonical_interaction_id"
    else:
        interactions["_llmsrec_event_id"] = [
            f"canonical_row:{i}" for i in range(len(interactions))
        ]
        event_identity = "canonical_row_position_with_source_fingerprint"
    users, items, interactions = _filter(task.manifest["filter"]).filter(
        users=users,
        items=items,
        interactions=interactions,
    )
    split_options = {
        key: value for key, value in task.manifest["splitter"].items() if key != "class"
    }
    train = RandomFractionSplitter(**split_options).split(interactions).train
    catalog = items[["item_id", "title"]].rename(columns={"title": "item_title"}).copy()
    feature_config = task.manifest.get("item_feature_builder")
    if feature_config is not None:
        if feature_config.get("class") != "TrainItemRatingStatistics":
            raise ValueError(
                "Unsupported item feature builder for train projection verification"
            )
        builder = TrainItemRatingStatistics(
            **{
                key: feature_config[key]
                for key in ("value_column", "mean_column", "count_column")
            }
        )
        items, _ = builder.enrich_items(
            items=items, train_interactions=train, item_column="item_id"
        )
    context = list(task.schema.history_context_columns)
    if context != task.manifest.get("history_context_columns"):
        raise ValueError("Task history context disagrees with its manifest")
    rows = train[["user_id", "item_id", *context]].copy()
    rows[task.schema.target_column] = train[task.manifest["target_source_column"]]
    rows[task.schema.sampled_column] = False
    rows[task.schema.candidate_group_column] = pd.NA
    projection = TaskBuilder._with_features(
        rows=rows,
        user_features=TaskBuilder._prefixed_features(users, "user_id", "user"),
        item_features=TaskBuilder._prefixed_features(items, "item_id", "item"),
        columns=list(task.train.columns),
        user_column="user_id",
        item_column="item_id",
    )
    try:
        pd.testing.assert_frame_equal(
            projection.reset_index(drop=True),
            task.train.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise ValueError(
            "Canonical training projection does not match the materialized task"
        ) from error
    if any(
        (p.stat().st_size, p.stat().st_mtime_ns) != stats_before[name]
        for name, p in paths.items()
    ):
        raise ValueError(
            "Canonical source changed while constructing timestamp mapping"
        )
    timestamp = split_options["timestamp_column"]
    events = (
        train[["user_id", "item_id", "_llmsrec_event_id", timestamp]]
        .rename(
            columns={"_llmsrec_event_id": "event_id", timestamp: "timestamp"},
        )
        .copy()
    )
    events.index = task.train.index.copy()
    provenance = {
        "canonical_name": dataset.name,
        "canonical_version": dataset.version,
        "canonical_file_sha256": fingerprints,
        "event_identity": event_identity,
        "mapping": "exact_train_projection_and_stable_event_occurrence_order",
        "training_rows": len(events),
        "timestamp_column": timestamp,
        "training_fingerprint": digest(
            {
                "sources": fingerprints,
                "filter": task.manifest["filter"],
                "splitter": task.manifest["splitter"],
            }
        ),
    }
    return PreparedTraining(events=events, catalog=catalog, provenance=provenance)
