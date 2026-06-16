from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal

import pandas as pd

from beyond_click_sim.data.canonical import CanonicalDataset


TaskSplit = Literal["train", "val", "test"]
# TargetType = [
#     "interaction",# interacted or not
#     "preference", # liked or not
#     "regression", # rating, playtime, ...
# ]

# TaskFormat = [
#     "pointwise", # alignment logic (like Agent4Rec, SimUser): is LLM able to guess user behaviour 
#     "ranking",  # ranking logic (like AgentRecBench): is LLM able to rank items correctly?
# ] 


@dataclass(frozen=True)
class SplitFrames:
    """Named train/validation/test interaction splits."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class TaskSchema:
    """Column contract for task split dataframes.

    `feature_columns` is only a default/model-agnostic suggestion. Predictors
    may use ids, metadata columns, histories, or their own preprocessing.
    """

    target_column: str = "target"
    feature_columns: tuple[str, ...] = ()
    id_columns: tuple[str, ...] = ("user_id", "item_id")
    candidate_group_column: str | None = None  # for cases where we have candidate sets
    sampled_column: str | None = None  # for cases where we sampled negatives for training, but for an LLM we still need only real history (positives)
    # Interaction-side columns visible only for observed train-history rows.
    # Examples: rating, playtime_forever, target_like_ge4. Task builders should
    # set these columns to missing in val/test candidates to avoid feedback leakage.
    history_context_columns: tuple[str, ...] = ()


@dataclass
class Task:
    """Materialized task splits ready for predictors and evaluators."""

    name: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    schema: TaskSchema
    manifest: dict[str, Any] = field(default_factory=dict)


def split_xy(
    frame: pd.DataFrame,
    *,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a task dataframe into independent model inputs and targets."""

    if target_column not in frame.columns:
        raise ValueError(f"Missing target column: {target_column!r}")
    return frame.drop(columns=[target_column]).copy(), frame[target_column].copy()


class DatasetFilter(ABC):
    """Dataset-level row/user/item filter used before splitting."""

    @abstractmethod
    def filter(
        self,
        users: pd.DataFrame,
        items: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return filtered users, items, and interactions."""


class Splitter(ABC):
    """Split filtered interaction rows into train/validation/test parts."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    @abstractmethod
    def split(
        self,
        interactions: pd.DataFrame,
    ) -> SplitFrames:
        """Return train, validation, and test interaction rows."""


class CandidateSampler(ABC):
    """Build candidate rows from held-out observed interactions.

    Candidate sets can be used for pointwise alignment/classification or for
    ranking metrics. A sampler may add unobserved negatives, produce full-catalog
    candidates, or construct any other explicit candidate table needed by a task.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    @abstractmethod
    def sample(
        self,
        positives: pd.DataFrame,
        *,
        interactions: pd.DataFrame,
        items: pd.DataFrame,
        excluded_pairs: set[tuple[Any, Any]] | None = None,
    ) -> pd.DataFrame:
        """Return candidate rows for one held-out split."""


class ItemFeatureBuilder(ABC):
    """Build split-dependent item features from train interactions only."""

    @abstractmethod
    def enrich_items(
        self,
        *,
        items: pd.DataFrame,
        train_interactions: pd.DataFrame,
        item_column: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Return items with extra feature columns and a manifest block."""


class TaskBuilder(ABC):
    """Build a materialized task from a canonical dataset."""

    def __init__(
        self,
        *,
        name: str,
        target_source_column: str,
        dataset_filter: DatasetFilter | None = None,
        splitter: Splitter | None = None,
        sampler: CandidateSampler | None = None,
        target_column: str = "target",
        user_column: str = "user_id",
        item_column: str = "item_id",
        sampled_column: str | None = None,  # Boolean column. False = real observed row, True = synthetic sampled row.
        candidate_group_column: str | None = None,
        # Extra interaction columns available to history-aware models only for
        # observed train rows. They are not generic item/user features.
        history_context_columns: tuple[str, ...] = (),
        item_feature_builder: ItemFeatureBuilder | None = None,
    ) -> None:
        self.name = name
        self.target_source_column = target_source_column
        self.dataset_filter = dataset_filter
        self.splitter = splitter
        self.sampler = sampler
        self.target_column = target_column
        self.user_column = user_column
        self.item_column = item_column
        self.sampled_column = sampled_column
        self.candidate_group_column = candidate_group_column
        self.history_context_columns = history_context_columns
        self.item_feature_builder = item_feature_builder

    @abstractmethod
    def build(self, dataset: CanonicalDataset) -> Task:
        """Return train/validation/test task dataframes."""

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
        """Fail fast when an input dataframe misses required columns."""

        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    @staticmethod
    def _prefixed_features(
        frame: pd.DataFrame,
        id_column: str,
        prefix: str,
    ) -> pd.DataFrame:
        """Prefix non-id columns before joining user/item features."""

        rename = {
            column: f"{prefix}_{column}"
            for column in frame.columns
            if column != id_column
        }
        return frame.rename(columns=rename)

    @staticmethod
    def _feature_columns(
        user_features: pd.DataFrame,
        item_features: pd.DataFrame,
        *,
        user_column: str,
        item_column: str,
    ) -> tuple[str, ...]:
        """Return model-visible feature columns after user/item prefixing."""

        return tuple(
            column
            for column in [*user_features.columns, *item_features.columns]
            if column not in {user_column, item_column}
        )

    @staticmethod
    def _with_features(
        *,
        rows: pd.DataFrame,
        user_features: pd.DataFrame,
        item_features: pd.DataFrame,
        columns: list[str],
        user_column: str,
        item_column: str,
    ) -> pd.DataFrame:
        """Join task rows with user/item features and enforce column order."""

        if rows.empty:
            return pd.DataFrame(columns=columns)

        merged = rows.merge(user_features, on=user_column, how="left")
        merged = merged.merge(item_features, on=item_column, how="left")
        return merged.loc[:, columns]

    def _enrich_items_from_train(
        self,
        *,
        items: pd.DataFrame,
        train_interactions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any] | None]:
        """Apply optional train-only item feature construction."""

        if self.item_feature_builder is None:
            return items, None
        return self.item_feature_builder.enrich_items(
            items=items,
            train_interactions=train_interactions,
            item_column=self.item_column,
        )

    @staticmethod
    def _component_manifest(component: object) -> dict[str, Any]:
        """Serialize a filter/splitter/sampler config for reproducibility."""

        payload: dict[str, Any] = {"class": component.__class__.__name__}
        if is_dataclass(component):
            raw_payload = asdict(component)
        elif hasattr(component, "__dict__"):
            raw_payload = vars(component)
        else:
            raw_payload = {}
        payload.update(
            {
                key: TaskBuilder._manifest_value(value)
                for key, value in raw_payload.items()
            }
        )
        return payload

    @staticmethod
    def _manifest_value(value: Any) -> Any:
        """Convert nested task components into JSON-friendly values."""

        task_component_types = (
            DatasetFilter,
            Splitter,
            CandidateSampler,
            ItemFeatureBuilder,
        )
        if isinstance(value, task_component_types):
            return TaskBuilder._component_manifest(value)
        if isinstance(value, tuple):
            return [TaskBuilder._manifest_value(item) for item in value]
        if isinstance(value, list):
            return [TaskBuilder._manifest_value(item) for item in value]
        if isinstance(value, dict):
            return {
                key: TaskBuilder._manifest_value(item)
                for key, item in value.items()
            }
        return value
