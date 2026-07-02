"""Task construction contracts."""

from beyond_click_sim.tasks.alignment import AlignmentInteractionTaskBuilder
from beyond_click_sim.tasks.cold_start import (
    ColdStartSplitFrames,
    ColdStartTask,
    ColdStartTaskBuilder,
    ColdUserHoldoutSplitter,
)
from beyond_click_sim.tasks.base import (
    CandidateSampler,
    DatasetFilter,
    ItemFeatureBuilder,
    SplitFrames,
    Splitter,
    Task,
    TaskBuilder,
    TaskSchema,
    TaskSplit,
    split_xy,
)
from beyond_click_sim.tasks.filters import (
    MinUserInteractionsFilter,
    SampleUsersFilter,
    SequentialDatasetFilter,
)
from beyond_click_sim.tasks.item_statistics import (
    PREFIXED_ITEM_RATING_COUNT_COLUMN,
    PREFIXED_ITEM_RATING_MEAN_COLUMN,
    PREFIXED_ITEM_RATING_STATS_COLUMNS,
    TrainItemRatingStatistics,
)
from beyond_click_sim.tasks.policies import Policy, PopularityPolicy, RandomPolicy
from beyond_click_sim.tasks.policy_ranking import PolicyRankingTaskBuilder
from beyond_click_sim.tasks.regression import RegressionPredictionTaskBuilder
from beyond_click_sim.tasks.samplers import (
    CappedUserInteractionCandidateSampler,
    NonInteractionCandidateSampler,
    PostSplitUserSampler,
)
from beyond_click_sim.tasks.splitters import RandomFractionSplitter

__all__ = [
    "AlignmentInteractionTaskBuilder",
    "CandidateSampler",
    "CappedUserInteractionCandidateSampler",
    "ColdStartSplitFrames",
    "ColdStartTask",
    "ColdStartTaskBuilder",
    "ColdUserHoldoutSplitter",
    "DatasetFilter",
    "ItemFeatureBuilder",
    "MinUserInteractionsFilter",
    "NonInteractionCandidateSampler",
    "Policy",
    "PolicyRankingTaskBuilder",
    "PopularityPolicy",
    "PostSplitUserSampler",
    "PREFIXED_ITEM_RATING_COUNT_COLUMN",
    "PREFIXED_ITEM_RATING_MEAN_COLUMN",
    "PREFIXED_ITEM_RATING_STATS_COLUMNS",
    "RandomFractionSplitter",
    "RandomPolicy",
    "RegressionPredictionTaskBuilder",
    "SampleUsersFilter",
    "SequentialDatasetFilter",
    "SplitFrames",
    "Splitter",
    "Task",
    "TaskBuilder",
    "TaskSchema",
    "TaskSplit",
    "TrainItemRatingStatistics",
    "split_xy",
]
