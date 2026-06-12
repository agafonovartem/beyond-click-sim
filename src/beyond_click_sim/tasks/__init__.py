"""Task construction contracts."""

from beyond_click_sim.tasks.alignment import AlignmentInteractionTaskBuilder
from beyond_click_sim.tasks.base import (
    CandidateSampler,
    DatasetFilter,
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
    "DatasetFilter",
    "MinUserInteractionsFilter",
    "NonInteractionCandidateSampler",
    "PostSplitUserSampler",
    "RandomFractionSplitter",
    "RegressionPredictionTaskBuilder",
    "SampleUsersFilter",
    "SequentialDatasetFilter",
    "SplitFrames",
    "Splitter",
    "Task",
    "TaskBuilder",
    "TaskSchema",
    "TaskSplit",
    "split_xy",
]
