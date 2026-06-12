from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from beyond_click_sim.tasks import Task

from runners.in_distribution.regression_prediction.methods.mean import run as run_mean


MethodRunner = Callable[[Task, Path], dict[str, object]]

METHOD_RUNNERS: dict[str, MethodRunner] = {
    "mean_regressor": run_mean,
}
