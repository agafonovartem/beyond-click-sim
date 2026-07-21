from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.in_distribution.llm_error_budget import (  # noqa: E402
    LLMErrorRateExceededError,
    check_error_budget,
)


def _fake_errors(n: int) -> list[dict[str, object]]:
    return [
        {
            "candidate_group": f"g{idx}",
            "attempts": 1,
            "errors": [f"failure_{idx}"],
        }
        for idx in range(n)
    ]


def test_check_error_budget_noop_before_min_groups_without_force() -> None:
    check_error_budget(
        errors=_fake_errors(2),
        attempted=10,
        total=100,
        method_name="m",
        task_name="t",
        max_error_rate=0.10,
        min_groups_before_check=20,
        force=False,
    )


def test_check_error_budget_raises_when_threshold_exceeded_after_min_groups() -> None:
    with pytest.raises(LLMErrorRateExceededError) as raised:
        check_error_budget(
            errors=_fake_errors(3),
            attempted=20,
            total=100,
            method_name="m",
            task_name="t",
            max_error_rate=0.10,
            min_groups_before_check=20,
            force=False,
        )

    error = raised.value
    assert error.attempted == 20
    assert error.total == 100
    assert len(error.errors) == 3
    assert "15.0% (3/20 groups attempted out of 100 total)" in str(error)
    assert "candidate_group='g0'" in str(error)


def test_check_error_budget_force_ignores_min_groups_gate() -> None:
    with pytest.raises(LLMErrorRateExceededError):
        check_error_budget(
            errors=_fake_errors(2),
            attempted=3,
            total=3,
            method_name="m",
            task_name="t",
            max_error_rate=0.10,
            min_groups_before_check=20,
            force=True,
        )
