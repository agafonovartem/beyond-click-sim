from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.ood.kuairand_policy_shift.nested_support_audit import (
    compute_nested_eligibility,
)


def test_nested_eligibility_requires_both_policies_and_every_seed() -> None:
    targets = _targets()
    all_items = targets["item_id"].unique()
    standard_degree = pd.Series(2, index=all_items, dtype=int)
    random_degree = pd.Series(2, index=all_items, dtype=int)
    # u2 passes seed 42 but one candidate fails random-source support in seed 43.
    random_degree.loc["43-random-u2-c3"] = 1

    candidates, user_seed, common = compute_nested_eligibility(
        targets,
        positive_degrees={
            "standard": standard_degree,
            "random": random_degree,
        },
        min_positive_degree=2,
    )

    assert common == ["u1"]
    assert len(candidates) == 24
    assert len(user_seed) == 4
    status = user_seed.set_index(["task_seed", "user_id"])[
        "passes_both_target_policies"
    ]
    assert bool(status.loc[(42, "u2")]) is True
    assert bool(status.loc[(43, "u2")]) is False
    assert user_seed.groupby("user_id")[
        "in_common_cross_seed_cohort"
    ].all().to_dict() == {"u1": True, "u2": False}


def _targets() -> pd.DataFrame:
    rows = []
    for task_seed in (42, 43):
        for user_id in ("u1", "u2"):
            for target_policy in ("standard", "random"):
                for position in (1, 2, 3):
                    rows.append(
                        {
                            "task_seed": task_seed,
                            "user_id": user_id,
                            "candidate_group": (
                                f"{task_seed}-{target_policy}-{user_id}"
                            ),
                            "target_policy": target_policy,
                            "item_id": (
                                f"{task_seed}-{target_policy}-{user_id}-"
                                f"c{position}"
                            ),
                        }
                    )
    return pd.DataFrame(rows)
