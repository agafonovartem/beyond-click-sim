"""Tests for the preference-target policy ranking agreement protocol.

The preference variant must do two things together: score policies against a
positive-preference ground truth (`target_like_ge4` / `target_played_120`)
rather than the raw interaction indicator, *and* ask the simulator the matching
preference question. Grading preference labels while eliciting an interaction
answer would make the agreement number uninterpretable, so the prompt
assertions below are the core of this file.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from beyond_click_sim.tasks import Task, TaskSchema

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from beyond_click_sim.scorers import LLMPreferenceYesNoScorer  # noqa: E402
from runners.in_distribution.policy_ranking_agreement.task_builders import (  # noqa: E402
    TASK_BUILDERS as INTERACTION_TASK_BUILDERS,
)
from runners.in_distribution.policy_ranking_agreement_preference.methods.llm_yes_no import (  # noqa: E402
    TARGET_DESCRIPTIONS,
)
from runners.in_distribution.policy_ranking_agreement_preference.methods.popularity import (  # noqa: E402
    run,
)
from runners.in_distribution.policy_ranking_agreement_preference.metrics import (  # noqa: E402
    MAIN_METRIC,
    METRICS_FILENAME,
)
from runners.in_distribution.policy_ranking_agreement_preference.task_builders import (  # noqa: E402
    DATASET_TARGET_SOURCE_COLUMNS,
    TASK_BUILDERS,
)


# ---------------------------------------------------------------------------
# Helpers (same FakeClient pattern as test_policy_ranking_llm_dedup.py)
# ---------------------------------------------------------------------------


class FakeChatCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response))]
        )


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        completions = FakeChatCompletions(responses)
        self.chat = SimpleNamespace(completions=completions)
        self.completions = completions


def _fitted_preference_scorer(
    client: FakeClient,
    *,
    prompt_style: str,
) -> LLMPreferenceYesNoScorer:
    X_train = pd.DataFrame({"user_id": ["u1"], "item_title": ["Alien"]})
    scorer = LLMPreferenceYesNoScorer(
        target_description=TARGET_DESCRIPTIONS["ml-1m"],
        client=client,
        model="fake",
        item_description_columns=("item_title",),
        candidate_group_column="_llm_group_",
        prompt_style=prompt_style,
    )
    scorer.fit(X_train, pd.Series([1]))
    return scorer


def _prompt_text(client: FakeClient) -> str:
    messages = client.completions.calls[0]["messages"]
    return "\n".join(message["content"] for message in messages)


def _make_preference_task() -> Task:
    """Minimal policy-ranking Task carrying a positive-preference target.

    Train deliberately covers the test items with differing preference-positive
    counts, so `PopularityScorer` assigns them distinct scores and the simulated
    utilities are not constant across policies.

    All three policies cover u1 and u2; u3 is covered only by RandomPolicy and
    PopularityPolicy, so common-coverage utility aggregation must exclude u3.
    """
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"],
            "item_id": ["i1", "i1", "i1", "i2", "i3", "i3", "i4", "i5"],
            # Unlike target_interact, a preference target is not all-ones.
            "target": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        }
    )
    test = pd.DataFrame(
        {
            "user_id": [
                "u1", "u1", "u2", "u2",
                "u1", "u1", "u2", "u2",
                "u1", "u1", "u2", "u2",
                "u3", "u3",
                "u3", "u3",
            ],
            "item_id": [
                "i1", "i2", "i2", "i3",
                "i3", "i4", "i1", "i5",
                "i5", "i6", "i4", "i6",
                "i1", "i6",
                "i2", "i4",
            ],
            "target": [
                1.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 1.0, 0.0,
                1.0, 0.0, 0.0, 0.0,
                1.0, 1.0,
                1.0, 1.0,
            ],
            "policy": [
                "RandomPolicy", "RandomPolicy", "RandomPolicy", "RandomPolicy",
                "PopularityPolicy", "PopularityPolicy",
                "PopularityPolicy", "PopularityPolicy",
                "ItemKNNPolicy", "ItemKNNPolicy",
                "ItemKNNPolicy", "ItemKNNPolicy",
                "RandomPolicy", "RandomPolicy",
                "PopularityPolicy", "PopularityPolicy",
            ],
            "rank": [1, 2] * 8,
        }
    )
    return Task(
        name="toy_policy_ranking_preference_seed0",
        train=train,
        val=pd.DataFrame(columns=train.columns),
        test=test,
        schema=TaskSchema(
            target_column="target",
            feature_columns=("user_id", "item_id"),
        ),
        manifest={
            "protocol": "policy_ranking",
            "dataset": "ml-1m",
            "target_source_column": "target_like_ge4",
            "policy_recommendation_metrics": {
                "protocol": "held_out_recommendation_eval",
                "policies": [{"policy": "RandomPolicy", "mean_hit_rate": 0.5}],
            },
        },
    )


# ---------------------------------------------------------------------------
# Target definition and task registry
# ---------------------------------------------------------------------------


def test_dataset_targets_are_preference_not_interaction() -> None:
    assert DATASET_TARGET_SOURCE_COLUMNS == {
        "ml-1m": "target_like_ge4",
        "steam": "target_played_120",
    }
    assert "target_interact" not in DATASET_TARGET_SOURCE_COLUMNS.values()


def test_task_names_do_not_collide_with_interaction_protocol() -> None:
    """Preference and interaction task registries must be disjoint.

    Both runners write per-task artifacts keyed by task name; a shared key would
    let one protocol's outputs be mistaken for the other's.
    """
    assert set(TASK_BUILDERS) & set(INTERACTION_TASK_BUILDERS) == set()
    assert all("_policy_ranking_preference" in name for name in TASK_BUILDERS)


def test_target_descriptions_match_preference_thresholds() -> None:
    """Prompt wording must state the same thresholds the labels encode."""
    assert "at least 4 out of 5" in TARGET_DESCRIPTIONS["ml-1m"]
    assert "at least 120 minutes" in TARGET_DESCRIPTIONS["steam"]


# ---------------------------------------------------------------------------
# Elicited question — batch and itemwise
# ---------------------------------------------------------------------------


def test_batch_prompt_asks_the_preference_question() -> None:
    client = FakeClient(["C1: yes"])
    scorer = _fitted_preference_scorer(client, prompt_style="batch")
    group = pd.DataFrame(
        {"user_id": ["u1"], "_llm_group_": ["g0"], "item_title": ["Portal"]}
    )

    scorer.score(group)

    prompt = _prompt_text(client)
    assert TARGET_DESCRIPTIONS["ml-1m"] in prompt
    assert "positive-preference target" in prompt


def test_itemwise_prompt_asks_the_preference_question() -> None:
    """Regression test: itemwise mode must not inherit the interaction prompt.

    `LLMPreferenceYesNoScorer` originally inherited `_build_itemwise_messages`
    from the interaction scorer, which asks "would the user interact with this
    recommendation" and has no slot for `target_description`. Itemwise
    preference runs would then have been graded against liking labels while
    being asked about interaction.
    """
    client = FakeClient(["yes"])
    scorer = _fitted_preference_scorer(client, prompt_style="itemwise")
    group = pd.DataFrame(
        {"user_id": ["u1"], "_llm_group_": ["g0"], "item_title": ["Portal"]}
    )

    scorer.score(group)

    prompt = _prompt_text(client)
    assert TARGET_DESCRIPTIONS["ml-1m"] in prompt
    assert "positive-preference target" in prompt
    assert "interact" not in prompt.lower()


def test_itemwise_prompt_still_parses_bare_yes_no() -> None:
    """The preference itemwise prompt keeps the bare yes/no answer contract."""
    client = FakeClient(["no"])
    scorer = _fitted_preference_scorer(client, prompt_style="itemwise")
    group = pd.DataFrame(
        {"user_id": ["u1"], "_llm_group_": ["g0"], "item_title": ["Portal"]}
    )

    scores = scorer.score(group)

    assert scores.iloc[0] == 0.0
    assert len(client.completions.calls) == 1


# ---------------------------------------------------------------------------
# Popularity runner on a preference task
# ---------------------------------------------------------------------------


def test_popularity_runner_writes_all_artifacts(tmp_path: Path) -> None:
    run(_make_preference_task(), tmp_path)

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / METRICS_FILENAME).exists()
    assert (tmp_path / "predictions.parquet").exists()
    assert (tmp_path / "policy_metrics.json").exists()


def test_popularity_runner_reports_agreement_over_three_policies(
    tmp_path: Path,
) -> None:
    metrics = run(_make_preference_task(), tmp_path)

    assert metrics["protocol"] == "policy_ranking"
    assert metrics["main_metric"] == MAIN_METRIC
    test_metrics = metrics["test"]
    assert test_metrics["n_policies"] == 3
    # With >= 3 policies and non-constant utilities both correlations are real
    # numbers, not None (the K=2 case) and not NaN (the constant-input case).
    assert math.isfinite(test_metrics["kendall_tau"])
    assert math.isfinite(test_metrics["spearman_rho"])
    assert len(set(test_metrics["simulated_utilities"].values())) > 1


def test_utilities_are_restricted_to_common_coverage_users(tmp_path: Path) -> None:
    """u3 is not covered by ItemKNNPolicy, so it must not enter any utility."""
    metrics = run(_make_preference_task(), tmp_path)

    coverage = metrics["policy_coverage"]
    assert coverage["common_users"] == 2
    assert coverage["per_policy"]["RandomPolicy"]["n_users"] == 3
    assert coverage["per_policy"]["PopularityPolicy"]["n_users"] == 3
    assert coverage["per_policy"]["ItemKNNPolicy"]["n_users"] == 2

    # Hand-computed over the common users u1 and u2 only (train preference
    # counts: i1=3, i2=1, i3=2, i4=0, i5=1, i6=0).
    sim = metrics["test"]["simulated_utilities"]
    real = metrics["test"]["real_utilities"]
    assert sim == {
        "RandomPolicy": 1.75,
        "PopularityPolicy": 1.5,
        "ItemKNNPolicy": 0.25,
    }
    assert real == {
        "RandomPolicy": 0.75,
        "PopularityPolicy": 0.5,
        "ItemKNNPolicy": 0.25,
    }
    # Identical orderings on both sides.
    assert metrics["test"]["kendall_tau"] == 1.0


def test_popularity_runner_manifest_records_preference_target(
    tmp_path: Path,
) -> None:
    """Provenance: the artifact must say which target it was scored against."""
    run(_make_preference_task(), tmp_path)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["task"]["manifest"]["target_source_column"] == "target_like_ge4"
