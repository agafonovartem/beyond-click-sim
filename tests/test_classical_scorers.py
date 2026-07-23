from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

from beyond_click_sim.scorers import (
    ALSScorer,
    BPRScorer,
    ItemKNNScorer,
    LightGCNScorer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.in_distribution.interaction_prediction.methods.classical import (  # noqa: E402
    coverage_summary,
)


# Same CF fixture as tests/test_policies.py: u2..u5 co-interact with A and C,
# u1 only with A, u6 with B and D. Every scorer should therefore rank C above
# B and D for u1. Learning-rate/iteration overrides mirror test_policies.py —
# a 4-item graph needs more steps than the production defaults to converge.
def _train() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("u1", "A"),
            ("u2", "A"), ("u2", "C"),
            ("u3", "A"), ("u3", "C"),
            ("u4", "A"), ("u4", "C"),
            ("u5", "A"), ("u5", "C"),
            ("u6", "B"), ("u6", "D"),
        ],
        columns=["user_id", "item_id"],
    )


def _y(train: pd.DataFrame) -> pd.Series:
    # The interaction-prediction train split is positives-only.
    return pd.Series([1] * len(train), name="target")


SCORER_FACTORIES = {
    "item_knn": lambda: ItemKNNScorer(n_neighbors=5),
    "als": lambda: ALSScorer(n_factors=8, iterations=30, seed=0),
    "bpr": lambda: BPRScorer(n_factors=8, learning_rate=0.05, iterations=300, seed=0),
    "lightgcn": lambda: LightGCNScorer(
        n_factors=8, n_layers=2, learning_rate=0.05, iterations=400, seed=0
    ),
}
ALL_SCORERS = sorted(SCORER_FACTORIES)


# ---------------------------------------------------------------------------
# Shared scorer contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_SCORERS)
def test_ranks_cooccurring_item_above_unrelated_items(name: str) -> None:
    train = _train()
    candidates = pd.DataFrame(
        [("u1", "B"), ("u1", "C"), ("u1", "D")],
        columns=["user_id", "item_id"],
    )

    scores = SCORER_FACTORIES[name]().fit(train, _y(train)).score(candidates)

    assert scores.iloc[1] > scores.iloc[0]  # C beats B
    assert scores.iloc[1] > scores.iloc[2]  # C beats D


@pytest.mark.parametrize("name", ALL_SCORERS)
def test_cold_user_and_cold_item_score_zero(name: str) -> None:
    train = _train()
    candidates = pd.DataFrame(
        [("cold_user", "C"), ("u1", "COLD_ITEM")],
        columns=["user_id", "item_id"],
    )

    scores = SCORER_FACTORIES[name]().fit(train, _y(train)).score(candidates)

    assert scores.tolist() == [0.0, 0.0]


@pytest.mark.parametrize("name", ALL_SCORERS)
def test_score_preserves_index_and_name(name: str) -> None:
    train = _train()
    candidates = pd.DataFrame(
        [("u1", "B"), ("u1", "C")],
        columns=["user_id", "item_id"],
        index=["row_a", "row_b"],
    )

    scores = SCORER_FACTORIES[name]().fit(train, _y(train)).score(candidates)

    assert list(scores.index) == ["row_a", "row_b"]
    assert scores.name == "score"
    assert len(scores) == len(candidates)


@pytest.mark.parametrize("name", ALL_SCORERS)
def test_deterministic_across_separate_fits(name: str) -> None:
    """Re-fitting with the same seed must reproduce scores exactly.

    This is what forced num_threads=1 on the implicit models: their parallel
    SGD is otherwise non-reproducible even with random_state set.
    """
    train = _train()
    candidates = pd.DataFrame(
        [("u1", "B"), ("u1", "C"), ("u1", "D")],
        columns=["user_id", "item_id"],
    )

    first = SCORER_FACTORIES[name]().fit(train, _y(train)).score(candidates)
    second = SCORER_FACTORIES[name]().fit(train, _y(train)).score(candidates)

    pd.testing.assert_series_equal(first, second)


@pytest.mark.parametrize("name", ALL_SCORERS)
def test_ignores_y_because_train_split_is_positives_only(name: str) -> None:
    """Scorers treat train (user, item) pairs as implicit positives.

    y is accepted for API consistency only, so a different y must not change
    the scores. Guards against silently reusing these scorers on a task with
    real 0/1 labels, where ignoring negatives would be wrong.
    """
    train = _train()
    candidates = pd.DataFrame(
        [("u1", "B"), ("u1", "C")],
        columns=["user_id", "item_id"],
    )
    ones = pd.Series([1] * len(train), name="target")
    zeros = pd.Series([0] * len(train), name="target")

    with_ones = SCORER_FACTORIES[name]().fit(train, ones).score(candidates)
    with_zeros = SCORER_FACTORIES[name]().fit(train, zeros).score(candidates)

    pd.testing.assert_series_equal(with_ones, with_zeros)


@pytest.mark.parametrize("name", ALL_SCORERS)
def test_requires_fit_before_score(name: str) -> None:
    candidates = pd.DataFrame([("u1", "C")], columns=["user_id", "item_id"])

    with pytest.raises(RuntimeError, match="not fitted"):
        SCORER_FACTORIES[name]().score(candidates)


@pytest.mark.parametrize("name", ALL_SCORERS)
@pytest.mark.parametrize("missing", ["user_id", "item_id"])
def test_fit_requires_id_columns(name: str, missing: str) -> None:
    train = _train().drop(columns=[missing])

    with pytest.raises(ValueError, match="Missing column"):
        SCORER_FACTORIES[name]().fit(train, pd.Series([1] * len(train)))


# ---------------------------------------------------------------------------
# Per-scorer parameter validation
# ---------------------------------------------------------------------------

def test_item_knn_rejects_non_positive_n_neighbors() -> None:
    with pytest.raises(ValueError, match="n_neighbors"):
        ItemKNNScorer(n_neighbors=0)


def test_item_knn_rejects_unknown_aggregation() -> None:
    with pytest.raises(ValueError, match="aggregation"):
        ItemKNNScorer(aggregation="median")


def test_item_knn_sum_and_mean_agree_on_ranking_for_one_user() -> None:
    """mean only rescales by the profile size, so within-user order is identical."""
    train = _train()
    candidates = pd.DataFrame(
        [("u1", "B"), ("u1", "C"), ("u1", "D")],
        columns=["user_id", "item_id"],
    )

    mean_scores = ItemKNNScorer(n_neighbors=5, aggregation="mean").fit(
        train, _y(train)
    ).score(candidates)
    sum_scores = ItemKNNScorer(n_neighbors=5, aggregation="sum").fit(
        train, _y(train)
    ).score(candidates)

    assert list(mean_scores.rank()) == list(sum_scores.rank())


@pytest.mark.parametrize("scorer_cls", [ALSScorer, BPRScorer])
def test_mf_rejects_non_positive_n_factors(scorer_cls) -> None:
    with pytest.raises(ValueError, match="n_factors"):
        scorer_cls(n_factors=0)


@pytest.mark.parametrize("scorer_cls", [ALSScorer, BPRScorer])
def test_mf_rejects_non_positive_iterations(scorer_cls) -> None:
    with pytest.raises(ValueError, match="iterations"):
        scorer_cls(iterations=0)


def test_lightgcn_rejects_non_positive_n_factors() -> None:
    with pytest.raises(ValueError, match="n_factors"):
        LightGCNScorer(n_factors=0)


def test_lightgcn_rejects_non_positive_n_layers() -> None:
    with pytest.raises(ValueError, match="n_layers"):
        LightGCNScorer(n_layers=0)


def test_lightgcn_rejects_non_positive_iterations() -> None:
    with pytest.raises(ValueError, match="iterations"):
        LightGCNScorer(iterations=0)


# ---------------------------------------------------------------------------
# Coverage diagnostic used in the classical runner manifests
# ---------------------------------------------------------------------------

def test_coverage_summary_counts_cold_rows_and_nonzero_scores() -> None:
    X_train = pd.DataFrame(
        {"user_id": ["u1", "u2"], "item_id": ["i1", "i2"]},
    )
    X_split = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u3", "u2"],
            "item_id": ["i1", "i3", "i1", "i2"],
        }
    )
    y_split = pd.Series([1, 0, 1, 0])
    scores = pd.Series([0.5, 0.0, 0.0, 0.0])

    summary = coverage_summary(
        X_split=X_split, y_split=y_split, scores=scores, X_train=X_train
    )

    assert summary["rows"] == 4
    assert summary["cold_user_rows"] == 1  # u3 absent from train
    assert summary["cold_item_rows"] == 1  # i3 absent from train
    assert summary["nonzero_score_rows"] == 1
    assert summary["nonzero_score_fraction"] == 0.25
    assert summary["positive_rows"] == 2
    # 1 of the 2 positives received a nonzero score
    assert summary["positive_nonzero_score_fraction"] == 0.5


def test_coverage_summary_handles_all_zero_scores() -> None:
    X_train = pd.DataFrame({"user_id": ["u1"], "item_id": ["i1"]})
    X_split = pd.DataFrame({"user_id": ["u1"], "item_id": ["i1"]})

    summary = coverage_summary(
        X_split=X_split,
        y_split=pd.Series([0]),
        scores=pd.Series([0.0]),
        X_train=X_train,
    )

    assert summary["nonzero_score_fraction"] == 0.0
    assert summary["positive_rows"] == 0
    assert summary["positive_nonzero_score_fraction"] == 0.0
