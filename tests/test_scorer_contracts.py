from __future__ import annotations

import pandas as pd

from beyond_click_sim.scorers import Scorer


class ConstantScorer(Scorer):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ConstantScorer":
        self.value = float(y.mean())
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=X.index, name="score")


def test_scorer_contract_supports_fit_and_score() -> None:
    X_train = pd.DataFrame({"item_id": ["i1", "i2"]})
    y_train = pd.Series([1, 0], name="target")
    X_test = pd.DataFrame({"item_id": ["i3", "i4"]}, index=["a", "b"])

    scorer = ConstantScorer().fit(X_train, y_train)
    scores = scorer.score(X_test)

    assert scores.tolist() == [0.5, 0.5]
    assert list(scores.index) == ["a", "b"]
