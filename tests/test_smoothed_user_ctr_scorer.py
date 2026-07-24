from __future__ import annotations

import pandas as pd
import pytest

from beyond_click_sim.scorers import SmoothedUserCTRScorer


def test_smoothed_user_ctr_uses_fixed_beta_prior() -> None:
    history = pd.DataFrame(
        {
            "user_id": ["u0"] * 5 + ["u1"] * 5,
            "history_click": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )
    scorer = SmoothedUserCTRScorer(alpha=1, beta=1).fit(
        history, history["history_click"]
    )

    scores = scorer.score(
        pd.DataFrame({"user_id": ["u0", "u1", "u0"]})
    )

    assert scores.tolist() == pytest.approx([1 / 7, 6 / 7, 1 / 7])
    assert scorer.user_count_by_user_ == {"u0": 5, "u1": 5}
    assert scorer.fit_diagnostics_["prior_mean"] == pytest.approx(0.5)


def test_smoothed_user_ctr_rejects_nonbinary_and_unknown_users() -> None:
    bad = pd.DataFrame(
        {"user_id": ["u"], "history_click": [2]}
    )
    with pytest.raises(ValueError, match="binary"):
        SmoothedUserCTRScorer().fit(bad, pd.Series([2]))

    good = pd.DataFrame(
        {"user_id": ["u"], "history_click": [1]}
    )
    scorer = SmoothedUserCTRScorer().fit(good, pd.Series([1]))
    with pytest.raises(ValueError, match="Missing fitted history"):
        scorer.score(pd.DataFrame({"user_id": ["cold"]}))
