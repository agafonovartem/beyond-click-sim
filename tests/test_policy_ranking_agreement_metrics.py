from __future__ import annotations

import pytest

from beyond_click_sim.evaluation.policy_ranking import policy_ranking_agreement_metrics


def test_perfect_agreement():
    result = policy_ranking_agreement_metrics(
        policy_names=["a", "b", "c"],
        simulated_utilities=[0.9, 0.6, 0.3],
        real_utilities=[0.8, 0.5, 0.2],
    )
    assert result["kendall_tau"] == pytest.approx(1.0)
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["n_policies"] == 3
    assert result["warning"] is None


def test_perfect_disagreement():
    result = policy_ranking_agreement_metrics(
        policy_names=["a", "b", "c"],
        simulated_utilities=[0.9, 0.6, 0.3],
        real_utilities=[0.2, 0.5, 0.8],
    )
    assert result["kendall_tau"] == pytest.approx(-1.0)
    assert result["spearman_rho"] == pytest.approx(-1.0)


def test_k_less_than_2_raises():
    with pytest.raises(ValueError):
        policy_ranking_agreement_metrics(
            policy_names=["a"],
            simulated_utilities=[0.5],
            real_utilities=[0.5],
        )


def test_k_equals_2_returns_none_with_warning():
    result = policy_ranking_agreement_metrics(
        policy_names=["a", "b"],
        simulated_utilities=[0.9, 0.3],
        real_utilities=[0.7, 0.2],
    )
    assert result["kendall_tau"] is None
    assert result["spearman_rho"] is None
    assert result["kendall_tau_pvalue"] is None
    assert result["spearman_rho_pvalue"] is None
    assert result["warning"] is not None
    assert result["n_policies"] == 2


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        policy_ranking_agreement_metrics(
            policy_names=["a", "b", "c"],
            simulated_utilities=[0.9, 0.6],
            real_utilities=[0.8, 0.5, 0.2],
        )


def test_output_keys_present():
    result = policy_ranking_agreement_metrics(
        policy_names=["a", "b", "c"],
        simulated_utilities=[0.9, 0.6, 0.3],
        real_utilities=[0.8, 0.5, 0.2],
    )
    required = {
        "kendall_tau", "kendall_tau_pvalue",
        "spearman_rho", "spearman_rho_pvalue",
        "n_policies", "simulated_rank", "real_rank",
        "simulated_utilities", "real_utilities", "warning",
    }
    assert required.issubset(result.keys())


def test_simulated_and_real_ranks_are_sorted():
    result = policy_ranking_agreement_metrics(
        policy_names=["a", "b", "c"],
        simulated_utilities=[0.9, 0.3, 0.6],
        real_utilities=[0.1, 0.8, 0.5],
    )
    # simulated: a > c > b → ["a", "c", "b"]
    assert result["simulated_rank"] == ["a", "c", "b"]
    # real: b > c > a → ["b", "c", "a"]
    assert result["real_rank"] == ["b", "c", "a"]


def test_ties_do_not_crash():
    result = policy_ranking_agreement_metrics(
        policy_names=["a", "b", "c"],
        simulated_utilities=[0.5, 0.5, 0.5],
        real_utilities=[0.3, 0.7, 0.5],
    )
    assert result["kendall_tau"] is not None or result["warning"] is not None


def test_utilities_stored_as_dicts():
    result = policy_ranking_agreement_metrics(
        policy_names=["a", "b", "c"],
        simulated_utilities=[0.9, 0.6, 0.3],
        real_utilities=[0.8, 0.5, 0.2],
    )
    assert result["simulated_utilities"] == {"a": 0.9, "b": 0.6, "c": 0.3}
    assert result["real_utilities"] == {"a": 0.8, "b": 0.5, "c": 0.2}
