from __future__ import annotations


def policy_ranking_agreement_metrics(
    policy_names: list[str],
    simulated_utilities: list[float],
    real_utilities: list[float],
) -> dict:
    """Compute Kendall's tau and Spearman's rho between two policy rankings.

    Parameters
    ----------
    policy_names:
        Ordered list of K policy identifiers (K >= 2).
    simulated_utilities:
        One scalar utility per policy, in the same order as policy_names.
        Higher is better (e.g. mean simulated score).
    real_utilities:
        One scalar real utility per policy, same order.
        Higher is better (e.g. mean real hit rate on held-out data).

    Returns
    -------
    dict with keys:
        kendall_tau, kendall_tau_pvalue  — None when K < 3
        spearman_rho, spearman_rho_pvalue — None when K < 3
        n_policies                         — int
        simulated_rank                     — list[str], sorted by simulated utility desc
        real_rank                          — list[str], sorted by real utility desc
        simulated_utilities                — dict[str, float]
        real_utilities                     — dict[str, float]
        warning                            — str | None

    Edge cases
    ----------
    K < 2: raises ValueError.
    K == 2: tau/rho/pvalues are None, warning is set.
    Ties in utilities: scipy uses tau-b and average ranks, both handle ties.
    """
    n = len(policy_names)
    if n < 2:
        raise ValueError(
            f"At least 2 policies are required, got {n}."
        )
    if len(simulated_utilities) != n or len(real_utilities) != n:
        raise ValueError(
            "policy_names, simulated_utilities, and real_utilities must have "
            f"the same length. Got {n}, {len(simulated_utilities)}, "
            f"{len(real_utilities)}."
        )

    tau = tau_p = rho = rho_p = None
    warning: str | None = None

    if n >= 3:
        from scipy.stats import kendalltau, spearmanr

        tau_result = kendalltau(simulated_utilities, real_utilities)
        rho_result = spearmanr(simulated_utilities, real_utilities)
        tau = float(tau_result.statistic)
        tau_p = float(tau_result.pvalue)
        rho = float(rho_result.statistic)
        rho_p = float(rho_result.pvalue)
    else:
        warning = (
            "n_policies=2: rank correlation statistics are undefined; "
            "min 3 needed. Returning None for tau/rho."
        )

    sim_ranked = [
        p
        for _, p in sorted(
            zip(simulated_utilities, policy_names), key=lambda x: x[0], reverse=True
        )
    ]
    real_ranked = [
        p
        for _, p in sorted(
            zip(real_utilities, policy_names), key=lambda x: x[0], reverse=True
        )
    ]

    return {
        "kendall_tau": tau,
        "kendall_tau_pvalue": tau_p,
        "spearman_rho": rho,
        "spearman_rho_pvalue": rho_p,
        "n_policies": n,
        "simulated_rank": sim_ranked,
        "real_rank": real_ranked,
        "simulated_utilities": dict(zip(policy_names, [float(v) for v in simulated_utilities])),
        "real_utilities": dict(zip(policy_names, [float(v) for v in real_utilities])),
        "warning": warning,
    }
