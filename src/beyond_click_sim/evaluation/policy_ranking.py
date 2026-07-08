from __future__ import annotations

import pandas as pd

from beyond_click_sim.evaluation.ranking import (
    grouped_ranking_metrics,
    user_grouped_ranking_metrics,
)


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


def evaluate_policy_recommendations(
    recs: pd.DataFrame,
    *,
    targets: pd.Series,
    user_column: str = "user_id",
    policy_name: str,
    k: int,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    tie_policy: str = "average",
    fit_recommend_seconds: float | None = None,
) -> dict[str, object]:
    """Compute held-out ranking quality for one policy recommendation frame."""

    if len(recs) != len(targets):
        raise ValueError(
            "recs and targets must have same length. "
            f"Got {len(recs)} and {len(targets)}."
        )
    if "rank" not in recs.columns:
        raise ValueError("recs must contain 'rank' column.")
    if user_column not in recs.columns:
        raise ValueError(f"recs must contain user column: {user_column!r}")

    frame = recs.copy()
    frame["target"] = targets.to_numpy()
    # rank 1 should be highest score for ranking metrics.
    frame["score"] = -frame["rank"].astype(float)
    frame["candidate_group"] = (
        frame[user_column].astype(str) + "::" + str(policy_name)
    )

    macro_by_group = grouped_ranking_metrics(
        frame["target"],
        frame["score"],
        frame["candidate_group"],
        ks=ks,
        tie_policy=tie_policy,  # type: ignore[arg-type]
    )
    macro_by_user = user_grouped_ranking_metrics(
        frame["target"],
        frame["score"],
        frame["candidate_group"],
        frame[user_column],
        ks=ks,
        tie_policy=tie_policy,  # type: ignore[arg-type]
    )

    headline_k = k if k in ks else max(ks)
    headline_key = f"ndcg@{headline_k}"
    result: dict[str, object] = {
        "policy": policy_name,
        "k": int(k),
        "n_users": int(frame[user_column].nunique()),
        "n_recommendations": int(len(frame)),
        "mean_hit_rate": float(frame["target"].mean()) if len(frame) else 0.0,
        "headline_metric": f"macro_by_user_group_mean.{headline_key}",
        "headline_value": float(macro_by_user[headline_key]),
        "ranking": {
            "ks": list(ks),
            "tie_policy": tie_policy,
            "macro_by_group": macro_by_group,
            "macro_by_user_group_mean": macro_by_user,
        },
    }
    if fit_recommend_seconds is not None:
        result["fit_recommend_seconds"] = float(fit_recommend_seconds)
    return result
