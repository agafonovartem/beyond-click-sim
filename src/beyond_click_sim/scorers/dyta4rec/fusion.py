"""Source-derived asymmetric weights, with exact mathematical tie preservation."""

from fractions import Fraction


def weights(cluster_flag: str, sequence_flag: str):
    if cluster_flag not in {"yes", "no", "other"} or sequence_flag not in {
        "yes",
        "no",
        "other",
    }:
        raise ValueError("Missing or invalid pattern flag")
    if cluster_flag != "yes":
        return (Fraction(1), Fraction(1), Fraction(0))
    return (
        Fraction(1) if sequence_flag == "yes" else Fraction(1, 2),
        Fraction(1),
        Fraction(1),
    )


def fuse(branches, *, labels, cluster_flag, sequence_flag, strategy="rrf", rrf_k=60):
    if strategy not in {"rrf", "bc"} or rrf_k < 1:
        raise ValueError("Invalid fusion settings")
    active = weights(cluster_flag, sequence_flag)
    ranks = {}
    for branch in ("sequence", "profile", "cluster"):
        ranking = branches[branch]
        if len(ranking) != len(labels) or set(ranking) != set(labels):
            raise ValueError("Each branch must be a complete permutation")
        ranks[branch] = {label: rank for rank, label in enumerate(ranking, 1)}
    exact = {
        label: sum(
            (
                weight / (rrf_k + ranks[branch][label])
                if strategy == "rrf"
                else weight * (len(labels) - ranks[branch][label] + 2)
                for branch, weight in zip(
                    ("sequence", "profile", "cluster"), active, strict=True
                )
            ),
            Fraction(0),
        )
        for label in labels
    }
    return {
        "scores": [float(exact[label]) for label in labels],
        "exact_scores": {label: str(exact[label]) for label in labels},
        "weights": [float(w) for w in active],
        "active_weight": float(sum(active)),
        "ranks": ranks,
        "strategy": strategy,
        "candidate_count": len(labels),
        "tie_groups": [
            [label for label in labels if exact[label] == value]
            for value in dict.fromkeys(exact.values())
            if sum(v == value for v in exact.values()) > 1
        ],
    }
