"""Sweep baseline hyperparameters against Di Palma et al. (2025) Table 3.

Their baselines were produced with Elliot (see `evaluate_recommendations.py`,
which reads `data/movielens_1M/elliot/test.tsv`) and the configs were never
published, so the only way to recover them is to sweep our own implementations on
the *identical* split and compare.

Grounding facts already established:
  - our file_order split is byte-identical to their released training/test.tsv
    (6040/6040 users, same order);
  - MostPop reproduces their row exactly => split, candidate protocol and metric
    definitions are correct.
So any remaining gap is hyperparameters / implementation only.

Run:
  uv run python memorization/tune_baselines.py --model itemknn --neighbors 10 20 40 80 150
  uv run python memorization/tune_baselines.py --model bprmf --factors 10 64 --lr 0.001 0.01
"""

from __future__ import annotations

import argparse
import itertools

import pandas as pd

from beyond_click_sim.tasks.policies import BPRPolicy, LightGCNPolicy

from baselines import StandardItemKNN
from common import load_items, load_split
from metrics import aggregate_user_metrics, exact_hit_flags, ranking_metrics_from_hits

KS = (1, 5, 10)

# Di Palma et al. 2025, Tables/table3_recommendations.tex (file-order split, all 6040 users).
PAPER = {
    "Random":   {"hit_rate@1": 0.0093, "hit_rate@5": 0.0442, "hit_rate@10": 0.0851,
                 "ndcg@1": 0.0093, "ndcg@5": 0.0092, "ndcg@10": 0.0094},
    "MostPop":  {"hit_rate@1": 0.0212, "hit_rate@5": 0.0775, "hit_rate@10": 0.1520,
                 "ndcg@1": 0.0212, "ndcg@5": 0.0222, "ndcg@10": 0.0251},
    "ItemKNN":  {"hit_rate@1": 0.0394, "hit_rate@5": 0.1217, "hit_rate@10": 0.1828,
                 "ndcg@1": 0.0394, "ndcg@5": 0.0353, "ndcg@10": 0.0337},
    "BPRMF":    {"hit_rate@1": 0.0406, "hit_rate@5": 0.1278, "hit_rate@10": 0.2149,
                 "ndcg@1": 0.0406, "ndcg@5": 0.0350, "ndcg@10": 0.0356},
    "LightGCN": {"hit_rate@1": 0.0358, "hit_rate@5": 0.1136, "hit_rate@10": 0.1882,
                 "ndcg@1": 0.0358, "ndcg@5": 0.0306, "ndcg@10": 0.0311},
}


def evaluate(policy, *, train, items, eval_users, test_by_user) -> dict[str, float]:
    policy.fit(train, items=items)
    recs = policy.recommend(
        pd.DataFrame({"user_id": eval_users}), train_interactions=train, items=items
    )
    recs_by_user: dict[str, list[str]] = {}
    if not recs.empty:
        for user_id, grp in recs.sort_values("rank").groupby("user_id", sort=False):
            recs_by_user[user_id] = grp["item_id"].tolist()
    per_user = [
        ranking_metrics_from_hits(
            exact_hit_flags(recs_by_user.get(u, []), test_by_user.get(u, set())),
            len(test_by_user.get(u, set())),
            ks=KS,
        )
        for u in eval_users
    ]
    agg = aggregate_user_metrics(per_user, ks=KS)
    return {k.replace("mean_", ""): v for k, v in agg.items() if k.startswith("mean_")}


def report(label: str, got: dict[str, float], target_name: str) -> None:
    target = PAPER[target_name]
    cells, diffs = [], []
    for key in ("hit_rate@1", "hit_rate@5", "hit_rate@10", "ndcg@5", "ndcg@10"):
        g, t = got[key], target[key]
        cells.append(f"{key.replace('hit_rate', 'HR')}={g:.4f}(Δ{g - t:+.4f})")
        diffs.append(abs(g - t))
    mae = sum(diffs) / len(diffs)
    print(f"  {label:38s} MAE={mae:.4f}  " + "  ".join(cells), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["itemknn", "bprmf", "lightgcn"], required=True)
    parser.add_argument("--split", default="file_order")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--neighbors", nargs="*", type=int, default=[10, 20, 40, 80, 150])
    parser.add_argument("--factors", nargs="*", type=int, default=[10, 64])
    parser.add_argument("--lr", nargs="*", type=float, default=[0.001, 0.01])
    parser.add_argument("--iterations", nargs="*", type=int, default=[10, 100])
    parser.add_argument("--reg", nargs="*", type=float, default=[0.01])
    parser.add_argument("--layers", nargs="*", type=int, default=[3])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = load_split(args.split)
    frame["user_id"] = frame["user_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)
    train = frame[frame["split"] == "train"].copy()
    test = frame[frame["split"] == "test"].copy()
    items = load_items().copy()
    items["item_id"] = items["item_id"].astype(str)
    eval_users = sorted(test["user_id"].unique().tolist(), key=int)
    test_by_user = test.groupby("user_id")["item_id"].agg(set).to_dict()

    print(f"split={args.split}  users={len(eval_users)}  model={args.model}")
    t = PAPER[{"itemknn": "ItemKNN", "bprmf": "BPRMF", "lightgcn": "LightGCN"}[args.model]]
    print(f"  TARGET (paper): HR@1={t['hit_rate@1']:.4f} HR@5={t['hit_rate@5']:.4f} "
          f"HR@10={t['hit_rate@10']:.4f} nDCG@5={t['ndcg@5']:.4f} nDCG@10={t['ndcg@10']:.4f}")

    if args.model == "itemknn":
        for n in args.neighbors:
            got = evaluate(
                StandardItemKNN(k=args.k, n_neighbors=n, seed=args.seed),
                train=train, items=items, eval_users=eval_users, test_by_user=test_by_user,
            )
            report(f"ItemKNNStd(neighbors={n})", got, "ItemKNN")
    elif args.model == "bprmf":
        for f, lr, it, rg in itertools.product(args.factors, args.lr, args.iterations, args.reg):
            got = evaluate(
                BPRPolicy(k=args.k, n_factors=f, learning_rate=lr, iterations=it,
                          regularization=rg, seed=args.seed),
                train=train, items=items, eval_users=eval_users, test_by_user=test_by_user,
            )
            report(f"BPR(f={f},lr={lr},it={it},reg={rg})", got, "BPRMF")
    else:
        for f, lr, it, ly in itertools.product(args.factors, args.lr, args.iterations, args.layers):
            got = evaluate(
                LightGCNPolicy(k=args.k, n_factors=f, learning_rate=lr, iterations=it,
                               n_layers=ly, seed=args.seed),
                train=train, items=items, eval_users=eval_users, test_by_user=test_by_user,
            )
            report(f"LightGCN(f={f},lr={lr},it={it},L={ly})", got, "LightGCN")


if __name__ == "__main__":
    main()
