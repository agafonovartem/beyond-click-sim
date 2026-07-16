"""Run classical candidate-free recommenders on both splits and score Recall/NDCG@K.

Reuses the ranking policies from ``beyond_click_sim.tasks.policies`` (the
recommender side), fitting each on the FULL train split (all users' train
interactions provide the collaborative signal) and generating a top-K list for
the fixed eval-user sample. Each recommended list already excludes the user's
train-seen items; the held-out test items are the relevance set.

Note the asymmetry vs the LLM runner: collaborative models use every user's train
data, whereas the candidate-free LLM sees only one user's history plus its own
prior. This is intended and documented, not a bug.

Run:  uv run python memorization/run_classical.py --split both --k 50
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime

import pandas as pd

from beyond_click_sim.tasks.policies import (
    ALSPolicy,
    BPRPolicy,
    ItemKNNPolicy,
    LightGCNPolicy,
    PopularityPolicy,
    RandomPolicy,
)

from baselines import StandardItemKNN
from common import load_eval_users, load_items, load_split, mem_out_dir
from metrics import DEFAULT_KS, aggregate_user_metrics, exact_hit_flags, ranking_metrics_from_hits

# Hyperparameters chosen by memorization/tune_baselines.py, sweeping against
# Di Palma et al. Table 3 on their (byte-identical) file-order split, all 6040 users.
#
# ItemKNNStd(n_neighbors=200) reproduces their ItemKNN to MAE=0.0003
# (HR@1 0.0392 vs 0.0394, nDCG@10 0.0336 vs 0.0337). The src `ItemKNNPolicy` uses a
# non-standard neighbour formulation and lands at HR@1 0.0248; kept only for reference.
ITEMKNN_STD_NEIGHBORS = 200
# BPRMF: best of 59 configs (MAE 0.0194). NOTE every config undershoots their BPRMF
# (best HR@10 0.1697 vs their 0.2149) — implicit's BPR is not Elliot's, so this is as
# close as our implementation gets. Our BPRMF is therefore a conservative baseline.
BPR_PARAMS = {"n_factors": 10, "learning_rate": 0.01, "iterations": 100, "regularization": 0.01}


def build_policies(k: int, seed: int) -> dict[str, object]:
    """Policy registry. Names mirror the appendix baseline list."""
    return {
        "Random": RandomPolicy(k=k, seed=seed),
        "MostPop": PopularityPolicy(k=k, seed=seed),
        "ItemKNNStd": StandardItemKNN(k=k, n_neighbors=ITEMKNN_STD_NEIGHBORS, seed=seed),
        "ItemKNN": ItemKNNPolicy(k=k, seed=seed),
        "BPRMF": BPRPolicy(k=k, seed=seed, **BPR_PARAMS),
        "ALS": ALSPolicy(k=k, seed=seed),
        "LightGCN": LightGCNPolicy(k=k, seed=seed),
    }


def evaluate_policy(
    policy,
    *,
    train: pd.DataFrame,
    items: pd.DataFrame,
    eval_users: list[str],
    test_by_user: dict[str, set[str]],
    ks: tuple[int, ...],
) -> tuple[dict, float]:
    users_df = pd.DataFrame({"user_id": eval_users})
    t0 = time.perf_counter()
    policy.fit(train, items=items)
    recs = policy.recommend(users_df, train_interactions=train, items=items)
    elapsed = time.perf_counter() - t0

    recs_by_user: dict[str, list[str]] = {}
    if not recs.empty:
        for user_id, grp in recs.sort_values("rank").groupby("user_id", sort=False):
            recs_by_user[user_id] = grp["item_id"].tolist()

    per_user = []
    for user_id in eval_users:
        test_items = test_by_user.get(user_id, set())
        ranked = recs_by_user.get(user_id, [])
        flags = exact_hit_flags(ranked, test_items)
        per_user.append(ranking_metrics_from_hits(flags, len(test_items), ks=ks))

    metrics = aggregate_user_metrics(per_user, ks=ks)
    metrics["n_users_with_recs"] = int(len(recs_by_user))
    return metrics, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["file_order", "random", "both"], default="both")
    parser.add_argument("--policies", nargs="*", default=None, help="subset of policy names")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--ks", nargs="*", type=int, default=list(DEFAULT_KS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-sample", action="store_true",
                        help="score only eval_users.csv; default is ALL users (paper protocol)")
    args = parser.parse_args()

    ks = tuple(args.ks)
    splits = ["file_order", "random"] if args.split == "both" else [args.split]
    items = load_items().copy()
    items["item_id"] = items["item_id"].astype(str)

    registry = build_policies(args.k, args.seed)
    policy_names = args.policies or list(registry)

    summary_rows = []
    for split_name in splits:
        frame = load_split(split_name)
        frame["item_id"] = frame["item_id"].astype(str)
        frame["user_id"] = frame["user_id"].astype(str)
        train = frame[frame["split"] == "train"].copy()
        test = frame[frame["split"] == "test"].copy()

        if args.eval_sample:
            eval_users = [str(u) for u in load_eval_users()]
        else:
            # Paper protocol: ML-1M "without any filtering" -> every user.
            eval_users = sorted(test["user_id"].unique().tolist(), key=int)

        test_by_user = test.groupby("user_id")["item_id"].agg(set).to_dict()

        for name in policy_names:
            policy = build_policies(args.k, args.seed)[name]
            print(f"[{split_name}] fitting {name} ...", flush=True)
            metrics, elapsed = evaluate_policy(
                policy,
                train=train,
                items=items,
                eval_users=eval_users,
                test_by_user=test_by_user,
                ks=ks,
            )
            print(f"    HR@10={metrics.get('mean_hit_rate@10', float('nan')):.4f}  "
                  f"nDCG@10={metrics.get('mean_ndcg@10', float('nan')):.4f}  ({elapsed:.1f}s)")

            run_dir = mem_out_dir() / "classical" / split_name / name
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "created_at": datetime.now(UTC).isoformat(),
                "experiment": "memorization/candidate_free",
                "method": name,
                "method_family": "classical_policy",
                "dataset": "ml-1m",
                "split": split_name,
                "k": args.k,
                "ks": list(ks),
                "seed": args.seed,
                "user_set": "eval_sample" if args.eval_sample else "all_users_no_filtering",
                "n_eval_users": len(eval_users),
                "fit_recommend_seconds": round(elapsed, 2),
                "policy_class": type(policy).__name__,
                "policy_params": {
                    k: v for k, v in vars(policy).items() if not k.endswith("_")
                },
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
            (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", "utf-8")

            row = {"split": split_name, "method": name, "n_users": metrics["n_users"]}
            row.update({k: v for k, v in metrics.items() if k.startswith("mean_")})
            summary_rows.append(row)

    # Rebuild summary.csv from every metrics.json on disk, not just this invocation's
    # rows, so running a subset of policies extends the table instead of truncating it.
    all_rows: list[dict] = []
    for mpath in sorted((mem_out_dir() / "classical").glob("*/*/manifest.json")):
        man = json.loads(mpath.read_text(encoding="utf-8"))
        met = json.loads((mpath.parent / "metrics.json").read_text(encoding="utf-8"))
        row = {"split": man["split"], "method": man["method"], "n_users": met["n_users"]}
        row.update({k: v for k, v in met.items() if k.startswith("mean_")})
        all_rows.append(row)

    summary = pd.DataFrame(all_rows).sort_values(["split", "method"]).reset_index(drop=True)
    summary_path = mem_out_dir() / "classical" / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\n=== SUMMARY (all policies on disk) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.to_string(index=False))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
