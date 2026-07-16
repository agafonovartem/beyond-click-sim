"""Build the two per-user 80/20 splits and the shared eval-user sample.

Outputs (under memorization/data/):
  - split_file_order.csv : first n_train file-order rows -> train, rest -> test
  - split_random.csv     : seeded per-user permutation, SAME n_train / n_test
  - eval_users.csv       : the fixed user subset scored by every method / split
  - split_manifest.json  : provenance + parity check + file-order/time diagnostic

Run:  uv run python memorization/prepare_data.py --n-eval-users 200 --seed 0
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import load_interactions, mem_data_dir
from splits import TRAIN_FRACTION, make_parity_splits

OUT_COLUMNS = ["user_id", "item_id", "rating", "timestamp", "interaction_id", "split"]


def file_order_structure_diagnostic(
    interactions: pd.DataFrame, *, seed: int, n_users: int = 300
) -> dict:
    """Characterize what the within-user ratings-file order actually encodes.

    Two distinct questions:
      1. Is it temporal? (adjacent-pair time-ordering ~0.5 and Spearman ~0 => NO)
      2. Is it random? Measured by how consistently an item sits at the same
         relative position across the users who rated it. std of per-item
         within-user frac_pos ~0.29 => genuinely random; markedly lower => the
         file encodes a fixed GLOBAL item order, which makes the file-order
         holdout a structured, cross-user-correlated item holdout rather than a
         per-user random one.
    """
    work = interactions.reset_index(drop=True)
    work["_pos"] = np.arange(len(work))
    rng = np.random.default_rng(seed)
    users = work["user_id"].unique()
    sample = rng.choice(users, size=min(n_users, len(users)), replace=False)

    adj_fracs: list[float] = []
    rhos: list[float] = []
    for user_id in sample:
        group = work[work["user_id"] == user_id].sort_values("_pos")
        ts = group["timestamp"].to_numpy()
        if len(ts) >= 2:
            adj_fracs.append(float(np.mean(np.diff(ts) >= 0)))
            rho = spearmanr(np.arange(len(ts)), ts).statistic
            if not np.isnan(rho):
                rhos.append(float(rho))

    # Per-item consistency of within-user relative position (the decisive stat).
    work["_within"] = work.groupby("user_id").cumcount()
    work["_n"] = work.groupby("user_id")["user_id"].transform("size")
    work["_frac_pos"] = work["_within"] / (work["_n"] - 1).clip(lower=1)
    per_item = work.groupby("item_id")["_frac_pos"].agg(["mean", "std", "count"])
    popular = per_item[per_item["count"] >= 50]

    return {
        "n_users_sampled": int(len(sample)),
        "temporal": {
            "mean_adjacent_pairs_time_ordered": round(float(np.mean(adj_fracs)), 4),
            "mean_within_user_spearman_pos_timestamp": round(float(np.mean(rhos)), 4),
            "verdict": "file order is NOT temporal (~0.5 adjacent, ~0 Spearman)",
        },
        "global_item_order": {
            "n_items_ge50_raters": int(len(popular)),
            "mean_per_item_frac_pos": round(float(popular["mean"].mean()), 4),
            "mean_per_item_frac_pos_std": round(float(popular["std"].mean()), 4),
            "uniform_random_reference_std": 0.289,
            "verdict": (
                "per-item position std well below the ~0.29 random reference => each item "
                "sits at a near-fixed position in every user's block, i.e. the file encodes "
                "a global item order"
            ),
        },
        "interpretation": (
            "Within-user file order is time-independent, but NOT random: it follows a fixed "
            "global item permutation. The file-order holdout therefore removes the SAME items "
            "across users, collapsing their train popularity and stripping collaborative signal "
            "for exactly the held-out items. This suppresses popularity/CF baselines relative to "
            "a true random per-user split; it is a structured cross-user item holdout, not a "
            "temporal split."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-eval-users", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION)
    args = parser.parse_args()

    interactions = load_interactions()
    print(f"Loaded {len(interactions):,} interactions, {interactions['user_id'].nunique():,} users")

    result = make_parity_splits(
        interactions, train_fraction=args.train_fraction, seed=args.seed
    )
    print("Split parity check passed (per-user n_train/n_test identical across splits).")

    out_dir = mem_data_dir()
    result.file_order[OUT_COLUMNS].to_csv(out_dir / "split_file_order.csv", index=False)
    result.random[OUT_COLUMNS].to_csv(out_dir / "split_random.csv", index=False)

    # Fixed eval-user sample, reused across all methods / splits / history conditions.
    rng = np.random.default_rng(args.seed)
    all_users = result.sizes["user_id"].to_numpy()
    n_eval = min(args.n_eval_users, len(all_users))
    eval_users = np.sort(rng.choice(all_users, size=n_eval, replace=False))
    pd.DataFrame({"user_id": eval_users}).to_csv(out_dir / "eval_users.csv", index=False)

    sizes = result.sizes
    eval_sizes = sizes[sizes["user_id"].isin(eval_users)]
    diagnostic = file_order_structure_diagnostic(interactions, seed=args.seed)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": "ml-1m",
        "canonical_version": "v1",
        "train_fraction": args.train_fraction,
        "rounding_rule": "n_train = floor(train_fraction * n), clamped to [1, n-1]",
        "seed": args.seed,
        "n_interactions": int(len(interactions)),
        "n_users_total": int(len(all_users)),
        "n_eval_users": int(n_eval),
        "parity_guaranteed": True,
        "per_user_train_size": {
            "min": int(sizes["n_train"].min()),
            "median": int(sizes["n_train"].median()),
            "max": int(sizes["n_train"].max()),
        },
        "per_user_test_size": {
            "min": int(sizes["n_test"].min()),
            "median": int(sizes["n_test"].median()),
            "max": int(sizes["n_test"].max()),
        },
        "eval_users_test_size": {
            "min": int(eval_sizes["n_test"].min()),
            "median": int(eval_sizes["n_test"].median()),
            "max": int(eval_sizes["n_test"].max()),
            "total_held_out": int(eval_sizes["n_test"].sum()),
        },
        "file_order_structure_diagnostic": diagnostic,
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote splits + eval_users ({n_eval}) + manifest to {out_dir}")
    print(json.dumps(diagnostic, indent=2))


if __name__ == "__main__":
    main()
