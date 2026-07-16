"""Re-score saved LLM generations under different matching rules — no LLM calls.

run_llm.py stores every raw generation in predictions.jsonl, so any matching rule
or threshold can be evaluated afterwards for free, on the CPU, without re-querying
the model. Two questions this answers:

1. Which threshold did Di Palma et al. actually use? The paper never says, and
   their code holds three different values (dead default 80, function default 0.85,
   __main__ 1.0 = exact). `--thresholds 80 85 100` reports MAE against their
   published Llama-3.1 8B row; 85 wins (MAE 0.0056).

2. How much does their matching rule distort the LLM's score? `--modes` applies
   each correction separately (`article`, `dedup`, `in_catalog`) and all together
   (`fair`), which isolates the contribution of each defect.

Requires the local row-level predictions.jsonl (git-ignored).

Examples:
  uv run python memorization/rescore.py                      # modes at threshold 85
  uv run python memorization/rescore.py --thresholds 80 85 100 --modes paper
  uv run python memorization/rescore.py --scorers ratio WRatio --modes paper
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import catalog_titles, load_split, mem_out_dir, title_map
from metrics import (
    DEFAULT_KS,
    MATCHING_MODES,
    SCORERS,
    aggregate_user_metrics,
    hit_flags,
    normalizer_for,
    ranking_metrics_from_hits,
)

# Di Palma et al. 2025, Table 3 — the file-order split, all 6040 users, full history.
# NOTE their row reports HR@1=0.0687 but nDCG@1=0.0697, which is impossible for
# binary relevance (nDCG@1 == HR@1) — a typo in their table.
PAPER_TABLE3 = {
    "Llama-3.1 8B": {"hit_rate@1": 0.0687, "hit_rate@5": 0.2281, "hit_rate@10": 0.3500,
                     "ndcg@5": 0.0609, "ndcg@10": 0.0571},
    "Llama-3.3 70B": {"hit_rate@1": 0.2293, "hit_rate@5": 0.4985, "hit_rate@10": 0.5922,
                      "ndcg@5": 0.1693, "ndcg@10": 0.1359},
    "GPT-4o": {"hit_rate@1": 0.2796, "hit_rate@5": 0.5889, "hit_rate@10": 0.6897,
               "ndcg@5": 0.2276, "ndcg@10": 0.1948},
    "GPT-4o mini": {"hit_rate@1": 0.0316, "hit_rate@5": 0.2132, "hit_rate@10": 0.3091,
                    "ndcg@5": 0.0451, "ndcg@10": 0.0413},
    "GPT-3.5 turbo": {"hit_rate@1": 0.2298, "hit_rate@5": 0.4217, "hit_rate@10": 0.5902,
                      "ndcg@5": 0.1281, "ndcg@10": 0.1229},
}


def score_run(recs, test_by_user, titles, *, mode, threshold, scorer, catalog, ks):
    per_user = []
    catalog_cache: dict[str, bool] = {}  # shared across users: same titles recur constantly
    for r in recs:
        test_titles = [titles.get(i, "") for i in test_by_user.get(r["user_id"], [])]
        flags = hit_flags(r["parsed"], test_titles, mode=mode, threshold=threshold,
                          catalog=catalog, scorer=scorer, catalog_cache=catalog_cache)
        per_user.append(ranking_metrics_from_hits(flags, len(test_titles), ks=ks))
    agg = aggregate_user_metrics(per_user, ks=ks)
    return {k.replace("mean_", ""): v for k, v in agg.items() if k.startswith("mean_")}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--modes", nargs="*", default=list(MATCHING_MODES),
                        choices=list(MATCHING_MODES))
    parser.add_argument("--thresholds", nargs="*", type=float, default=[85.0])
    parser.add_argument("--scorers", nargs="*", default=["ratio"], choices=list(SCORERS))
    parser.add_argument("--ks", nargs="*", type=int, default=list(DEFAULT_KS))
    parser.add_argument("--paper-row", default="Llama-3.1 8B", choices=list(PAPER_TABLE3),
                        help="Table 3 row to compare against (file_order + history=all only)")
    parser.add_argument("--out", default="llm_rescore.csv")
    args = parser.parse_args()
    ks = tuple(args.ks)

    titles = title_map()
    catalogs = {m: (catalog_titles(normalizer_for(m)) if MATCHING_MODES[m]["in_catalog"] else None)
                for m in args.modes}
    split_cache: dict[str, dict] = {}
    rows: list[dict] = []

    for pred_path in sorted((mem_out_dir() / "llm").glob("*/*/*/predictions.jsonl")):
        manifest = json.loads((pred_path.parent / "manifest.json").read_text(encoding="utf-8"))
        split, history, model = manifest["split"], manifest["history"], manifest["model"]
        if split not in split_cache:
            frame = load_split(split)
            frame["user_id"] = frame["user_id"].astype(str)
            frame["item_id"] = frame["item_id"].astype(str)
            test = frame[frame["split"] == "test"]
            split_cache[split] = test.groupby("user_id")["item_id"].agg(list).to_dict()
        test_by_user = split_cache[split]
        recs = [json.loads(line) for line in pred_path.open(encoding="utf-8")]

        for mode in args.modes:
            for scorer in args.scorers:
                for thr in args.thresholds:
                    got = score_run(recs, test_by_user, titles, mode=mode, threshold=thr,
                                    scorer=scorer, catalog=catalogs[mode], ks=ks)
                    row = {"model": model, "split": split, "history": history, "matching": mode,
                           "scorer": scorer, "threshold": thr, "n_users": len(recs),
                           **{k: round(v, 4) for k, v in got.items()}}
                    # MAE is only meaningful against their exact condition.
                    target = PAPER_TABLE3[args.paper_row]
                    if split == "file_order" and history == "all":
                        row["MAE_vs_paper"] = round(
                            float(np.mean([abs(got[k] - v) for k, v in target.items()])), 4
                        )
                    rows.append(row)
                    print(f"  scored {split}/{history}/{mode}@{thr:g}", flush=True)

    if not rows:
        print("No predictions.jsonl found under memorization/outputs/llm/. Run run_llm.py first.")
        return

    df = pd.DataFrame(rows)
    out = mem_out_dir() / args.out
    df.to_csv(out, index=False)
    cols = ["model", "split", "history", "matching", "scorer", "threshold", "n_users",
            "hit_rate@1", "hit_rate@5", "hit_rate@10", "ndcg@5", "ndcg@10", "MAE_vs_paper"]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print("\n" + df[[c for c in cols if c in df]].to_string(index=False))
    t = PAPER_TABLE3[args.paper_row]
    print(f"\nPAPER {args.paper_row} (file_order, all): " +
          "  ".join(f"{k.replace('hit_rate', 'HR')}={v}" for k, v in t.items()))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
