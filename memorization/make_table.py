"""Build the committable results table from all runs under memorization/outputs/.

Writes:
  outputs/results.csv      one row per (family, method, split, history, matching)
  outputs/results.md       the same, formatted for reading / pasting into the paper

Families:
  paper_reference  Di Palma et al. 2025, Table 3 (published numbers, for the
                   reproduction check — not produced by this repo)
  classical        our classical policies (exact item-id matching)
  llm             our candidate-free LLM runs (fuzzy title matching)

Run:  uv run python memorization/make_table.py
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from common import mem_out_dir

KS = (1, 5, 10)
METRICS = [f"{name}@{k}" for k in KS for name in ("hit_rate", "ndcg")]

# Di Palma et al. 2025, Tables/table3_recommendations.tex — file-order split, all
# 6040 users, full history. Their nDCG@1 for Llama-3.1 8B (0.0697) contradicts its
# HR@1 (0.0687); nDCG@1 == HR@1 for binary relevance, so we record HR@1 for both
# and flag it in the README.
PAPER_TABLE3 = {
    "Random":        [0.0093, 0.0093, 0.0442, 0.0092, 0.0851, 0.0094],
    "MostPop":       [0.0212, 0.0212, 0.0775, 0.0222, 0.1520, 0.0251],
    "UserKNN":       [0.0306, 0.0306, 0.1209, 0.0306, 0.2250, 0.0347],
    "ItemKNN":       [0.0394, 0.0394, 0.1217, 0.0353, 0.1828, 0.0337],
    "BPRMF":         [0.0406, 0.0406, 0.1278, 0.0350, 0.2149, 0.0356],
    "EASEr":         [0.0295, 0.0295, 0.1124, 0.0278, 0.1975, 0.0299],
    "LightGCN":      [0.0358, 0.0358, 0.1136, 0.0306, 0.1882, 0.0311],
    "GPT-4o":        [0.2796, 0.2796, 0.5889, 0.2276, 0.6897, 0.1948],
    "GPT-4o mini":   [0.0316, 0.0316, 0.2132, 0.0451, 0.3091, 0.0413],
    "GPT-3.5 turbo": [0.2298, 0.2298, 0.4217, 0.1281, 0.5902, 0.1229],
    "Llama-3.3 70B": [0.2293, 0.2293, 0.4985, 0.1693, 0.5922, 0.1359],
    "Llama-3.1 405B": [0.1975, 0.1975, 0.4165, 0.1294, 0.5119, 0.1039],
    "Llama-3.1 70B": [0.1302, 0.1302, 0.3828, 0.1095, 0.5148, 0.0969],
    "Llama-3.1 8B":  [0.0687, 0.0687, 0.2281, 0.0609, 0.3500, 0.0571],
}
PAPER_ORDER = ["hit_rate@1", "ndcg@1", "hit_rate@5", "ndcg@5", "hit_rate@10", "ndcg@10"]


def collect() -> pd.DataFrame:
    rows: list[dict] = []

    for method, vals in PAPER_TABLE3.items():
        rows.append({"family": "paper_reference", "method": method, "split": "file_order",
                     "history": "all", "matching": "paper", "n_users": 6040,
                     **dict(zip(PAPER_ORDER, vals, strict=True))})

    # Read per-run metrics.json rather than classical/summary.csv: it carries n_users
    # and stays consistent with the LLM path.
    for mpath in sorted((mem_out_dir() / "classical").glob("*/*/manifest.json")):
        man = json.loads(mpath.read_text(encoding="utf-8"))
        met = json.loads((mpath.parent / "metrics.json").read_text(encoding="utf-8"))
        rows.append({"family": "classical", "method": man["method"], "split": man["split"],
                     "history": "", "matching": "exact_item_id",
                     "n_users": met.get("n_users", 0),
                     **{m: met.get(f"mean_{m}") for m in METRICS}})

    rescore = mem_out_dir() / "llm_rescore.csv"
    if rescore.exists():
        df = pd.read_csv(rescore)
        for _, r in df.iterrows():
            rows.append({"family": "llm", "method": r["model"], "split": r["split"],
                         "history": r["history"], "matching": r["matching"],
                         "n_users": int(r["n_users"]),
                         **{m: r.get(m) for m in METRICS}})
    else:  # fall back to each run's own metrics.json (single matching mode)
        for mpath in sorted((mem_out_dir() / "llm").glob("*/*/*/manifest.json")):
            man = json.loads(mpath.read_text(encoding="utf-8"))
            met = json.loads((mpath.parent / "metrics.json").read_text(encoding="utf-8"))
            rows.append({"family": "llm", "method": man["model"], "split": man["split"],
                         "history": man["history"], "matching": man.get("matching", "paper"),
                         "n_users": met["n_users"],
                         **{m: met.get(f"mean_{m}") for m in METRICS}})

    df = pd.DataFrame(rows)
    for m in METRICS:
        if m in df:
            df[m] = pd.to_numeric(df[m], errors="coerce").round(4)
    return df


def to_markdown(df: pd.DataFrame) -> str:
    head = "| family | method | split | history | matching | n | " + \
           " | ".join(m.replace("hit_rate", "HR") for m in METRICS) + " |"
    sep = "|---|---|---|---|---|---:|" + "---:|" * len(METRICS)
    lines = ["# Candidate-free recommendation on MovieLens-1M", "",
             "HR@K / nDCG@K, macro-averaged over users. `paper_reference` = Di Palma et al.",
             "(2025) Table 3, for the reproduction check.", "", head, sep]
    for _, r in df.iterrows():
        cells = "".join(
            f" {r[m]:.4f} |" if pd.notna(r[m]) else " - |" for m in METRICS
        )
        lines.append(f"| {r['family']} | {r['method']} | {r['split']} | {r['history']} | "
                     f"{r['matching']} | {r['n_users']} |{cells}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    df = collect()
    if df.empty:
        print("No runs found under memorization/outputs/.")
        return
    order = {"paper_reference": 0, "classical": 1, "llm": 2}
    df = df.sort_values(
        by=["family", "split", "method", "history", "matching"],
        key=lambda s: s.map(order) if s.name == "family" else s,
    ).reset_index(drop=True)

    csv_path = mem_out_dir() / "results.csv"
    md_path = mem_out_dir() / "results.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(df), encoding="utf-8")

    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.max_rows", 200):
        print(df.to_string(index=False))
    print(f"\nWrote {csv_path}\nWrote {md_path}")


if __name__ == "__main__":
    main()
