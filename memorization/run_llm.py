"""Candidate-free LLM recommendation: reproduce Di Palma et al. (2025) + ablations.

Their setup: the prompt carries the dataset name, the user id and the user's
training history (Fig. 3, transcribed verbatim in prompts.py); the LLM emits a
ranked list of 50 movie titles with no candidate set; generated titles are fuzzy
matched against the user's held-out test titles.

Defaults follow the paper: ALL users (no filtering), full training history,
temperature=0, top_p=1, frequency/presence penalty=0, seed=42, k=50.

Serving is OpenAI-compatible, so the same code drives either backend:
  --client vllm_local   local/remote vLLM   (--model = --served-model-name)
  --client openai       OpenAI API          (OPENAI_API_KEY read from .env)
See beyond_click_sim.llm_clients.make_llm_client for the full list.

Ablations: --split {file_order,random} (split sensitivity), --history {20,50,all}
(history sensitivity).

Requests are issued concurrently (vLLM continuous batching / API parallelism) and
results are keyed by user id, so output does not depend on completion order.
Raw generations are written to predictions.jsonl, which lets rescore.py re-evaluate
any matching rule or threshold later without re-querying the model.

Examples:
  # local vLLM, their exact condition, all 6040 users
  uv run python memorization/run_llm.py --client vllm_local \
      --model llama-3.1-8b-instruct --split file_order --history all

  # OpenAI (key from .env); start small — this is 6040 requests
  uv run python memorization/run_llm.py --client openai --model gpt-4o-mini \
      --split file_order --history all --concurrency 8 --limit 20
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from beyond_click_sim.llm_clients import make_llm_client

from common import load_eval_users, load_split, mem_out_dir, title_map
from metrics import (
    DEFAULT_KS,
    MATCHING_MODES,
    aggregate_user_metrics,
    hit_flags,
)
from metrics import ranking_metrics_from_hits
from prompts import PROMPT_VERSION, build_messages, parse_numbered_list

MAX_ATTEMPTS = 5
BACKOFF_BASE = 2.0


def sample_history(item_ids: list[str], history: str, *, rng: np.random.Generator) -> list[str]:
    """History to show: 'all', or a random subset of size 20/50.

    Users with fewer than the requested number fall back to all of their train
    items; the count is reported as `n_history_fallback` in the manifest.
    """
    if history == "all":
        return list(item_ids)
    h = int(history)
    if len(item_ids) <= h:
        return list(item_ids)
    idx = rng.choice(len(item_ids), size=h, replace=False)
    return [item_ids[i] for i in idx]


def call_llm(client, model: str, messages, *, args):
    """One chat completion. Returns (text, prompt_tokens, error).

    Retries with exponential backoff + jitter: OpenAI rate-limits aggressively at
    6040-request scale, and a transient 429 must not silently drop a user.
    """
    last_err = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                seed=args.seed,
                max_tokens=args.max_tokens,
            )
            usage = getattr(resp, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0)) if usage else 0
            return resp.choices[0].message.content, prompt_tokens, None
        except Exception as err:  # noqa: BLE001 — record and retry any serving error
            last_err = f"{type(err).__name__}: {err}"
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(BACKOFF_BASE**attempt + random.random())
    return None, 0, last_err


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=["file_order", "random"], default="file_order")
    parser.add_argument("--history", choices=["20", "50", "all"], default="all")
    parser.add_argument("--client", default="vllm_local",
                        help="vllm_local | openai | ollama_local | openai_vk_proxy (see llm_clients)")
    parser.add_argument("--model", default="llama-3.1-8b-instruct",
                        help="vLLM --served-model-name, or an OpenAI model id (e.g. gpt-4o-mini)")
    parser.add_argument("--k", type=int, default=50, help="length of the generated list")
    parser.add_argument("--ks", nargs="*", type=int, default=list(DEFAULT_KS))
    # Matching (see metrics.py). 'paper' reproduces Di Palma et al. exactly.
    parser.add_argument("--matching", choices=list(MATCHING_MODES), default="paper")
    parser.add_argument("--threshold", type=float, default=85.0,
                        help="fuzz.ratio cutoff; 85 reproduces their Table 3 (see README)")
    # Paper generation config (their Methodology section).
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--context-window", type=int, default=17408,
                        help="server max-model-len; used only to flag truncated prompts")
    parser.add_argument("--history-seed", type=int, default=0,
                        help="seed for sampling the 20/50 history subsets")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--eval-sample", action="store_true",
                        help="score only data/eval_users.csv instead of ALL users")
    parser.add_argument("--limit", type=int, default=None, help="cap #users (debugging)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ks = tuple(args.ks)

    titles = title_map()
    frame = load_split(args.split)
    frame["user_id"] = frame["user_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)
    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"]
    train_by_user = train.groupby("user_id")["item_id"].agg(list).to_dict()
    test_by_user = test.groupby("user_id")["item_id"].agg(list).to_dict()

    if args.eval_sample:
        users = [str(u) for u in load_eval_users()]
    else:
        users = sorted(test_by_user.keys(), key=int)  # all users, no filtering (paper protocol)
    if args.limit:
        users = users[: args.limit]

    client = make_llm_client(args.client)
    print(f"[{args.client}:{args.model}] {args.split} / history={args.history} / "
          f"{len(users)} users / concurrency={args.concurrency}", flush=True)

    lock = threading.Lock()
    done = {"n": 0}
    results: dict[str, dict] = {}

    def work(user_id: str) -> None:
        rng = np.random.default_rng(args.history_seed * 100003 + int(user_id))
        shown_items = sample_history(train_by_user.get(user_id, []), args.history, rng=rng)
        shown_titles = [titles.get(i, "") for i in shown_items if titles.get(i)]
        messages = build_messages(user_id, shown_titles, k=args.k)
        content, prompt_tokens, err = call_llm(client, args.model, messages, args=args)
        parsed = parse_numbered_list(content, max_items=args.k) if content else []
        with lock:
            results[user_id] = {"n_shown": len(shown_titles), "prompt_tokens": prompt_tokens,
                                "parsed": parsed, "raw": content, "error": err}
            done["n"] += 1
            if done["n"] % 200 == 0:
                print(f"  [{done['n']}/{len(users)}] generated", flush=True)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(work, users))
    gen_seconds = time.perf_counter() - t0

    # Score in deterministic user order, independent of completion order.
    per_user: list[dict[str, float]] = []
    records: list[dict] = []
    n_errors = n_fallback = 0
    prompt_tokens_seen: list[int] = []
    for user_id in users:
        r = results[user_id]
        test_items = test_by_user.get(user_id, [])
        test_titles = [titles.get(i, "") for i in test_items]
        n_errors += r["error"] is not None
        if r["prompt_tokens"]:
            prompt_tokens_seen.append(r["prompt_tokens"])
        if args.history != "all" and len(train_by_user.get(user_id, [])) <= int(args.history):
            n_fallback += 1
        flags = hit_flags(r["parsed"], test_titles, mode=args.matching, threshold=args.threshold)
        per_user.append(ranking_metrics_from_hits(flags, len(test_items), ks=ks))
        records.append({
            "user_id": user_id, "n_shown": r["n_shown"], "n_test": len(test_items),
            "n_parsed": len(r["parsed"]), "n_hits": int(sum(flags)),
            "prompt_tokens": r["prompt_tokens"], "error": r["error"],
            "parsed": r["parsed"], "raw": r["raw"],
        })

    agg = aggregate_user_metrics(per_user, ks=ks)
    agg["n_errors"] = n_errors
    agg["n_history_fallback"] = n_fallback
    agg["generation_seconds"] = round(gen_seconds, 1)
    if prompt_tokens_seen:
        agg["prompt_tokens_median"] = int(np.median(prompt_tokens_seen))
        agg["prompt_tokens_max"] = int(np.max(prompt_tokens_seen))
        agg["n_prompts_at_context_limit"] = int(
            sum(t >= args.context_window - 8 for t in prompt_tokens_seen)
        )
    else:
        agg["prompt_tokens_median"] = agg["prompt_tokens_max"] = 0
        agg["n_prompts_at_context_limit"] = 0

    model_slug = args.model.replace("/", "_").replace(":", "_")
    run_dir = mem_out_dir() / "llm" / model_slug / args.split / f"hist_{args.history}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": "memorization/candidate_free",
        "method": f"llm:{args.model}",
        "method_family": "candidate_free_llm",
        "dataset": "ml-1m",
        "split": args.split,
        "history": args.history,
        "client": args.client,
        "model": args.model,
        "k": args.k,
        "ks": list(ks),
        "matching": args.matching,
        "fuzzy_threshold": args.threshold,
        "generation": {
            "temperature": args.temperature, "top_p": args.top_p,
            "frequency_penalty": args.frequency_penalty,
            "presence_penalty": args.presence_penalty,
            "seed": args.seed, "max_tokens": args.max_tokens,
        },
        "history_seed": args.history_seed,
        "concurrency": args.concurrency,
        "context_window": args.context_window,
        "n_users": len(users),
        "user_set": "eval_sample" if args.eval_sample else "all_users_no_filtering",
        "prompt_version": PROMPT_VERSION,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(agg, indent=2) + "\n", "utf-8")
    # Row-level generations: local-only (git-ignored), consumed by rescore.py.
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n[{args.split} | hist={args.history} | {args.model} | n={len(users)}]")
    for k in ks:
        print(f"  HR@{k}={agg[f'mean_hit_rate@{k}']:.4f}  nDCG@{k}={agg[f'mean_ndcg@{k}']:.4f}")
    print(f"  errors={n_errors}  fallback={n_fallback}  "
          f"prompt_tok_max={agg['prompt_tokens_max']}  gen={gen_seconds / 60:.1f} min")
    if agg["n_prompts_at_context_limit"]:
        print(f"  WARNING: {agg['n_prompts_at_context_limit']} prompts hit the context window "
              f"({args.context_window}) and were truncated.")
    if n_errors:
        print(f"  WARNING: {n_errors} users failed after {MAX_ATTEMPTS} attempts; "
              f"they score 0 and are counted in n_errors.")
    print(f"Wrote {run_dir}")


if __name__ == "__main__":
    main()
