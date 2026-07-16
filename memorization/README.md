# Candidate-free recommendation: reproducing and auditing Di Palma et al. (2025)

This directory reproduces the recommendation experiment of **Di Palma et al. (2025),
*Do LLMs Memorize Recommendation Datasets?*** (Table 3) on MovieLens-1M, and then
re-runs it under a fair split to test whether the reported LLM advantage survives.

A *candidate-free* recommender emits, per user, a ranked list of 50 items with **no
candidate set**: classical policies rank the catalog, LLMs generate 50 movie titles
from the prompt alone. Two questions:

1. **Split sensitivity** — does the LLM advantage persist under a *random* per-user
   80/20 split, or is it an artifact of their *ratings-file-order* split?
2. **History sensitivity** — does candidate-free LLM recommendation need the full
   history? (20 / 50 / all training interactions)

These are standalone scripts by design; they reuse the ranking policies from
`beyond_click_sim.tasks.policies` and the MovieLens adapter, and touch nothing else.

## Reproduction status

Our setup reproduces their published numbers, which is what licenses the ablations:

| check | result |
|---|---|
| Our `file_order` split vs their released `training.tsv`/`test.tsv` | **byte-identical** — all 6040 users, same order, 0 differences |
| MostPop vs their Table 3 | **exact** to 4 decimals (0.0212 / 0.0775 / 0.1520 / 0.0251) |
| ItemKNN (`ItemKNNStd`, `n_neighbors=200`) | **MAE 0.0003** |
| Llama-3.1 8B (fp16, vLLM), their matching rule, threshold 85 | **MAE 0.0056** |

MostPop is the decisive probe: it has no hyperparameters and no RNG, so an exact
match proves the split, the candidate protocol (catalog minus train-seen) and the
metric definitions are theirs. Metrics **must** be computed on all 6040 users — the
paper says ML-1M "without any filtering".

Not reproducible exactly, and why:

- **Random** — their RNG/seed differs (0.0101 vs 0.0093; matches in expectation).
- **BPRMF / LightGCN** — produced with **Elliot** (their own framework; their repo
  references `data/movielens_1M/elliot/test.tsv` but ships no baseline code or
  config, and the paper lists no hyperparameters). A 59-config sweep
  (`tune_baselines.py`) never reaches their BPRMF (best HR@10 0.1697 vs 0.2149):
  `implicit`'s BPR is not Elliot's. **Our CF baselines are therefore weaker than
  theirs, which makes our conclusion conservative.**
- **UserKNN / EASEr** — not implemented here.

## Findings

### 1. The file-order split is a structured cross-user item holdout

Within a user, ratings-file order is **not temporal** (mean adjacent-pair
time-ordering 0.52 ≈ random; Spearman(position, timestamp) ≈ 0). It is a **fixed
global item permutation**: each item sits at a near-constant relative position in
*every* user's block (per-item `frac_pos` std **0.13** vs **0.29** for a true random
partition). So "last 20% by file order" holds out the *same items across users*,
collapsing their train popularity (train/true-popularity ratio std 0.31, min 0.0 —
held out for nearly everyone; random: 0.11) and destroying the collaborative signal
for exactly the held-out items. File-order train top-50 overlaps the true top-50 by
only 34/50 (Spearman 0.86); random overlaps 50/50 (Spearman 0.9994).

`prepare_data.py` records this diagnostic in `data/split_manifest.json`.

### 2. The ranking flips between splits

HR@10, all 6040 users, per-user train/test sizes identical across splits (only
*membership* changes):

| method | file_order | random | factor |
|---|---:|---:|---:|
| Random (control) | 0.0859 | 0.0841 | 1.0 |
| MostPop | 0.1520 | 0.6730 | 4.4 |
| ItemKNNStd | 0.1831 | 0.8980 | 4.9 |
| BPRMF | 0.1268 | 0.8263 | 6.5 |
| LightGCN | 0.1810 | 0.6818 | 3.8 |
| ALS | 0.1945 | **0.9142** | 4.7 |
| Llama-3.1 8B (paper matching) | **0.3558** | 0.2556 | 0.7 |
| Llama-3.1 8B (fair matching) | 0.5060 | 0.4543 | 0.9 |

On their split the LLM beats every classical baseline (1.8x on HR@10, 3x on HR@1) —
their claim reproduced. On a fair split the LLM loses to everything, including
trivial MostPop (2.6x) and ALS (3.6x on HR@10, 7.9x on HR@1). The LLM is nearly
split-invariant because it never uses the training signal; the **Random control is
flat**, so this is not "the random split is easier" — it is destruction of the
collaborative signal.

Reported honestly: file-order also *flatters* the LLM (its held-out items are more
popular — 12.2% in the global top-50 vs 10.3% for random), which is why the LLM
scores slightly higher there.

### 3. Their LLM matching rule has three defects

The paper never documents the matching procedure at all — the words *fuzzy*,
*threshold* and *similarity* never appear in that context (the only "80%" is the
train/test split). Their code holds three contradictory values: a dead default of
80 in `is_similar`, a function default of `0.85` in `evaluate_checkpoint`, and
`threshold = 1` (exact) in `__main__`. We recovered the real one from their numbers:
**0.85** (MAE 0.0056; 80 gives 0.0204, 100 gives 0.0090).

| defect | direction | effect (file_order, HR@10) |
|---|---|---|
| **article** — MovieLens stores "Matrix, The (1999)"; their `normalize_title` never moves the trailing article, so it matches an LLM's "The Matrix" at `fuzz.ratio` = **60**. 20.8% of the catalog (807/3883) is affected, skewed toward popular films; the LLM writes the leading form 63.7% of the time. | **deflates** | 0.3558 → 0.4283 (**+20.4% rel.**) |
| **no dedup** — one held-out title can be credited by many recommendations | **inflates** nDCG only (HR is an "any hit" indicator) | nDCG@10 0.0592 → 0.0579 |
| **out-of-catalog titles** — their prompt's format example ``(e.g., `1. Harry Potter')`` is echoed as rank 1 by **18.2%** of users; ML-1M ends in 2000 and has no Harry Potter film, so rank 1 is systematically burned | **deflates** | 0.3558 → 0.4598 |

All three hit the **LLM only** — their classical baselines are scored by exact
`item_id` with no title matching. With all three repaired (`fair`), Llama-3.1 8B
reaches HR@10 0.5060 on file-order and 0.4543 on random: **better than they report**,
which argues *against* our thesis and is stated as such. It does not rescue the
claim — ALS on the fair split still scores 0.9142.

Their own diagnostic hides the article bug: `check_absent_items` reports "The Matrix"
as *absent from the dataset* rather than as a matching failure.

Also a typo in their Table 3: Llama-3.1 8B has HR@1 0.0687 but nDCG@1 0.0697 —
impossible, since nDCG@1 ≡ HR@1 for binary relevance.

## Protocol

- **Split parity (load-bearing).** One cut per user, `n_train = floor(0.8·n)`
  (clamped so both sides are non-empty), applied two ways: `file_order` = last
  `n_test` rows in ratings-file order; `random` = seeded per-user permutation with
  the **same** counts. Only *which* items are held out changes, never how many.
  `splits.py` asserts per-user parity. `floor(0.8·n)` is verified to be their exact
  rounding (byte-identical split).
- **Users.** All 6040, no filtering (paper protocol). `--eval-sample` restricts to
  `data/eval_users.csv` (a fixed 200-user sample) for cheap debugging only.
- **Candidates.** Full catalog minus the user's train-seen items; relevance = the
  user's held-out test items; list length k=50.
- **Metrics.** HR@{1,5,10}, nDCG@{1,5,10}, macro-averaged over users.
- **Generation** (their Methodology): `temperature=0, top_p=1, frequency_penalty=0,
  presence_penalty=0, seed=42`. Prompt = their Figure 3, transcribed verbatim
  (`prompts.py`, `PROMPT_VERSION`).
- **Intended asymmetry.** Collaborative models are fit on the *full* train split
  (every user's data); the LLM sees one user's history plus its own prior. Classical
  scoring is exact-id, LLM scoring is lenient fuzzy matching. Both favour the LLM,
  so the fair-split conclusion is conservative.

## Setup

```bash
uv sync
cp .env.example .env       # then fill in what you need (see below)
```

`.env` (loaded automatically; it is git-ignored):

```
OPENAI_API_KEY=sk-...      # only for --client openai
```

vLLM needs no key. For a local server:

```bash
vllm serve <model-path> --served-model-name llama-3.1-8b-instruct --port 8000
# override the endpoint if it is not 127.0.0.1:8000:
export BEYOND_CLICK_SIM_VLLM_LOCAL_BASE_URL=http://<host>:8000/v1
```

Non-obvious constraints hit while serving Llama-3.1-8B **fp16** on 2x RTX 2080 Ti:
8B fp16 (~16 GB of weights) does not fit one 11 GB card, so `--tensor-parallel-size 2`;
Turing (sm75) has no bfloat16 (`--dtype float16`) and no FlashAttention
(`VLLM_ATTENTION_BACKEND=XFORMERS`, `VLLM_USE_V1=0`); the longest full-history prompt
is **15,540 tokens**, so `--max-model-len` must exceed ~17k; and TP=2 needs headroom
for NCCL buffers (`--enforce-eager`, `--max-num-seqs 16`, `--gpu-memory-utilization 0.92`).

## Pipeline

```bash
# 1. data: downloads ML-1M if absent, materializes it, builds both splits
uv run python memorization/prepare_data.py --n-eval-users 200 --seed 0

# 2. classical baselines (CPU), both splits, all 6040 users -> outputs/classical/
uv run python memorization/run_classical.py --split both

# 3. LLM, all 6040 users. vLLM:
uv run python memorization/run_llm.py --client vllm_local --model llama-3.1-8b-instruct \
    --split file_order --history all
uv run python memorization/run_llm.py --client vllm_local --model llama-3.1-8b-instruct \
    --split random --history all
#    OpenAI (key from .env) — this is 6040 requests, so try --limit first:
uv run python memorization/run_llm.py --client openai --model gpt-4o-mini \
    --split file_order --history all --concurrency 8 --limit 20

# 4. re-score the saved generations under every matching rule (CPU, no LLM calls)
uv run python memorization/rescore.py --modes paper article dedup in_catalog fair

# 5. final table
uv run python memorization/make_table.py        # -> outputs/results.csv, results.md
```

`run_llm.py` writes every raw generation to `predictions.jsonl`, so thresholds and
matching rules can be re-evaluated for free afterwards — never re-query the model to
change a metric.

## Files

| file | role |
|---|---|
| `prepare_data.py` | download/materialize ML-1M, build both splits + eval sample, record the file-order diagnostic |
| `splits.py` | per-user parity splits (`make_parity_splits`) |
| `metrics.py` | HR/nDCG, the `paper` matching rule and its corrected variants |
| `prompts.py` | Figure 3 prompt, verbatim + list parser |
| `baselines.py` | `StandardItemKNN` (see below) |
| `run_classical.py` | classical policies on both splits |
| `run_llm.py` | candidate-free LLM (vLLM or OpenAI) |
| `rescore.py` | re-score saved generations: thresholds, matching modes, scorers |
| `tune_baselines.py` | hyperparameter sweep against their Table 3 |
| `make_table.py` | build `outputs/results.csv` + `results.md` |
| `common.py` | paths and loaders |

**Why `baselines.py` exists.** `src/beyond_click_sim/tasks/policies.py::ItemKNNPolicy`
scores a candidate by walking the top-k neighbours of each item in the *user's
history*. Classic item-based KNN (and Elliot's) restricts to the top-k neighbours of
the *candidate* item; the two differ because the neighbour relation is asymmetric,
and the src variant lands at HR@1 0.0248 vs the paper's 0.0394. `StandardItemKNN`
implements the classic formulation and reaches MAE 0.0003. `src/` is shared with
other experiments and is deliberately left untouched.

## Outputs

Committed (compact provenance): `outputs/results.csv`, `outputs/results.md`,
`outputs/classical/summary.csv`, every run's `manifest.json` + `metrics.json`,
`data/split_manifest.json`, `data/eval_users.csv`.

Git-ignored (large / row-level): `data/split_*.csv`, `outputs/llm/**/predictions.jsonl`.
Notebooks or re-scoring that need the generations require a local run.

## Caveats

- Single model so far (Llama-3.1 8B fp16). The 70B/GPT rows of their Table 3 are
  reproduced only as `paper_reference`.
- `history=50` has not been run (the GPU host rebooted mid-grid); `20` and `all` are
  complete on both splits.
- `LightGCN` is a pure-NumPy reference at default hyperparameters, not a tuned SOTA;
  do not over-claim from its absolute value.
- The fair-matching variants are our own correction, not something the paper defines.
