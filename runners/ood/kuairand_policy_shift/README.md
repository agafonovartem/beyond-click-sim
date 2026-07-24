# KuaiRand exposure-policy shift

The forced-c3 task is an exploratory pilot and is not the protocol for the
next paid run. Its outcome-conditioned one-click/two-non-click candidate
construction is retained only for reproducibility. The runner now defaults to
the locked label-blind natural-exposure task. Its feasibility grid and exact
protocol are documented in
[`NATURAL_EXPOSURE_PROTOCOL.md`](NATURAL_EXPOSURE_PROTOCOL.md).

This runner evaluates whether a user-response model can transport behavioral
history between KuaiRand-Pure's standard-feed and inserted-random exposure
regimes.

The task is a full 2×2 square:

| Visible source history | Standard target | Random target |
| --- | --- | --- |
| Standard exposures | matched | transferred |
| Random exposures | transferred | matched |

Source history uses tab 1 exposures from 2022-04-22 through 2022-04-30.
Validation targets use 2022-05-01 through 2022-05-03; test targets use
2022-05-04 through 2022-05-08. The main task takes five label-blind natural
target exposures per user and policy. Missing interactions are never treated
as negatives.

The LLM sees released native-language captions (with cover text as fallback),
the released category hierarchy, and the source-history click label. It sees
opaque candidate aliases rather than raw user/video IDs, policy names,
timestamps, popularity, or target labels.

This is one-step response prediction, not off-policy evaluation:

- candidates come from separately logged exposures, not one simultaneous
  logged slate;
- probability scorers are evaluated on every natural exposure with user-macro
  log-loss and Brier score as the primary metrics;
- row-level ROC AUC, average precision, calibration intercept/slope, predicted
  mean, and observed CTR are secondary probability diagnostics;
- NDCG@5 and Hit@1 are secondary ranking diagnostics and are defined only for
  groups containing both click labels;
- standard-target and random-target absolute scores are not directly
  comparable because their candidate distributions differ;
- the natural shared cohort and candidate rows are selected from observation
  counts and exposure identity without consulting click labels;
- pretrained models may know released captions or public source videos.

The implemented competitors are:

- global, item, and user source-policy click-rate baselines;
- a fixed Beta(1,1) Laplace-smoothed CTR from the same five visible user
  labels, used as a scalar-history control;
- an exposure-aware logistic user/item-bias model trained on both clicks and
  non-clicks;
- positive-only ItemKNN and BPR, with explicit zero-score fallback and
  coverage diagnostics for users/items without positive source support;
- character n-gram caption/category history affinity;
- caption-plus-category LLM probability prediction.

The native standard and random source graphs are intentionally left at their
released sizes. Consequently, matched-versus-transferred collaborative gaps
measure robustness to the complete released regimes; they are not a
source-size-controlled causal estimate of policy alone. A balanced-source
sensitivity analysis is still needed for a stronger causal attribution.

## Commands

Run classical and content baselines:

```bash
uv run python -m runners.ood.kuairand_policy_shift.run \
  --protocol natural \
  --methods global_mean,user_ctr,user_ctr_laplace,item_ctr,exposure_logistic_bias,item_knn,bpr,text_affinity_caption_categories,text_affinity_categories_only \
  --task-seeds 0,1,2 \
  --history-controls observed
```

Preflight the full caption-aware `gpt-5.4-mini` probability experiment through
the VK proxy without inference:

```bash
uv run python -m runners.ood.kuairand_policy_shift.run \
  --protocol natural \
  --methods llm_openai_vk_gpt54_mini_probability_caption_categories \
  --task-seeds 0,1,2 \
  --history-controls auto \
  --hard-budget-usd 20 \
  --run-dir outputs/ood/kuairand_policy_shift/gpt54_mini_probability_full
```

Set `OPENAI_VK_PROXY_API_KEY`, inspect the emitted preflight, then add
`--execute-llm` to run. Use an explicit `--run-dir` for paid runs: the runner
stores atomic content-addressed group checkpoints there and safely resumes
without paying for completed groups again.

For the unsubsampled 540-user cohort, three correlated target resamples, and
all ten observed/no-history/permuted cells, exact preflight counts are 16,200
requests and 16,056,996 input tokens. Under the method registry's $0.75/M
input and $4.50/M output prices, the expected cost is $14.96, the maximum at
the configured 48-token response cap is $15.54, and the 10% guarded estimate
is $17.10. The completed experiment instead used 400 main users and 100
permutation-control users; see the results note for actual spend.

Before that full run, use this 20-user frontier-model format/cost check:

```bash
uv run python -m runners.ood.kuairand_policy_shift.run \
  --protocol natural \
  --methods llm_openai_vk_gpt56sol_probability_caption_categories \
  --task-seeds 0 \
  --history-controls auto \
  --max-users 20 \
  --hard-budget-usd 2 \
  --max-workers 8 \
  --run-dir outputs/ood/kuairand_policy_shift/gpt56_probability_pilot20
```

Its current exact preflight is 200 requests, 198,737 input tokens, and at most
9,600 output tokens: $1.23 expected, $1.28 capped, and $1.41 with the same 10%
guard. Add `--execute-llm` only after inspecting that preflight.

Natural-task item support requires eight exposure users in each full source
policy log and at least five users in each actual post-restriction training
graph. It never filters on source clicks. Positive source degree is an
analysis stratum only.

To reproduce the rejected forced-c3 pilot explicitly:

```bash
uv run python -m runners.ood.kuairand_policy_shift.run \
  --protocol forced-c3 \
  --task-seeds 42,43,44 \
  --min-positive-source-users-per-policy 1
```

Generate paired cross-method statistics:

```bash
uv run python -m runners.ood.kuairand_policy_shift.suite_analysis \
  --gpt-run-dir /path/to/gpt-run \
  --classical-run-dir /path/to/classical-run \
  --metric log_loss \
  --output-dir /path/to/gpt-run/suite_analysis
```

When the categories-only GPT ablation has a separate suite root, pass it with
`--categories-run-dir`. Probability reports must omit `--text-run-dir`
because text affinity emits ranking scores rather than probabilities; add the
text root only for NDCG/Hit-rate analysis.

The completed 400-user, three-seed run and its interpretation are recorded in
[`RESULTS_20260724.md`](RESULTS_20260724.md).

Suite contrasts are quality-normalized: positive means the first named method
or condition is better. Thus log-loss/Brier contrasts reverse the raw
left-minus-right sign, while NDCG contrasts do not.

Candidate seeds are correlated resamples for the same users, not independent
datasets. Mixed-label eligibility can vary by seed. Comparisons are therefore
paired within each eligible `seed × user` target group, then averaged over the
available eligible seeds inside each user before users are bootstrapped and
paired sign-flip tests are applied. The report exposes the minimum, mean, and
maximum eligible seeds per user rather than silently restricting analysis to
users whose targets are mixed under every seed. For the natural task, the Holm
family is code-declared before model scoring but is not externally
preregistered. The natural protocol is exploratory: its H/m choice was made
after a feasibility and aggregate label-distribution audit, but before any
model outcome or paid API response. It must not be described as a
confirmatory preregistered test.
