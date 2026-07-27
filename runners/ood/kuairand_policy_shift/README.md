# KuaiRand standard-to-random OOD experiment

This directory implements one intentionally small experiment:

> Given pre-intervention standard-feed behavior, can GPT-5.4-mini rank future
> randomized exposures better than source-item popularity?

The locked scientific specification is [`PROTOCOL.md`](PROTOCOL.md).
The English metadata artifact and its provenance are documented in
[`TRANSLATION_GPT55_20260727.md`](TRANSLATION_GPT55_20260727.md).
The completed first-run results are reported in
[`RESULTS_20260727.md`](RESULTS_20260727.md).

## Locked comparison

| Source | Target | Purpose |
| --- | --- | --- |
| standard, 2022-04-08--21 | standard, 2022-04-23--30 / 2022-05-01--08 | temporal reference |
| standard, 2022-04-08--21 | random, 2022-04-23--30 / 2022-05-01--08 | temporal + exposure-regime OOD |

The initial experiment contains one baseline and two GPT conditions:

- **GPT-5.4-mini, observed history:** existing target-response listwise ranking
  with the latest 20 distinct source interactions and translated caption,
  cover text, and category metadata;
- **GPT-5.4-mini, no history:** the same model, prompt target, candidates, and
  metadata with the user-history block removed; this is an ablation/control,
  not another recommender baseline;
- **Pop:** the exact number of positive responses received by each item in the
  complete standard source log.

Each target group contains ten natural logged exposures. Candidate seeds are
`0,1,2`. No negative items are generated and target labels are never used to
select users or candidates.

The primary target pool contains 4,098 common-warm items. Every item has at
least 20 source exposures and at least five distinct source users. Exact target
user-item pairs are first exposures across all released tabs and exposure
regimes.

Ranking metrics are computed only for groups containing both labels. The
headline metrics are NDCG@5 and HR@1.

## Running

Build the fixed tasks, run Pop, and materialize the complete GPT prompt/cost
preflight without API calls:

```bash
uv run python -m runners.ood.kuairand_policy_shift.run \
  --methods popularity,llm_openai_vk_gpt54_mini_listwise_english \
  --candidate-seeds 0,1,2 \
  --output-dir outputs/ood/kuairand_policy_shift_v2
```

The first-suite CLI is intentionally locked to the primary `common_warm`
target pool; `full_pool` is not exposed as a runnable method-suite condition.
Add `--execute-llm` to use the VK proxy after the preflight has been written.
For the first cached smoke, add `--max-users 1 --candidate-seeds 0`; this makes
four GPT calls. Reusing the same output directory allows a later full run to
reuse those content-addressed checkpoints. Legacy paid-run checkpoints are
read only for the exact matching task, prompt, item order, model, request,
provider, scorer, and parser implementation, then migrated under the current
identity key without another API call.

Any failed GPT group prevents a `complete` status: partial coverage is recorded
as `complete_with_errors`, zero successful groups as `failed`, and combined
GPT-versus-Pop analysis is skipped unless both method runs are `complete`.

The runner writes `task_audit.json` before model scoring. The old exploratory
reports, builders, and audit scripts were removed; Git history remains the
source for those discarded pilots.
