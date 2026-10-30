# DyTA4Rec offline adaptation

Implemented on 2026-10-29 for the shared MovieLens-1M interaction task. `DyTA4RecScorer` generates one three-branch ensemble and exposes its original fused scores for both ranking and validation-thresholded classification. Verification uses fake clients and embeddings; there are no experimental results from this implementation work.

The scientific specification is [dyta4rec_plan.md](dyta4rec_plan.md). The method implements Wanyan et al.'s *Temporal-Aware User Behaviour Simulation with Large Language Models for Recommender Systems*, CIKM 2025, [DOI 10.1145/3746252.3760878](https://doi.org/10.1145/3746252.3760878), with the explicitly reconstructed memory described below. It is not an exact reproduction of the complete simulator.

## Invocation

Run from the nested Python repository (`beyond-click-sim/`). Preflight validates the materialized task, reconstructs and verifies training event identities, checks metadata, and reports uncached costs and conservative prompt bounds. It does not construct a response client, resolve an embedding snapshot, download models, or generate text:

```bash
uv run python -m runners.in_distribution.interaction_prediction.methods.dyta4rec \
  --task ml-1m_cap20_eval_users1000_cg5_m1_seed0 \
  --output-dir outputs/dyta4rec/preflight \
  --preflight
```

A separately authorized inference run can use the same entry point without `--preflight`. Provision the existing `recagent` optional dependency and the MPNet snapshot beforehand: the provider uses `local_files_only=True`, CPU float32, 768 dimensions, no explicit normalization, and the encoder's 384-token limit. Record the exact response-model/serving revision and the resolved immutable embedding revision before reporting results.

```bash
uv run python -m runners.in_distribution.interaction_prediction.methods.dyta4rec \
  --task ml-1m_cap20_eval_users1000_cg5_m1_seed0 \
  --output-dir outputs/dyta4rec/qwen3-8b-rrf-seed0 \
  --model Qwen/Qwen3-8B \
  --model-revision YOUR_PROVISIONED_MODEL_REVISION \
  --embedding-revision YOUR_LOCAL_MPNET_COMMIT \
  --cache-dir outputs/dyta4rec/cache \
  --fusion rrf --seed 0
```

Opt-in registry names are `dyta4rec_offline_rrf_litellm_{qwen3_8b,qwen36_27b}_{smoke,full}`. Smoke caps each split at five whole groups; full uses the supplied groups. They are not added to any default experiment roster. The CLI exposes `--max-groups`, `--max-workers`, `--history-items`, `--icl-examples`, `--context-tokens`, `--temperature`, `--top-p`, and `--fusion bc`. The Python `run_method` accepts injected clients, embeddings, canonical datasets and `DyTA4RecConfig`; `fit` makes no model calls, and `build_agents` explicitly initializes requested users.

The source-like backbone setting remains configurable with `--client` set to an existing GPT-serving client, `--model gpt-4o-mini --temperature 0.1 --top-p 0.9`. This does not identify the author's historical model snapshot. Qwen defaults use temperature 0, top-p 1, and disabled hidden thinking. The request seed is recorded and passed to the provider; backend determinism is not guaranteed. Search orchestration is outside this method: configurations are selected externally by validation NDCG@5 separately per dataset/ratio/seed.

## Implemented architecture and provenance

- The temporal window is the latest 10 permitted training events, oldest first, including ratings. Stable event identities resolve timestamp ties; repeated interactions remain separate events. Three target-history next-item demonstrations are used, with a real zero-demonstration path and a short-history adjustment that retains context. This is not a ten-item total evidence budget.
- All permitted target training events supply long-term state. Blocks of 50 events are summarized, then consecutive sets of at most 10 summaries are merged recursively. Every block and merge, including the root, remains a separate record. Evidence references are validated against the corresponding inputs; schema checks cannot establish whether generated statements are factually supported.
- One importance call per record returns an integer 1–10, scaled by `0.9/10`. Records are embedded as summary prose, with creation/access times set to their latest supporting training timestamp. No summary is dropped for low importance.
- Long-term profile extraction uses the root summary, canonical age category/recorded gender/occupation category, and descriptive training statistics. Statistics include observed-event and item counts, rating histograms, per-genre counts/mean ratings, and population count/mean-rating associations over the user's events and each own-rating category. Empty categories are unavailable. Two separate calls construct recent behavioral and interest summaries. No traits, demographics, feelings or synthetic exposures are supplied as if observed.
- The population table is computed from the complete verified task training partition before selecting scored users. Candidates show title, genres, population training interaction count and population training mean rating; unseen items have count zero and unavailable mean. Task-provided all-data aggregates and held-out metadata are not used.
- Memory mode is `yulan_retrieval_two_level_v1`; profile version is `dyta_profile_v1`. The former recent-only proposal is unsupported and cannot satisfy current caches.

The inspected [DyTA4Rec revision `c4a0cc61c4a539b75cb426380261d6520859e84f`](https://github.com/WennyXY/DyTA4Rec/tree/c4a0cc61c4a539b75cb426380261d6520859e84f) imports unavailable custom `AgentMemory`/`AgentRetriever` dependencies. An audit of all 14 reachable commits found no implementations; [issue #1](https://github.com/WennyXY/DyTA4Rec/issues/1) was open with no comments on 2026-10-29. This concerns the inspected public history, not private code. The [recovery audit](../../../docs/paper_readings/dyta4rec_plan_2026_10_29/github_recovery.md) records the evidence.

We reuse only `RecAgentMemory.importance` and `retrieve` from the unchanged local port of the acknowledged [YuLan-Rec source at `fff935570079502d5a17509b0baf590becd1eaab`](https://github.com/RUC-GSAI/YuLan-Rec/blob/fff935570079502d5a17509b0baf590becd1eaab/agents/recagent_memory.py). The adapter constructs disposable records and does not invoke `observe`, `read`, `save`, reinforcement, sensory processing, reflection or forgetting. This is an upstream-derived substitution, not recovered DyTA4Rec memory.

For the exact label-free candidate observation, the nearest 100 summary vectors receive semantic relevance `1 - squared_L2_distance/sqrt(2)`; other records receive zero semantic contribution. Values are not clamped. Retrieval scores are `0.99**hours_since_access + importance + semantic_relevance`, at the target training cutoff plus one second. The top five records are selected, with stable insertion order for ties. Access-time updates are discarded with the group copy. The recent-history and long-term snapshot stay fixed across validation, test, retries and concurrent groups.

## One group and two evaluations

Each uncached successful group makes five generation calls, in this order:

1. History-only clustering analysis and `{yes,no,other}` flag.
2. Clustering-conditioned complete candidate ranking.
3. Sequential ICL ranking and its independent flag.
4. A separate history-only clustering analysis for the profile branch.
5. Profile/memory ranking using the retrieved summaries, fixed recent memories, profiles/statistics/demographics, reference clock and second analysis.

All branches use the same candidate order and opaque `C1...Cn` labels. Strict JSON requires every label exactly once and rejects unknown/duplicate fields, candidates and malformed flags. Positions carry no preference information. Even a zero-weight branch must succeed. No candidate pruning, extra bootstrap rounds or select-all path is introduced.

For branch order sequence/profile/cluster, weights are `(1,1,0)` when the cluster flag is `no` or `other`, `(.5,1,1)` when cluster is `yes` but sequence is low, and `(1,1,1)` when both are `yes`. The second clustering call's flag does not control fusion. Primary RRF uses `sum(w_b/(60 + rank_b))`, with one-based ranks. Optional BC uses `sum(w_b*(candidate_count - rank_b + 2))`, retaining the source's unusual offset while adapting its fixed list length. Exact rational arithmetic preserves mathematical ties; scores are neither normalized nor replaced by final integer ranks.

NDCG@5 and Recall@5 consume those exact fused scores with the shared average-tie rule. Group metrics are averaged within users and then across users. Binary predictions are `score >= threshold`, where validation F1 selects a single threshold over the same full requested validation groups. Threshold candidates include all-negative and all-positive endpoints; ties prefer the all-negative endpoint, then ascending score transitions, as in the existing evaluator. Classification adds zero LLM calls.

## Failures, artifacts and resumption

Initialization retries each failed stage up to five times and checkpoints successful stages. Ranking retries the whole five-call pipeline up to five times, starting from the same snapshot and stopping an attempt at its first failure. Valid low flags or inaccurate rankings never trigger retries. Incomplete ensembles are not cached as successes. Counts cover scorer-issued requests; any transport-level retries within a supplied client are outside this count.

A small method-local adapter uses unchanged shared metric primitives. Failed groups remain in the strict validation and reporting denominators. Missing outputs remain missing in saved rows; all-negative classification fallback is used only in strict summaries. Failed ranking groups get zero metric utility directly, not tied zero item scores. When validation wholly fails, the threshold and classification metrics are unavailable, while successful test rankings and all coverage/failure artifacts are still saved. A wholly failed test split also retains artifacts.

| Artifact | Contract |
| --- | --- |
| `predictions.parquet` | Original input rows and `source_row_index`, split, evaluator-appended target, unmodified fused `score`, nullable `prediction`, and status (`ok`, `score_failed`, `threshold_unavailable`) |
| `metrics.json` | Validation/test classification, fixed-population `*_failure_as_negative` summaries, success-only diagnostics, threshold and score/conversion coverage |
| `metrics_ranking.json` | NDCG@5/Recall@5, fixed-population `*_failure_as_zero_group` summaries and success-only diagnostics using original scores |
| `manifest.json` | Config/reconstruction/source fingerprints, verified training provenance, task/selection contract, resolved embedding settings, model/serving information, usage and cache counts |
| `preflight.json` | Permitted evidence/group counts, uncached call counts, metadata requirements and conservative prompt bounds |
| `dyta4rec_snapshots.json` | Immutable initialized user state, summary support, statistics, profiles, importance, embeddings and their provenance |
| `dyta4rec_trace.jsonl` | Stage/attempt identities, rendered inputs, returned text, token usage, retrieval terms, branch ranks, flags, weights, exact/floating scores and ties |
| `llm_errors.jsonl` | Initialization and exhausted-group errors |
| `evaluation_identity.json` | Guard against mixing different evaluations in one output directory |
| `dyta4rec_cache/` | Atomic JSON stage/snapshot/complete-ensemble checkpoints and the existing non-executable numeric embedding cache |

Use the failure-inclusive metric keys for reporting and configuration selection. `val`/`test` alone are success-only diagnostic fields, retained for artifact compatibility. When a threshold is unavailable, classification conversion coverage is zero even if ranking coverage is positive.

Inference cache keys include ordered permitted evidence, ratings, metadata/population statistics, clocks, prompt/stage identity, source versions, model/serving/decoding/seed settings and resolved embedding identity. No evaluation labels or hidden outcomes enter these keys. Evaluation identity includes labels separately. To recompute evaluation after changing labels or fusion settings, use a new output directory with the same `--cache-dir`. RRF/BC can reuse complete saved branches; thresholds and metrics are always recomputed.

Initialization costs `2*(B+R)+3` successful uncached LLM calls per user, where `B=ceil(N_train/50)` and `R` is the number of merge summaries; embedding activity covers `B+R` summaries plus group queries. One validation and one test group cost ten group-generation calls, independently of classification. Preflight reports pessimistic uncached costs; exact disk-cache compatibility is resolved only after local embedding metadata is available. UTF-8 byte token bounds plus 512 chat-overhead tokens and the response budget guard the declared Qwen/GPT context. Generated-text preflight bounds are pessimistic; every actual prompt is checked, and overflow fails explicitly without dropping evidence. Text/metadata field limits and MPNet encoder truncation are recorded separately.

## Claims and departures

The [manuscript](../../../paper_beyond_click_sim/www_short/main.tex) and implementation plan disclose the unavailable initialization/memory components, our summary/statistics/profile recipes, upstream component selection and MPNet substitution. Other differences are the shared fractional temporal split, multi-positive groups and sampled negatives, ten-history/three-ICL choice versus released six-ICL default, canonical genres/demographics with occupation mapped to the undefined source status field, no synthetic historical feelings, frozen independent group state, Qwen/decoding substitutions, strict JSON parsing, preserved fused ties, variable-length BC and added thresholded classification.

Fused scores measure relative slate preference, not calibrated interaction willingness. Raw adaptive weights change the score scale, and BC also depends on group size; thresholds are condition-specific. Population training aggregates do not establish a global historical cutoff. Unobserved sampled negatives are not verified rejections. Different methods retain different histories, metadata and call budgets, so the comparison concerns configured methods, not architecture alone. These tests establish implementation behavior, not predictive quality or online simulator fidelity.
