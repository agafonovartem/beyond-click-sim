# SimUSER(zero, text-only) interaction scorer

Implemented 2026-10-29 in [`SimUSERYesNoScorer`](../../src/beyond_click_sim/scorers/simuser/scorer.py), with an opt-in [interaction runner](../../runners/in_distribution/interaction_prediction/methods/simuser_yes_no.py). This is a text-only, history-initialized adaptation of Bougie and Watanabe, *SimUSER: Simulating User Behavior with Large Language Models for Recommender System Evaluation* ([arXiv v1](https://arxiv.org/abs/2504.12722v1), [ACL publication](https://aclanthology.org/2025.acl-industry.5/), DOI `10.18653/v1/2025.acl-industry.5`). It implements the agreed [plan](simuser_plan.md); fake-client verification is not an experiment or evidence of simulator quality.

Supported claim:

> We implemented a text-only, history-initialized adaptation of SimUSER's persona-matching, dual-memory retrieval, and iterative watch/skip response architecture. We reconstructed underspecified retrieval and prompting details and evaluated independent candidate groups under our interaction-occurrence protocol. The same native binary decisions supply classification predictions and tied ranking scores.

This is not an exact reproduction, the full interactive simulator, or reproduction of a particular row in the source's Table 1. There is no finetuning. Images/visual checks, simulated warm-up, subsequent ratings, navigation, reflection, exit, and cross-group memory updates are omitted. Backbone/history-budget searches and experiments remain outside this implementation.

## Invocation

Run from the Python repository root (`beyond-click-sim/`). Preflight reads local data and writes `preflight.json`; it constructs no service clients and sends no requests:

```bash
.venv/bin/python -m runners.in_distribution.interaction_prediction.methods.simuser_yes_no \
  --task ml-1m_cap20_eval_users1000_cg5_m1_seed0 \
  --output-dir outputs/simuser_preflight \
  --preflight
```

The following is an example experiment invocation, **not executed during implementation**. It requires a configured OpenAI-compatible chat endpoint and embedding credentials:

```bash
.venv/bin/python -m runners.in_distribution.interaction_prediction.methods.simuser_yes_no \
  --task ml-1m_cap20_eval_users1000_cg5_m1_seed0 \
  --output-dir outputs/simuser_ml1m_seed0 \
  --client litellm_local --model Qwen/Qwen3-8B \
  --model-revision YOUR_PINNED_MODEL_AND_SERVING_REVISION \
  --embedding-client openai \
  --embedding-identity YOUR_TEXT_EMBEDDING_3_SMALL_DEPLOYMENT_ID \
  --max-workers 4
```

Use `--splits test` for an externally fixed configuration. The default is `val test`, initialized together from the same training snapshot. The seed defaults to the materialized task's sampling seed; `--seed` overrides it. Configurations can therefore follow repetitions 0/1/2 without a separate scorer or search controller.

Opt-in registry names are `simuser_zero_text_only_litellm_{qwen3_8b,qwen36_27b}_{smoke,full}`. Smoke limits each requested split to five supplied groups. Full uses all supplied groups. Default expensive sweeps are unchanged. Qwen wrappers disable hidden thinking. All cognitive stages use the same declared chat backbone; the reference model can be selected with `--client openai --model gpt-4o-mini`. Embeddings use `text-embedding-3-small`; alternate clients/models require explicit identities, with no fallback or downloads. Generation uses temperature 0, stage-specific token limits, and a deterministic provider seed. API users can disable provider-seed forwarding with `SimUSERConfig(provider_seed=False)` if a provider does not support it.

Specify `--context-tokens` or the local-serving `BEYOND_CLICK_SIM_VLLM_MAX_MODEL_LEN` to check a conservative UTF-8 byte/token upper bound. This includes bounded generated text, all candidates and maximum expanded evidence; it is not measured tokenizer usage and can reject a configuration that a tokenizer would fit. When capacity is unspecified, preflight reports it as **unverified**, not sufficient. Exact request byte bounds are also checked before calls when capacity is configured. No candidates or retrieved records are silently removed to fit context. Serving revisions and context capacity must be set to the real deployment values.

## Data and architecture

Only MovieLens-1M temporal interaction tasks are supported. Preparation reuses the existing exact training-projection verification without changing it. The population consists of the complete permitted training partition, even when only one evaluation user is scored. It does not import canonical users excluded by the task filter. The runner uses an ID-checked copy of the permitted static catalog and leaves task frames untouched. All model-visible fields are explicitly selected: evaluation labels, held-out ratings/timestamps, sampling flags, group split names, true positive counts, and actual user demographics are never rendered.

Required catalog fields are title, canonical genres and frozen short summary. Metadata enrichment uses catalog-only fields, not candidate outcomes. Titles are bounded to 200 characters and summaries to 600; these rendering limits are versioned in the configuration. Quality is the mean observed training rating; popularity is the observed training interaction count. Unobserved catalog items have count zero and unavailable quality. No external augmentation quality/popularity fields are consumed. Missing required metadata fails preparation.

The current ML-1M summary source is Agent4Rec's frozen `movies_augmentation.csv`, commit `f4ee73e4fc686ef6177e9554d96ecf50add952b0`, SHA-256 `b8e791dc6e77245f6d29825ee08c0916a8a55344ef04ee76ef5686138f7d96d5`. The local file hash was rechecked during implementation. Each run records the actual canonical hashes and summary-source manifest. These are textual summaries, not SimUSER author artifacts or visual captions.

Initialization follows these steps:

1. Sample 50 training events per user without replacement (all if fewer), then render them chronologically. Create a taste summary and five distinct personas with age, occupation and Big Five integer scores 1–3. Names, gender and true demographics are omitted. Persona age/occupation menus are reconstructed defaults recorded in the manifest.
2. Form `j=3` target/other-history contrasts with `rho=min(10, distinct target items, eligible contrast items)`. Contrast items exclude the target's training items, not held-out membership. Choose one stable representative event per item, sample unique items within each subset and independently mix target/contrast order. All five personas see the same three draws. They return hypothetical 1–5 ratings without observed ratings or membership tags. Select the largest sum of target ratings minus contrast ratings; ties choose the earliest persona. These contrasts are not ground-truth dislikes.
3. Compute habits from the full user history: event count, mean squared rating deviation from population training item means, distinct genre count, and pickiness (`mean>=4.5`: not picky; `>=3.5`: moderately; otherwise extremely). Larger deviation is not greater conformity. Store an immutable snapshot with all persona scores, sampled IDs and provenance.
4. Build one episodic record per user training event. Preserve the source inconsistency: rating 4 is liked in summary/graph, but neutral in initial episodic text (`>4` liked, `<=2` disliked). All recorded ratings remain positive interactions for evaluation.

The real training graph includes user–item observed and typed preference relations, plus item–genre edges. Cast paths are omitted. The executable reconstruction is `simuser_graph_v1`:

```text
M_UI = B.T @ B                     # binary observed training membership
M_G  = C @ C.T                     # item–genre membership
PathSim_M(x,y) = 2 M[x,y] / (M[x,x] + M[y,y]), or 0 for zero denominator
S_item(x,y) = .5 PathSim_UI(x,y) + .5 PathSim_G(x,y)
S_user(u,y) = mean_h_in_unique_training_history S_item(h,y)
S_sem(x,y) = (1 + cosine(embedding(x), embedding(y))) / 2
R(u,x,y) = .75 * (.8 S_item(x,y) + .2 S_user(u,y)) + .25 S_sem(x,y)
```

Binary sparse matrices and bounded on-demand products implement these formulas. Item embeddings serialize title, genres and summary. There is no learned graph model. History projection replaces the source's unspecified heterogeneous user–item normalization; path catalogue, channel weights and convex fusion are disclosed reconstructions. A missing collaborative channel retains its zero contribution. Stable typed item IDs break retrieval ties. Retrieved evidence includes up to two actual witnesses per structural channel and user-prefixed witnesses for history projection; semantic-only evidence has no fabricated paths. Training feedback and population mean ratings remain separately labeled.

Each independent group starts from the same snapshot:

1. Form a query from the full supplied candidate group and interaction task; ask for exactly two follow-up questions. Search episodic memory separately for each of the three queries, initially five records per query. Merge by event ID, retaining maximum cosine and stable ties.
2. Retrieve three graph items per supplied candidate, excluding that candidate itself; other group candidates remain eligible. Deduplicate display records while retaining query-to-evidence mappings and all retrieval terms in the trace.
3. Generate a complete watch/skip vector, then run a separate consistency check. Initial, checking and revision prompts all use **interaction occurrence**, including the explicit rule that anticipated dislike does not logically contradict WATCH. No rating threshold or selection quota is imposed.
4. When the check requests revision, expand each episodic prefix by two and each graph prefix by two, then revise and check again. Stop at no requested revision, exhausted evidence, or two refinements. Keep the last complete valid vector and record unresolved flags. Semantic disagreement is not a technical failure. Strict JSON validates all candidate/evidence IDs, Boolean decisions and bounded reasons; at most two evidence citations per decision/check are allowed.

Initialization takes 17 successful uncached chat calls/user at `j=3`. A group takes 3–7 calls, plus embedding requests. Classification and ranking add zero generation calls. Initialization stages have up to five attempts and checkpoint valid results. Groups use the existing five-attempt whole-pipeline policy; no nested stage retries occur inside a group attempt. Embedding errors produce unavailable initialization/group outputs through the same existing failure accounting; valid cached embedding batches survive retries. There is no implicit transport retry introduced by this implementation.

The same trained state serves validation and test; outputs do not update taste, memory or graph. Other users' permitted training evidence can postdate a target's training cutoff under the shared static protocol. SimUSER does not add LLMSRec's separate target-time restriction or claim globally chronological deployment simulation.

## Output contract and selection

`predictions.parquet` preserves supplied row/user/item/group identity, split and evaluator-owned target, plus `task_row_index`, `score`, and `prediction`. WATCH maps to `1.0/True`, SKIP to `0.0/False`. Missing outputs remain null. There is no validation threshold or separate ranking inference.

`metrics.json` contains native binary F1/precision/recall for each requested split. `metrics_ranking.json` uses the same saved scores for NDCG@5 and Recall@5, preserving expected-average ties and group-then-user aggregation. Binary scores cannot order within WATCH or SKIP levels. For external per-seed configuration selection use `val.macro_by_user_group_mean["ndcg@5"]`; report the corresponding test view from the same selected configuration. The runner does not select models or orchestrate searches.

Successful-output fields may drive selection/reporting when actual requested-row/group coverage is complete. `manifest.json` records this condition in `successful_metrics_equal_strict`. Otherwise use the existing strict views: `<split>_failure_as_negative` for classification and `<split>_failure_as_zero_group` for ranking. Failed ranking groups receive zero utility directly; successful all-SKIP groups retain tied-list utility. Initialization failures stay in the requested population. Do not replace users or drop failures after observing responses.

Additional artifacts:

- `manifest.json`: exact method/configuration/source identities, adaptations, initialization coverage, per-split coverage, call/token usage, embedding statistics and serving metadata.
- `preflight.json`: evidence and metadata coverage, requested groups, uncached generation bounds and context-capacity status.
- `llm_errors.jsonl`: initialization and exhausted-group failures.
- `simuser_trace.jsonl`: exact messages/options/responses, stage/attempt/round IDs, usage, timing, selected persona scores, retrieval terms/witnesses, checks and stop reasons.
- `simuser_cache/agents/*.json`: checksummed immutable snapshots; `stages/*.json`: completed initialization responses; `groups/*.json`: complete group responses with queries, retrieved evidence and exact request records; `embeddings.sqlite`: exact-input normalized embedding cache. Cached-run traces retain this inference provenance. The in-memory group cache is bounded to 16 entries.

Generation cache identities include effective training evidence, metadata, configuration, implementation/prompt/model/embedding versions, exact messages/options, local seeds and ordered candidates as appropriate. The manifest includes actual scorer-module and runner SHA-256 hashes, so an uncommitted implementation is not identified by Git commit alone. Full canonical source hashes are recorded separately; hidden outcome changes do not enter effective inference identities. A pinned `model_revision` enables durable generation-cache reuse; without it reuse is limited to the current scorer session. The shared embedding utility receives a SimUSER-specific namespace and path without changing LLMSRec behavior. Fixed seeds stabilize inputs, not hosted generation bit-for-bit. Token totals are reported usage only; the manifest flags incomplete usage. Summed request durations are not parallel wall-clock duration.

## Verification and boundaries

Fake-client tests cover persona score mapping, random-50 versus full-history evidence, the rating-4 asymmetry, hand-calculated graph channels/fusion/witnesses, retrieval expansion, all stopping paths, target-consistent refinement, strict parsing, snapshot/cache isolation, hidden-label boundaries, per-split combined artifacts, and failure-inclusive metrics. Tests also exercise restart after partial initialization and changes in effective evidence/model identity. Existing-method regression checks establish behavioral isolation against the starting workspace. No live LLM experiment, model download, package installation or manuscript edit was performed.

Source ambiguities remain explicit: persona menus/contrast construction, graph schema and user projection, self-ask merge rules, bounded refinement, strict prompting/caching and token limits are reconstructed. Text-only is an architectural ablation; Qwen is a backbone substitution. The comparison evaluates configured methods with different metadata, histories, pretraining and compute budgets. Sampled unobserved negatives are not verified rejections, and this evaluation does not establish psychological accuracy or full simulator validity.
