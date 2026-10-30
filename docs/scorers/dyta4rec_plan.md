# DyTA4Rec: implementation plan for the shared interaction protocol

Status: implemented and verified with fake clients/embeddings on 2026-10-29. See [implementation and invocation documentation](dyta4rec.md). The selected `yulan_retrieval_two_level_v1` reconstruction supersedes the earlier recent-only proposal. The implementation adds isolated scorer/runner files and opt-in exports/registrations; shared method/evaluator behavior is preserved. Verification passed 79 DyTA4Rec checks and 328 existing-method/evaluator checks (407 total). No experiments, model downloads, paid calls or empirical results were produced. Reconstruction defaults remain our operational choices rather than recovered DyTA4Rec settings. The specification below records the accepted architecture and its disclosure requirements.

## 1. Scope, evidence, and permitted claim

Implement one **DyTA4Rec offline ranking scorer** for MovieLens-1M. Preserve the sequential ICL branch, two-step temporal-clustering branch, profile/memory branch, and adaptive rank fusion. Evaluate the resulting scores with both ranking metrics and validation-thresholded classification. No second inference pipeline or separate binary scorer is needed.

Sources:

- Wanyan et al., *Temporal-Aware User Behaviour Simulation with Large Language Models for Recommender Systems*, CIKM 2025, DOI [10.1145/3746252.3760878](https://doi.org/10.1145/3746252.3760878). The requested [local PDF](../../../papers/Temporal-Aware%20User%20Behaviour%20Simulation%20with%20Large%20Language%20Models%20for%20Recommender%20Systems.pdf) is arXiv:2509.16895v1, 21 September 2025. Relevant locations: §§2.1–2.4, Figure 1, §§3.1 and 3.4, Table 1, Figure 2.
- The supplied [architecture dossier](../../../docs/short_paper_selection/architecture_implementation_2026_10_25/reports/dyta4rec.md), cross-checked against a [fresh independent paper reading](../../../docs/paper_readings/dyta4rec_plan_2026_10_29/reports/dyta25.md).
- The [author repository](https://github.com/WennyXY/DyTA4Rec), including fresh snapshots of [the agent](../../../docs/paper_readings/dyta4rec_plan_2026_10_29/sources/dyta25/author-agent.md) and [the simulator/repository root](../../../docs/paper_readings/dyta4rec_plan_2026_10_29/sources/dyta25/author-repository-root.md). A subsequent [GitHub recovery audit](../../../docs/paper_readings/dyta4rec_plan_2026_10_29/github_recovery.md) resolved the matching published head to `c4a0cc61c4a539b75cb426380261d6520859e84f`; correspondence to the paper's experiments remains unverified.

The complete paper is available. The source gaps concern implementation: profile preprocessing/formatting, custom memory/retriever, evaluator, data loader, and configuration are absent from the inspected release. The paper also leaves their detailed algorithms unspecified. Preserve this distinction; an old report is not evidence that the implementation exists.

The recovery audit checked all 14 reachable commits on the sole published branch and found no missing memory/retriever implementation. [Author-repository issue #1](https://github.com/WennyXY/DyTA4Rec/issues/1) independently reports these missing classes and was open with zero comments when checked. YuLan-Rec, acknowledged by DyTA4Rec, publishes `RecAgentMemory`/`RecAgentRetriever`; these provide a concrete upstream reconstruction reference, not verified copies of DyTA4Rec's custom classes. Section 5 now uses their retrieval and importance components under pinned provenance, while retaining an explicitly reconstructed two-level history memory.

Permitted implementation claim, verified at the code/fixture level (not an empirical reproduction claim):

> We adapted DyTA4Rec's sequential and clustering-based temporal ranking, profile-conditioned ranking, and self-adaptive rank fusion to independent interaction-prediction groups. We reconstructed its underspecified profile initialization and two-level memory from permitted training histories, using retrieval and importance-scoring components from the acknowledged YuLan-Rec upstream implementation. These components are not verified copies of DyTA4Rec's unavailable custom memory. The same fused scores support ranking evaluation and validation-thresholded binary predictions.

Do **not** claim an exact reproduction, an implementation differing only in ranking-to-classification conversion, replication of the original result table, or validation of evolving user trajectories. The memory reconstruction in Section 5 is an explicit architectural substitution for an unavailable component; it is not forced by the common evaluation protocol. No LLM finetuning is added.

Out of scope: Steam and other datasets; full browsing/search/exit simulation; generated post-watch ratings/feelings; policy generation or retraining; live feedback updates; original-paper experimental reproduction; new hyperparameter/backbone-search orchestration; equalizing history/metadata budgets across methods; experiments or paid calls. The manuscript describes the implemented adaptation; it must not claim empirical validation or exact reproduction.

## 2. Shared evaluation contract

Follow the [current manuscript](../../../paper_beyond_click_sim/www_short/main.tex) and the materialized interaction task:

- Require an explicit per-user chronological split, primarily 70/10/20 train/validation/test. All recorded ML-1M ratings count as positive interactions, including low ratings. Historical rating values inform preferences but never redefine the target as rating ≥4.
- Use the shared cohort, targeting 1,000 eligible users, and its exact supplied validation/test candidate groups. Groups contain at most 20 candidates, at most five groups per user, and ratios 1:1, 1:3, 1:9, 1:19. Multi-positive and smaller final groups are valid. Do not substitute the paper's one-positive/nine-negative sampler.
- Keep the supplied candidate order identical across the three branches. Tell the LLM positions carry no preference information. Do not add LLMRank's three shuffled rounds or generate/prune candidates.
- Exclude held-out labels, ratings, timestamps, sampled-negative flags, split names, sampling ratios, and true positive counts from model inputs and inference caches. Use an explicit allowlist rather than serializing task rows. Candidate labels are local opaque identifiers, not raw dataset IDs.
- Build profiles, memories, examples, and statistics from permitted training evidence only. Validation outcomes never update the test state. All groups use the same immutable user snapshot; predictions and retrieval must not alter later groups.
- Evaluate the **original fused scores** with NDCG@5 and Recall@5, using the existing average-tie rule. Average groups within users and then users. Select one global classification threshold per dataset/ratio/seed by validation F1 with the same group/user aggregation, and apply it to those same scores.
- Configuration selection, when run externally, is per dataset/ratio/seed by validation NDCG@5, including failures in the fixed denominator. Expose validation artifacts for that selection. The scorer does not search backbones or silently choose the best test configuration. Seeds 0, 1, and 2 follow the common repetition setup.
- Check eligibility before inference and freeze the cohort. Keep existing retries, coverage, and strict failure accounting. Do not replace users after seeing outputs. Raw failures remain missing; strict classification uses the established all-negative fallback only in its summary; failed ranking groups receive zero metric utility directly.

Population statistics use the complete permitted training partition of the task, not only scored users and not otherwise excluded canonical users. This static protocol does **not** establish a global historical cutoff: another user's training event may occur after the target user's cutoff. DyTA4Rec retrieves no neighbor histories, but its item aggregates retain this shared limitation. Do not claim that temporal ordering alone establishes globally historical inputs. A broader as-of protocol would need consistent changes across comparison methods; it is outside this plan.

Sampled unobserved negatives are not verified rejections. The experiment measures recovery of logged interactions under the declared sampling, not natural acceptance prevalence or causal response to exposure.

## 3. Inputs, metadata, and explicit defaults

Reuse `require_temporal_task` and `prepare_training` from `runners/in_distribution/interaction_prediction/methods/_llmsrec_syn_data.py` unchanged when their contracts fit. Validate the entire training projection and recover stable event identities/timestamps; do not reconstruct evaluation groups. Join ratings and user/item metadata using verified identities: the helper's event table does not itself provide the historical rating values. Reject sampled training negatives as experiences.

Sort events by timestamp and stable event identity, preserving repeated events. Define the target's cutoff as its last training timestamp. A deterministic reference time of cutoff + one second replaces the author reaction wrapper's wall-clock date; format it in UTC and never use a validation/test timestamp.

| Input | Planned treatment | Source status / adaptation |
| --- | --- | --- |
| Recent history | Most recent **10** permitted interactions, oldest to newest | Author runtime buffer is 10; paper studies 5/10/15/20/25 and uses 10 for the ICL sweep. Main-table setting is not fully identified |
| Sequential demonstrations | Last **3** items of that window, narrated as successive next-item examples | Paper §3.4's best tested setting; released code defaults to 6. This choice is explicit, not a claim about the main table |
| Long-term evidence | All permitted target-user training interactions, summarized in bounded chronological blocks | Long-term initialization is described but its evidence budget and extraction algorithm are unavailable |
| User information | Canonical ML-1M age **category**, recorded gender, occupation category | Basic demographics are paper-supported. Do not invent exact ages from buckets. The code's `status` field is undefined; occupation is our documented mapping choice, not a verified synonym |
| Historical item information | Title, genres, observed rating 1–5; event order | Genre rendering is an explicit resolution of unspecified author item strings. Do not add fabricated feelings |
| Candidate information | Title, genres, training interaction count, training mean rating, clearly labeled as population statistics | Source describes popularity/high-rating tendencies; exact item serializer is unavailable. These definitions and their placement are reconstructed |
| Profile statistics | Own rating histogram, genre counts, and rating/popularity/quality associations defined below | Paper describes statistical patterns but gives no executable formulas or bins |
| Extra metadata | No ZIP code, fabricated names, reviews, plot summaries, images, or external popularity/rating values | These are not established inputs of the inspected ranking method. Their absence is not a claim that the protocol forbids them |

Keep timestamps for chronology/provenance and the reference clock; do not add elapsed-time features or timestamp embeddings to the temporal prompts. The source's runtime history strings do not contain separate timestamps, and the evaluator's formatting is unavailable. Here the temporal branches reason over ordered recent history; no learned time-decay or numerical clustering model is introduced. Memory retrieval separately uses the explicit fixed recency term in Section 5.

Short histories use `h=min(10, available_events)` and `k_eff=min(3,max(0,h-1))`, preserving at least one context item. Reject empty history during preparation. Support a declared `icl_examples=0` with a genuine no-demonstration path; do not copy the author's Python `[:-0]` slicing bug. History length and ICL count remain configurable and recorded, without an automatic sweep.

Report separately the 10-event temporal window, full training history used for long-term state, and population training evidence used for statistics. Calling this a “10-item total information budget” would be incorrect. Equal-budget comparisons remain deferred.

Missing mandatory metadata must be reported before calls. Existing missing values can be rendered explicitly as unavailable where the contract permits them; do not fill them with held-out outcomes. For an unseen candidate, count is zero and mean rating is unavailable. Any additional cohort eligibility restriction must be applied by shared cohort preparation, not as DyTA-only post-hoc row dropping.

## 4. Profile initialization and short-term updater

Implement versioned `dyta_profile_v1`; do not reuse Agent4Rec's trait extraction, percentile tiers, or RecAgent's full cognition merely because those packages already exist. The explicitly selected upstream retrieval/importance components in Section 5 are the limited exception, supported by source provenance.

Compute deterministic descriptive statistics from the allowed target history and allowed population training table:

1. Number of observed events and distinct items; own-rating histogram for 1–5; mean own rating.
2. Per-genre observed-event count and mean own rating, with each multi-genre event contributing once to each listed genre; distinct observed genres. Name these as observed statistics, not natural activity or personality measurements.
3. For every item, training event count `c_i` and mean rating `q_i`. Over the target's observed events, report mean/median `c_i` and `q_i`, plus their means within each own-rating category. Empty categories are unavailable, not zero. These associations operationalize the source's popularity/quality tendencies without unsupported “high/low” thresholds.

Counts and means include only permitted observations and remain independent of evaluation labels. Do not copy all-data augmentation statistics. These exact formulas are **our reconstruction**; the source does not establish them.

Build the long-term memory summary described in Section 5, then make one profile-extraction call conditioned on that summary, descriptive statistics, and available demographics. Require structured fields for stable movie tastes, inferred consumption/personality characteristics, rating/viewing tendencies, evidence references, and uncertainty. The LLM must distinguish observations from tentative inferences and may state insufficient evidence. Do not force Big Five values or infer missing demographic facts. The source's extraction prompts/schema are unavailable; save our prompt/schema version and the complete profile inputs.

Make two separate short-term calls over the recent window, following author `get_recent_summary_from_interactions` and `get_recent_interests_from_interactions`: a 2–3 sentence behavioral summary and a 2–3 sentence recent-interest summary. Preserve the focus on recent activity and genres/themes/tones/formats, without listing movie names or numerical ratings in these summaries. Explain the historical 1–5 rating scale in the input. Output evidence references separately if used; never insert hidden candidate outcomes.

Freeze these fields at the training cutoff. The source's repeated update every `n` rounds (five in visible code) has no role between independent benchmark groups. Rebuilding after an explicitly changed training history creates a new snapshot; scoring a group cannot trigger a state transition. This preserves a long-/short-term representation at one cutoff, **not an experimental test of dynamic updating**.

## 5. Explicit reconstruction of the memory module

Use **`memory_mode=yulan_retrieval_two_level_v1`**. The user selected an upstream-source-based reconstruction; the previous `history_summary_recent_v1` proposal is superseded and must not satisfy caches or resume checks for this version.

Pin the retrieved YuLan-Rec [memory source](https://github.com/RUC-GSAI/YuLan-Rec/blob/fff935570079502d5a17509b0baf590becd1eaab/agents/recagent_memory.py) to **`fff935570079502d5a17509b0baf590becd1eaab`**. Its [line-numbered primary snapshot](../../../docs/short_paper_selection/architecture_implementation_2026_10_25/sources/recagent/code-agents_recagent_memory.py.md) matches the freshly recovered source body. `RecAgentRetriever.get_relevant_documents` ranks memories by semantic salience, importance and recency; `RecAgentMemory._score_memory_importance` supplies the 1–10 importance judgment. Its `load_memory_variables` retrieves against the observation and returns the result as `most_recent_memories`: that field is not simply a latest-events list.

**Reuse those components, not the whole RecAgent memory lifecycle.** Retain two levels corresponding to DyTA4Rec's description. Do not add the upstream sensory stage, stochastic reinforcement/promotion, reflection or forgetting to this variant. Their omission is a declared reconstruction boundary, not evidence that DyTA4Rec lacked them. In particular, do not claim that setting `reflection_threshold=10` reproduces reflection when reflection is not executed.

Initialize once from permitted training events:

1. Retain source-event references with titles, genres, observed ratings, chronological positions and timestamps. Store no fabricated feelings or candidate exposures.
2. Short-term memory holds the recent **10** event records, supplied chronologically to the profile branch. Five is the long-term retrieval budget below, not a second recent-history truncation.
3. For long-term memory, summarize chronological blocks of **50** events, with a shorter last block. Merge consecutive groups of at most **10 summaries** recursively until one root remains. A single block is already the root. Retain every block and merge summary as a separately identified long-term record with source-event provenance, including the root. Use the root for the long-term profile extraction in Section 4; candidate scoring receives the selected records rather than the complete summary tree.
4. Score each summary record once using the upstream importance prompt and the same declared response backbone. Require an integer `a_j` in 1–10 and set `I_j=0.9*a_j/10`, following the wrapper's importance scale. Do not copy the sensory admission threshold or discard low-importance summaries. This application to hierarchical summaries is our initialization choice; the source does not prescribe it for DyTA4Rec.
5. Embed the summary prose with the existing frozen `sentence-transformers/all-mpnet-base-v2` provider; exclude IDs/provenance syntax from embedding text. Use the same provider for queries and preserve its resolved immutable revision, dimensions, float32 precision, normalization setting and encoder truncation policy in artifacts. The current local RecAgent provider uses 768 dimensions, no explicit normalization and a 384-token encoder limit. This embedding selection is not established as DyTA4Rec's missing `get_embedding_model` result. No automatic downloads are allowed.
6. Initialize each summary's creation/access time to the latest training-event timestamp in its support, not wall-clock generation time. The summary content is available only at the target's training cutoff. This timestamp convention for reconstructed records is our choice and controls their recency term.

For a group, query with the exact label-free candidate observation used by the profile-ranking stage, including its fixed candidate order and allowed item metadata. Use an isolated copy of the saved memory records and the cutoff + one-second reference clock. The upstream-derived retrieval recipe is made explicit through the existing local RecAgent port:

- Compute squared L2 distances in the declared embedding space. For the nearest `min(100, number_of_records)` records, use `v_j=1-d_j^2/sqrt(2)`; all other records receive semantic contribution zero. This preserves the visible FAISS relevance transform and the local port's 100-record semantic pool. Do not silently replace it with cosine similarity or clamp negative relevance values. The comment in the author source does not prove that relevance is bounded in [0,1].
- Score each record as `R_j=(1-0.01)^hours_since_access + I_j + v_j`, and select the highest **five**, or all if fewer. The local port makes the inherited time-weighted retrieval rule executable; these defaults remain upstream/local-port choices, not recovered DyTA4Rec constants. Stable insertion order resolves retrieval-score ties.
- The existing `RecAgentMemory.retrieve` and `importance` routines may be reused unchanged through a narrow adapter with an isolated `MemoryState`, or ported with equivalent fake-fixture verification. Do not call its `observe`, `read`, or `save` lifecycle, which would introduce sensory processing, reinforcement and reflection. Reuse the embedding/cache interface unchanged where compatible; do not change RecAgent defaults or existing callers.
- Access-time updates, if performed by that routine, affect only the disposable group copy. Return the selected summary texts/provenance to the profile-ranking prompt and discard the copy. The saved training snapshot, subsequent groups, temporal-branch prompts and initialization summaries remain unchanged. Query embeddings require no extra LLM generation calls.

The retrieved records supplement the fixed recent history and short-term profile. Record each selected ID, distance, semantic contribution, importance, recency term, final retrieval score and relevant model/provenance versions. Use token-aware preflight and explicit field limits; never silently drop summary blocks or selected records to fit a prompt. Validate evidence references against provided inputs, while acknowledging that schema validation cannot prove the factual quality of generated summaries.

The DyTA4Rec simulator creates a FAISS L2 store, an importance-aware custom retriever with `k=5`, and `AgentMemory(reflection_threshold=10)`. **Our selected routines come from acknowledged upstream code, not recovered DyTA4Rec classes.** Their use reduces arbitrary reinvention but does not establish identical initialization, storage, embedding, importance weighting, summary construction or retrieval behavior. It also does not implement the unavailable reflection mechanism. Keep this distinction in the method description and manifests.

The paper's formal recent-history tuple includes generated feelings. The runtime code instead appends item/rating strings to the temporal history and adds generated feelings separately to memory during simulation. Omitting synthetic feelings from logged-history initialization is deliberate and must be disclosed. Offline evaluation rules alone do not forbid generating feelings from training data; we choose not to manufacture unobserved emotional experiences.

If the complete DyTA4Rec dependencies become available before coding, inspect them and revise this declared reconstruction rather than silently changing its behavior or retaining an exact-reproduction claim. The currently chosen variant is implementable from the pinned upstream artifacts, existing local port and explicit initialization choices above.

## 6. One group: three rankings, five generation calls

Let `H` be the recent window and `C` the supplied candidate group. Assign stable local labels `C1...Cn`, backed by an exact ID-to-row map. All three ranking responses must be complete permutations of these labels, even if a branch will receive zero weight.

1. **Cluster analysis:** using `H` only, request concise observations about temporal groupings, shifts, and likely next genres/types, plus a typed `pattern_flag` in `{yes,no,other}`. This is prompted qualitative analysis, not a fitted clustering algorithm.
2. **Cluster ranking:** supply that analysis and `C` to a second call, returning a complete ranking. Preserve the source's two-stage analysis→ranking separation; do not combine it into one call.
3. **Sequential ranking:** begin with `H[:-k_eff]`; narrate each following history item as the next observed interaction, updating the narrated history in order. Then provide full `H`, explicitly identify its latest interaction, and rank `C` with a sequential `{yes,no,other}` flag. These examples are target-user training transitions, not retrieved neighbors or labeled evaluation slates. Handle `k_eff=0` explicitly.
4. **Profile-branch cluster analysis:** make a separate history-only clustering call, as author `_generate_reaction` does. Preserve this overlapping analysis rather than silently making the profile branch purely static. Its flag does not replace the flag from step 1 in adaptive fusion.
5. **Profile/memory ranking:** retrieve up to five long-term summary records as in Section 5, then supply fixed long-term profile/statistics/demographics, short-term summary/interests, `H`, retrieved and recent memories, fixed reference time, step 4's analysis, and `C`. Rank candidates by likely interaction. Profile plus recent context is an intentional source property, despite the author's `get_long_term_list` name.

Retain source intent, including the position-irrelevance instruction, but adapt output to strict JSON with candidate labels. Allow concise evidence-based observations; do not require unrestricted reasoning transcripts or repeat per-item explanations merely to parse rankings. Historical movies and candidate labels have separate namespaces. Reject omissions, duplicates, invented labels, incomplete JSON, and malformed flags. Map `yes` to high, `no/other` to low; missing/invalid flags are technical failures rather than silently low confidence.

The task instruction must say that interaction occurrence is being predicted and that an observed low rating still represents an interaction. Do not request exactly one selected item or the hidden positive count. Do not introduce the separate author `choose_movies_action` select-all path: it would be another response mechanism.

The source's profile branch can retain stale internal history while temporal branches receive a new external history. Use the same verified `H` in all branches; the static snapshot avoids that inconsistency. Keep the two analysis calls distinct in stage identities and caches, even at temperature zero. Reusing one as both would change the declared call graph.

## 7. Adaptive fusion and combined output contract

Use the author code's exact asymmetric rule. In branch order sequential/profile/cluster:

| Cluster flag | Sequence flag | Weights `(w_s,w_l,w_c)` |
| --- | --- | --- |
| high | high | `(1,1,1)` |
| high | low | `(0.5,1,1)` |
| low | high | `(1,1,0)` |
| low | low | `(1,1,0)` |

Do not change the last row to `(0,1,0)` or interpret flags as calibrated probabilities. These numeric rules come from released code, not fully specified manuscript equations. The paper's undefined inner `w_i` notation is not implemented as an invented learned coefficient.

Primary aggregation is **RRF**, with 1-based branch rank `r_b(i)`:

`score(i) = sum_b w_b / (60 + r_b(i))`.

Return those fused scores in original row order. Do not divide by the sum of weights, replace scores with final integer ranks, or break ties before evaluation. Use exact rational arithmetic for this small deterministic fusion, converting equal rational results identically to floating point for the common scorer API; floating summation order must not split mathematical ties.

Also allow declared **BC** as the other source aggregation variant:

`score_BC(i) = sum_b w_b * (n - r_b(i) + 2)`.

Here `n=|C|` explicitly replaces the released helper's fixed `recommend_len=10`, to accommodate variable-size groups. Retain its unusual `+2` convention: a ten-item list contributes 11 through 2, not 9 through 0. Record `bc_length_policy=candidate_count`. This is a size adaptation, not literal behavior of that default helper for non-ten-item groups. BC is not an additional LLM ensemble or an automatic model search; fusion can reuse saved branch outputs when all other settings match.

For either strategy, save native branch ranks, flags, weights, fused scores, and aggregate tie information. A deterministic display order may be recorded separately but must never replace scores in metrics. Source code returns a final permutation with insertion-order tie breaking; preserving ties for common metrics is an explicit evaluation adaptation.

Select `tau` on validation and save `prediction = (score >= tau)`, preserving missing outputs. NDCG@5/Recall@5 and classification must reference the same score column and the same chosen configuration. Classification requires **zero extra LLM calls**.

These scores express relative slate preferences, not calibrated interaction probabilities or absolute willingness. Unlike a simple fixed rank cutoff, raw adaptive sums also change scale with the flags/total active weight; BC additionally changes with candidate count. A global threshold can therefore respond to branch activation and group composition as well as rank agreement. Disclose this limitation and report active-weight/tie/group-size diagnostics. Do not silently normalize away this source behavior or claim threshold transfer across ratios.

## 8. Runner integration, failures, and isolation

Add isolated components:

| Location | Responsibility |
| --- | --- |
| `src/beyond_click_sim/scorers/dyta4rec/config.py`, `state.py` | Validated settings, immutable snapshots, schema/provenance versions |
| `profiles.py`, `memory.py` | Explicit statistics, bounded summaries, importance assignment and the isolated upstream-derived retrieval adapter |
| `prompts.py` | Source-mapped stage prompts, strict schemas and parsers |
| `fusion.py` | Exact weight table, RRF/BC, score/tie preservation |
| `scorer.py` | `DyTA4RecScorer.fit`, explicit `build_agents`, grouped `score`, diagnostics |
| `runners/in_distribution/interaction_prediction/methods/dyta4rec.py` | Temporal/metadata preparation, combined validation/test evaluation, wrappers |
| `tests/test_dyta4rec_*.py`, `docs/scorers/dyta4rec.md` | Fake-client verification, invocation examples, implementation status and disclosures |

Maintain the usual `fit(X,y) -> self` / `score(X) -> Series` interface. `fit` validates/prepares permitted evidence; explicit `build_agents` performs initialization calls for the union of requested validation/test users. Candidate scoring receives label-free copies and returns index-aligned numeric fused scores. Use existing compatible group-error and retry plumbing. Reuse pure helpers unchanged when appropriate, rather than importing another method's scientific profile logic.

Register opt-in Qwen3-8B and Qwen3.6-27B smoke/full methods, for example `dyta4rec_offline_rrf_litellm_qwen3_8b_full`, with `profile_version` and `memory_mode` mandatory in manifests/resume identity. BC configurations must have distinct identities. Do not add them to expensive default experiment rosters. Keep injectable fake clients and explicit model settings.

Preserve common artifacts and meanings:

- `predictions.parquet`: original row/user/item/group identity, split, evaluator-owned target, exact fused score, nullable thresholded prediction, and status. The evaluator appends targets after scoring.
- `metrics.json`, `metrics_ranking.json`: validation/test views, NDCG@5/Recall@5, group/user aggregation, threshold selection, fixed-denominator strict summaries, success-only diagnostics, and coverage.
- `manifest.json`, `preflight.json`, `llm_errors.jsonl`: task/cohort/configuration/source fingerprints, metadata/evidence budgets, model settings, costs, missing-state failures and coverage.
- DyTA-specific snapshots and `dyta4rec_trace.jsonl`: profile/memory provenance, summary/importance/embedding identities, selected memory IDs and retrieval terms, stage/attempt identities, branch permutations/flags/weights, analysis summaries, fusion strategy/exact scores, and calls/tokens. Preserve concise returned observations, not unrestricted hidden reasoning.

**Current shared-runner caveat:** `_listwise.evaluate_listwise_scorer` fits the threshold on successful validation rows and raises before artifacts when validation or test has no valid scores. Blind reuse therefore does not satisfy every manuscript failure rule.

Use a small DyTA-local orchestration adapter around unchanged metric primitives. Evaluate thresholds on the **same full requested validation groups**; failed groups contribute the all-negative fallback only to the strict classification objective. Consider valid score transition thresholds plus explicit all-positive/all-negative endpoints, with the existing deterministic threshold tie convention. For ranking, assign failed groups zero NDCG@5/Recall@5 directly, never a tied all-zero score vector. Save success-only diagnostics separately.

If validation has no valid groups, record `threshold=null` and classification unavailable; still save any test rankings, scores, coverage, and status. Do not invent a threshold or withhold all artifacts. A wholly failed test split also leaves artifacts. Distinguish score-generation coverage from classification-conversion coverage when threshold selection fails. With the intended zero-failure cohort, strict and success-only views coincide.

Use at most five attempts per initialization stage, checkpointing completed valid stages. Use at most five **whole-group** attempts for ranking, restarting its five-call pipeline from the same snapshot; do not nest five retries at every group stage. Stop an attempt at its first technical failure. No retries based on accuracy, surprising ranks, or valid low-pattern flags. If any required branch fails, the group fails, including when that branch would have zero weight; no partial ensemble fallback.

Expected edits are new DyTA files and additive exports/registrations. Preserve existing public methods, prompts, defaults, artifacts, splits, candidates, metrics, and other scorers' behavior. **Behavioral isolation is an acceptance criterion against the current workspace.** Record starting status and respect concurrent Agent4Rec work; do not reset unrelated edits or use this feature to refactor common evaluation semantics.

## 9. Models, caching, and bounded cost

All generation stages within a configuration use the same declared frozen response backbone. Qwen wrappers use temperature 0, an explicitly recorded top-p (proposed 1), and disabled hidden thinking where supported. Keep a configurable GPT-4o-mini reference setting with source temperature 0.1/top-p 0.9, without launching it or claiming that it recovers the unknown source model snapshot.

Proposed maximum response tokens: memory block/merge and long-term profile 1,024 each; importance 16; each recent summary 256; each clustering analysis 1,024; each ranking 2,048. These are engineering defaults, not source budgets. Record actual model/serving and embedding revisions, effective provider options, context limit, seed support and usage. Deterministically check context fit; any truncation policy must be explicit/versioned rather than silently dropping evidence or candidates. Record embedding encoder truncation separately from the full text shown in LLM prompts.

If `B=ceil(N_train_user/50)` and `R` is the number of merge calls needed to reduce those block summaries to one, uncached successful initialization costs **`2*(B + R) + 3` LLM calls/user**: one summary and one importance call for each block/merge record, one long-term profile, and two recent summaries. `R=0` when `B=1`. Also embed `B+R` summary texts. After initialization, each successful group costs **five LLM calls** plus a query embedding; retrieval adds no generation call. One validation and one test group therefore use ten successful group-generation calls, with no extra classification calls. Retries increase these totals; report actual calls/tokens and embedding/cache activity separately. These are accounting formulas for our reconstruction, not measured source costs.

Provide no-network preflight reporting evidence coverage, intended group counts, required missing snapshots, conservative prompt bounds and uncached call counts. No experiment is authorized by this implementation task; verification uses fake clients only.

Use atomic versioned JSON/Parquet caches rather than unsafe object serialization. Keys must include effective ordered training evidence/ratings, metadata and population-statistic hashes, cutoff, history/ICL settings, reconstruction versions, exact rendered inputs, stage identity, response model/serving configuration, decoding options and seed. Memory/retrieval keys additionally include pinned upstream/local-port identities, importance values, summary support/access times, exact embedding text and resolved provider metadata, semantic-pool/retrieval limits, scoring constants and reference clock. Group keys include ordered candidate identities/metadata, the rendered retrieval query and snapshot identity. Threshold/evaluation caches include validation identities/labels separately from inference caches; changing a label cannot alter inference cache identity or content.

Keep the two clustering-analysis stages distinct. Reuse completed initialization or complete group outputs on resume; do not let a partially completed failed attempt become a successful ensemble. Preserve independent seed/configuration identities. Profiles may be reused across candidate ratios only when all their actual inputs/settings are identical. Pure fusion may be recomputed from saved valid branches without new model calls.

Evaluation order, concurrency, and resumption must not mutate snapshots or deterministic prompts. Live LLM responses need not be bitwise repeatable at temperature zero; tests use deterministic fake clients and documentation must retain that practical limit.

## 10. Implementation sequence and acceptance criteria

1. Recheck current workspace/contracts and pin available primary-source hashes. Implement typed configuration, metadata allowlists, verified chronological evidence, statistics, and no-network preflight.
2. Implement versioned profile/memory initialization, importance/embedding preparation and stage checkpoints; verify the narrow upstream-derived retrieval adapter and freeze snapshots before candidate inference.
3. Implement all five group stages, robust label/flag parsing, and the explicit ICL construction.
4. Implement exact adaptive fusion and combined-output orchestration without changing shared evaluator defaults.
5. Add opt-in registrations, artifacts/documentation, and fake-client tests. Run focused tests and relevant existing-method regressions; report pre-existing failures separately.

Required tests:

- **Boundaries:** with fixed materialized training/catalog/candidates, changing test labels or hidden outcomes cannot change snapshots, prompts, branch scores or the validation threshold. Changing validation labels may change only threshold/selection metrics, not generated scores. Candidate-relevant metadata changes invalidate caches. Training-only population statistics do not shrink when fewer users are scored.
- **Temporal/evidence selection:** exact latest-10 selection, stable timestamp ties and repeated events; correct 3/short/zero-shot ICL; separate full-history long-term evidence. Discarded older events may affect long-term state but cannot enter the recent temporal window. Reject incompatible random/noninteraction tasks.
- **Profile/memory:** hand-computed statistics and missing-category cases; block boundaries/reduction, valid provenance, demographics category mapping, no invented feelings; all required memory/profile fields reach the correct branch. Candidate and label changes do not rebuild the user state; candidate metadata may change group-specific retrieval as intended.
- **Upstream-derived retrieval:** compare toy records against the current local RecAgent `importance`/`retrieve` routines using fake embeddings and clients. Verify 1–10 validation and `.9` scaling, squared-L2 conversion including negative relevance, the 100-record semantic pool, `.99^hours` recency, stable ties, top-five and fewer-than-five cases. Test query/record encoder provenance and changed embedding revisions. An adversarial fixture must show that semantic/importance retrieval can choose an older record over a newer one. Access-time updates stay in disposable copies, and no sensory/reinforcement/reflection/forgetting routine is invoked.
- **Call graph:** two distinct analysis calls, one sequence ranking, one cluster ranking, one profile ranking; no clustering-analysis reuse, zero-weight shortcut or bootstrap rounds. Verify `2*(B+R)+3` initialization LLM calls, summary/query embedding activity and ten group-generation calls for one validation plus one test group on the first successful attempt.
- **Fusion:** all four flag combinations, especially both-low `(1,1,0)`; hand-calculated RRF/BC, actual candidate counts, disagreeing branch permutations and mathematical ties. Invalid/missing flags fail rather than becoming low. Invalid permutations cause whole-group failure.
- **Combined outputs:** saved scores equal mapped fusion; saved labels threshold those exact scores. Metrics consume the unmodified aggregate scores with ties. Cover different group sizes, all-positive/all-negative threshold endpoints, identical-score groups and flags that change score scale. No extra classification call.
- **Failure handling:** failed initialization, partial attempt, exhausted group, wholly failed validation and wholly failed test; raw missing values, full denominators, unavailable threshold and persistent artifacts. A valid tied ranking and an unsuccessful group must have different metric treatment.
- **Cache/state:** stage/model/prompt/evidence changes invalidate the correct entries; immutable snapshots survive reordered groups, concurrency, retries and resumed execution. Failed partial attempts do not satisfy a completed-group cache entry.
- **Isolation:** compare representative current direct-history, LLMRank, LLMSRec-Syn, RecAgent, Agent4Rec and conventional fixtures. Run affected exports/registry, timestamp-training, threshold, scorer/runner and ranking-metric checks. Existing method prompts, scores, output schemas and default rosters remain unchanged.

Use fake clients only for verification. No package installations, model downloads, cluster launches or paid calls are needed. Tests establish implementation behavior, not predictive quality or scientific reproduction.

## 11. Required divergence disclosures and later manuscript update

| Difference / uncertainty | Planned treatment | Why it matters |
| --- | --- | --- |
| Full interactive simulator with changing memories/profiles | One frozen training snapshot for independent groups | Protocol adaptation; cannot assess feedback-driven dynamics, fatigue, exit, policy effects or temporal recovery |
| Last-item holdout, one positive plus nine negatives | Shared fractional temporal split and multi-positive groups/ratios | Different prediction problem and candidate distribution; no source-table comparability |
| Long-term initialization and statistical recipes unavailable | Explicit full-training summaries, raw descriptive statistics and profile schema | Reconstruction; not recovered author traits or prompts |
| Custom semantic/importance memory unavailable | Pinned YuLan-Rec/local-port retrieval and importance over reconstructed hierarchical summaries | Upstream-derived substitution; no evidence of identical DyTA4Rec classes, initialization or settings |
| YuLan-Rec's full three-stage memory lifecycle | Reuse retrieval/importance only, retaining two-level history memory | Explicit component selection; no sensory compression, reinforcement, reflection or forgetting claim |
| Missing DyTA4Rec embedding provider and record times | Frozen MPNet with recorded encoder behavior; summary support timestamps and fixed group clock | Reconstruction choices affect semantic/recent evidence and must be reported |
| Generated feelings and online reflection | No fictional retrospective feelings or post-response updates | Emotional state/transition mechanisms are not tested; omitted initialization choices are deliberate, not protocol necessities |
| Paper best ICL 3, released default 6, main-table default unclear | Recent history 10, ICL 3, both configurable | Source conflict is disclosed; do not label these exact Table 1 settings |
| Item/profile formatting and `status` meaning unavailable | Explicit canonical demographics/genres, train-only statistics and documented occupation mapping | Metadata reconstruction and additional information relative to methods that omit it |
| Temporal proximity wording, no explicit runtime timestamp features | Chronological event order, deterministic cutoff clock | No claim to model real time gaps or calendar effects; wall-clock substitution is explicit |
| Author final-permutation tie breaking | Preserve raw fused ties for common average-tie metrics | Changed output/evaluation convention |
| Fixed BC helper length 10 | Candidate-count length, same source offset | Group-size adaptation; score scale and threshold semantics must be recorded |
| No rank-derived classification in reported paper experiment | Validation F1 threshold on fused ranking scores | Additional evaluation; scores/labels do not establish absolute willingness |
| Missing/permissive title parsing and undefined failure metrics | Full label permutations, typed flags, bounded retries and strict failure metrics | Engineering/protocol adaptation; may change which outputs count as valid |
| GPT-4o-mini, temperature .1/top-p .9 | Configurable Qwen defaults, temperature 0/top-p 1 | Backbone/decoding substitution, not required by offline evaluation |
| Public code revision resolved, paper-run revision and model snapshot unknown | Pin public head/source hashes and actual run configuration | Public source provenance is recoverable; exact author-run reproduction is still unestablished |

The manuscript now describes this implemented adaptation, citing Wanyan et al. (2025), DOI `10.1145/3746252.3760878`, under bibliography key `wanyan2025dyta4rec`. State the three branches, adaptive RRF and source-derived weights, fixed training snapshot, profile/history reconstruction, pinned upstream retrieval/importance provenance, omitted upstream lifecycle components, history/ICL defaults, model/embedding substitutions, and combined output contract. Cite the inspected DyTA4Rec revision and missing-code issue without claiming the files are absent from every possible private source. Expose `yulan_retrieval_two_level_v1` in reproducibility records. Compile the existing manuscript after editing, and change planned-status wording only after implementation is verified; keep empirical results missing until experiments exist.

The comparison evaluates configured methods. Different profile construction, full-history access, demographics, metadata and call budgets are not errors to hide, but they prevent claiming that performance differences isolate temporal architecture alone. The plan's central remaining fidelity limitation is unavailable author initialization/memory code; the proposed defaults make an honest adaptation implementable without pretending that limitation has been solved.
