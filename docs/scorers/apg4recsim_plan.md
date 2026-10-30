# APG4RecSim implementation plan

Status: proposed implementation, 29 October 2026. Planning only; no APG4RecSim code, experiments, or manuscript changes are established by this document.

Implement APG4RecSim's three-stage profile generation and profile-conditioned **ranking** simulator for the shared temporal MovieLens-1M interaction task. Save ranking scores and validation-thresholded binary labels from the same responses. The user explicitly selected the manuscript's **actual counterfactual perturb-and-rerun mechanism**, rather than the released code's verbal counterfactual self-assessment.

## Sources and intended claim

- Wanyan et al., *Task-Aware Automated User Profile Generation for Recommendation Simulation Using Large Language Models*, [arXiv:2605.13497v1](https://arxiv.org/abs/2605.13497v1), especially §§3.2–3.4 and 4.1. This is the supplied 11-page PDF, SHA256 `d7f1ee24f475be82970c4f715fdbf131766e6f284b8c64c3c6d27331ec0be6f6`; equivalence to the SIGIR final is not established.
- The supplied [implementation dossier](../../../docs/short_paper_selection/architecture_implementation_2026_10_25/reports/apg4recsim.md), checked against the manuscript and primary source files.
- [Author repository](https://github.com/WennyXY/APG4Sim/tree/fe32688d601b7c942da0aea698511bd88a0a25f7), pinned commit `fe32688d601b7c942da0aea698511bd88a0a25f7`; [local source provenance](../../../docs/paper_readings/interaction_protocol_2026_10_28/sources/apg4recsim26/provenance.json). This newer snapshot resolves the dossier's unavailable-commit limitation, but its correspondence to the reported experiments remains unverified.
- The [current manuscript](../../../paper_beyond_click_sim/www_short/main.tex), particularly Splitting and Information Boundaries, Shared Candidate Groups, Two Views of Native Predictions, and Additional Simulator Coverage. [Independent source reading and review](../../../docs/paper_readings/apg4recsim_plan_2026_10_29/index.md).

After implementation and verification, the defensible method description is:

> We reimplemented APG4RecSim's three-stage profile-generation architecture and profile-conditioned ranking simulator for our interaction-prediction protocol. We instantiated the paper's counterfactual trait-to-step mapping with explicit simulator reruns and documented choices for otherwise unspecified probes, perturbations, stability checks, and decision-step outputs. The same rankings supply ranking metrics and validation-thresholded binary predictions.

Do **not** claim an exact reproduction, equivalence to the released executable pipeline, or that ranking-to-classification is the only change. The paper omits necessary algorithmic details, the release differs on a central mechanism, and the shared experiment changes the data and model settings.

## Scope and shared evaluation contract

1. Support only the temporal MovieLens-1M **interaction** task initially. Every recorded rating is an interaction positive, including ratings 1–2. Historical rating values remain evidence for profiling; they do not redefine the target as liking or rating >=4. Unobserved sampled items are not verified rejections.
2. Consume the existing per-user chronological 0.7/0.1/0.2 split. Keep training evidence and profiles fixed for validation and test; never append validation events. Verify timestamps, stable event-order ties, and the materialized training projection. Reject legacy randomized or unsupported tasks before constructing an LLM client.
3. Consume the common frozen users, candidate identities, and presentation order. Target 1,000 eligible users, at most five groups per user, cap 20 candidates, ratios 1:1, 1:3, 1:9, 1:19, and seeds 0, 1, 2. Do not reconstruct candidates, filter them by generated preferences, reveal the positive count, or rerank the presentation before inference.
4. Use NDCG@5 as primary ranking metric and Recall@5 as companion, averaging groups within users and then users. Select the binary threshold independently for each dataset/ratio/seed using validation macro-by-user/group F1; apply it unchanged to test scores. No additional classification inference or classifier is needed.
5. Cohort eligibility is checked before model calls, including required metadata and enough permitted catalog items for initialization probes. Reuse a common cohort across methods. Do not replace users based on model responses, profile influence, accuracy, or failures. Zero failed groups remains the intended experiment condition; preserve retries and coverage accounting for unexpected failures.
6. Keep Qwen3-8B and Qwen3.6-27B available through the existing configurable clients and smoke/full registration pattern. Backbone search/selection orchestration, equalizing history budgets across methods, new datasets, other targets, and experimental results are outside this implementation scope.

Per-user chronology does not give a global historical cutoff. Population training statistics may include other users' events later than a target's cutoff. This is the shared static-offline protocol's limitation, not something APG4RecSim can eliminate without changing the common information policy. APG4RecSim adds no neighbor retrieval.

## Evidence and metadata

Use the most recent **15 permitted training interactions**, or all available if fewer, with deterministic timestamp/event-ID ordering. Fifteen is the released initialization CLI default, **not a recovered main-paper setting**; the paper reports only a fixed recent-history window and a 5/10/15/20/25 sensitivity study. Record the actual window and event IDs. Equal-history comparisons remain deferred.

Preserve the source MovieLens formatter's fields in both profile history and candidates, with the following placement and restrictions:

| Field | Profile generation | Probe and evaluation candidates | Provenance/adaptation |
| --- | --- | --- | --- |
| Title | Yes | Yes | Canonical title; local candidate labels disambiguate duplicate titles |
| Genres | Yes | Yes | Canonical item genres; APG4RecSim does not require Agent4Rec's generated-genre mechanism |
| Description | Yes | Yes | Frozen, provenance-checked movie description/summary; reuse the project's title-derived summaries and disclose that their provenance need not match the source's unspecified enrichment |
| Average rating | Yes | Yes | Mean from the full permitted population training partition, before history truncation or evaluation-user selection |
| Popularity | Yes | Yes | Permitted population training interaction count, explicitly labeled as a training count |
| User rating | Exact 1–5 history bucket | Never | Includes low-rated interactions; no candidate-specific held-out ratings |
| Timestamp | Selection/audit only | Never | Retains the released MovieLens rating-bucket formatting, which does not print dates; do not claim the prompt provides temporal trajectories |
| Age, gender, occupation, location | No | No | Paper §4.1.3 explicitly excludes demographic inputs |

Keep the selected events in chronological order within rating buckets. An empty bucket is explicitly empty. Missing titles/genres/descriptions are data-preparation prerequisites for the primary configuration, not reasons for a silent reduced-metadata fallback. No-training-observation items have count zero and an unavailable mean rating, never a fabricated zero rating or a mean filled from held-out data.

Allowlist predictor inputs. Keep targets, sampled/positive flags, candidate held-out ratings, split membership, and true positive counts outside all LLM prompts. User/item IDs are cache/join keys; expose local candidate and history-reference labels rather than raw dataset IDs. Raw extraction evidence references remain auditable without exposing evaluation labels.

The task builder may use all observed membership to avoid sampling known positives for evaluation, as permitted by the common protocol. **Profile initialization must not reuse that held-out-aware sampler.** Its probes and examples use only permitted training evidence and catalog metadata. Changing held-out events must not change profile initialization when the training projection and catalog are held fixed.

## Stage 1: three independent attribute extractions

Follow paper §3.2 and the active `initialisation_prompt2` template:

- Make three independent completions from the same selected, rating-bucketed history. Do not bootstrap three different histories or select just one of the generated profiles.
- Ask for granular natural-language behavioral traits, brief evidence summaries, and explicit uncertainty where support is weak. Do not import Agent4Rec's numerical activity, conformity, or diversity calculations.
- Preserve all three outputs and their union, including paraphrases for Stage 2 to consolidate. Use stable source-trait IDs and record which extraction produced each trait.
- Use temperature 0.1 for extraction and consolidation. Give each extraction a distinct sample identifier and seed when supported; cache keys must not collapse three identical prompts into one completion. Independence means separate sampling calls, not guaranteed different text.
- Adapt the source's free-text blocks to a validated JSON list of atomic statements, evidence references, and qualitative confidence. This is an output-schema adaptation, not a new fitted trait estimator. Empty/invalid responses retry; do not silently treat transport failure as an empty valid profile.

## Stage 2: context-aware consolidation

Implement the missing orchestration around paper §3.3: merge the overcomplete union using the schema/admissible fields, the recommendation-response task family, and one frozen illustrative input/output example. Deduplicate semantically overlapping traits, preserve materially different preferences and conditions, retain uncertainty, and trace every consolidated trait to its inputs. Do not select traits using validation/test accuracy.

Use a **versioned synthetic MovieLens-like example with fictional items** shared across users. It illustrates field meanings and a complete ranking, not a real user's held-out answer. Its profile and expected order are explicitly illustrative. This replaces the release's per-user observed/unobserved exemplar and avoids its full-history negative-exclusion dependency. The exact example is an engineering choice, not a recovered source prompt.

The source `merge_prompt` already demands task-specific filtering, while the manuscript calls Stage 2 domain-grounded/task-agnostic and reserves decision-step mapping for Stage 3. Follow the manuscript: consolidate and ground here; do not preemptively remove a supported trait solely because an LLM verbally predicts it will not affect ranking. Record the recommendation-family context, so “task-agnostic” does not misleadingly mean no task context at all.

Emit stable IDs and atomic affirmative statements suitable for a single-statement negation. Do not silently impose a 10- or 12-trait cap copied from unused alternative source prompts. Check context size and report truncation/failure explicitly rather than dropping traits to reduce counterfactual cost.

## Stage 3: executable counterfactual mapping

The manuscript specifies the perturb-and-rerun principle but does not specify the probe count, exact path, node output instrumentation, stochastic controls, or pruning policy. The defaults below are **proposed reconstruction choices**, frozen before evaluating held-out outcomes. They implement the user's paper-mechanism choice; they are not claimed as author settings.

### Fixed expert decision path and observable outputs

Use the expert-template branch explicitly allowed by §3.4.1. Freeze three ordered nodes: (1) candidate suitability under expressed constraints/aversions, (2) comparative preference and behavioral trade-offs, (3) overall predicted interaction order. A low rating/aversion may influence an interaction ranking but cannot make a logged training interaction cease to be an interaction.

For an observable ranking-valued decision at every node, request a complete candidate permutation for each node in one structured response. The final node is the native ranking output. This is explicit instrumentation of the otherwise unspecified “output at that step,” not proof of access to the model's internal cognition. Do not ask for private reasoning traces or add memory/planning tools. All candidates remain present, even at the suitability node; there is no hard candidate removal.

Use the same simulator prompt, node schema, model, decoding settings, and candidate formatter for initialization probes and final evaluation. Intermediate orders are audit outputs, not additional predictions to ensemble or independent LLM calls.

### Frozen probes and perturbations

- Build two fixed, independently seeded 10-item probes per user before validation/test scoring. Each contains five distinct items sampled from that user's selected history and five distinct catalog items absent from the user's full permitted training history. Items can overlap across probes but cannot repeat within a probe. The current minimum-history protocol supplies at least five training events; check unique-item sufficiency explicitly.
- Select catalog items using only metadata eligibility and permitted training membership. An unknown item is **not labeled negative**, and may coincidentally appear in held-out data; do not inspect held-out membership to prevent that coincidence. Freeze candidate order. Probes are sensitivity tests, not labeled training examples or evaluation groups.
- Hide known/unknown membership and historical candidate ratings from the simulator. No correct answer or target is needed to measure a change between two outputs.
- For each consolidated atomic trait, create one deterministic negation, `It is not the case that: <original statement>`. Change exactly that trait's runtime statement, preserving all other runtime fields and candidate bytes. Do not send contradictory source-evidence explanations with the counterfactual; evidence is provenance, not part of the simulator's trait text.
- Probe each trait against the same full consolidated baseline profile, before any mapping or deprioritization. Do not sequentially remove traits and thereby change later traits' baselines. Neither probes nor mapping use the binary threshold.

### Stability, influence, and refinement

Use temperature **0 for the ranking simulator**, both in probes and evaluation, to reduce stochastic ambiguity; this departs from the paper's 0.1 response temperature. Retain temperature 0.1 for the three extraction samples. Pin seeds where supported, record server determinism limitations, and do not assume temperature zero guarantees identical responses.

For each probe, make two genuinely separate baseline calls. For each altered trait, make two genuinely separate calls with that one intervention. At a node:

- A baseline is stable if its two parsed permutations agree. An intervention is stable if its two parsed permutations agree.
- Count influence only when both are stable and their orders differ. Any order change counts, matching the manuscript's ranking criterion; there is no correctness or NDCG test here.
- A stable identical result is “no detected change on this probe.” Unstable or failed comparisons are “inconclusive,” never evidence of irrelevance. Stable differences are evidence of model-output sensitivity, not real-world causal identification.

Attach a trait to every node with a verified change on at least one probe, retaining the exact tested trait wording. Keep traits with stable no-change results on all probes as lower-priority background; the paper permits **deprioritizing** rather than deleting them. Preserve inconclusive traits with an explicit unverified status and no claimed node influence. Do not substitute a verbal relevance score for the required reruns, and do not rewrite influential statements after testing without rerunning the affected tests.

Store all baseline/intervention orders, trait replacements, stable/inconclusive flags, and node mappings. Report final-node influence separately from earlier-node changes. If no trait changes an output, retain a valid profile with no verified mappings; do not regenerate until a preferred result appears or exclude that user. Finite probes cannot prove that a trait is universally irrelevant.

Freeze the resulting task profile for all validation/test groups in that run. The final mapped placement and lower-priority background are the Stage-3 refinement; no extra free-form refiner may silently change the tested content.

## Native ranking and combined outputs

For each evaluation group, perform one structured role-play call using only the frozen ranking profile, fixed path, task definition, and candidate metadata. Do not reintroduce raw history as a parallel response predictor. Require each node to return every local candidate label exactly once. Reject omissions, duplicates, unknown labels, truncated JSON, and partial node outputs; never repair an invalid ranking using reference labels.

For the final complete order, with one-based rank `r` and group size `n`, save `score = n - r`. Map back to the original row identities and use these scores directly for NDCG@5 and Recall@5. A valid native output is a strict permutation; it has no within-group score ties. Preserve the common average-tie metric policy without creating an unnecessary ranking ensemble.

Use the project's standard DCG discount `1/log2(rank+1)`. The released `utils.DCG` instead gives ranks 1 and 2 equal weight; the missing evaluation driver prevents establishing whether that helper produced the paper's tables. Do not copy that helper or promise numerical comparability with the source's reported NDCG merely because the metric name matches.

Select `tau` on validation and save `prediction = (score >= tau)` alongside **those exact scores**, native ranks, and status. Rank thresholding measures relative ordering: equal-size complete groups select a fixed number of items at a fixed threshold. It is not an estimate of absolute interaction willingness, exposure-conditional click probability, or natural acceptance prevalence; smaller final groups also have different score ranges.

Keep the standard `predictions.parquet`, `manifest.json`, pointwise/ranking metric filenames, and coverage/error artifacts. Add APG-specific profile and probe artifacts without changing existing columns or other methods' meanings. No second ranking class, discrimination call, calibration model, or rerun for classification is required.

APG4RecSim's native discrimination branch would require a separately constructed task profile. It is outside scope. Comparing such a future branch against this ranking branch would change profile construction as well as output interface, so it would not be the manuscript's fixed-profile interface diagnostic.

## Integration, failure behavior, and isolation

Proposed additive implementation layout:

| Location | Responsibility |
| --- | --- |
| `src/beyond_click_sim/scorers/apg4recsim/profiles.py` | Typed traits, provenance, consolidation, immutable task profiles, staged initialization |
| `src/beyond_click_sim/scorers/apg4recsim/prompts.py` | Versioned extraction/consolidation prompts, synthetic example, expert path, simulator schema |
| `src/beyond_click_sim/scorers/apg4recsim/counterfactual.py` | Train-only probes, single-trait replacement, repeated comparisons and mapping |
| `src/beyond_click_sim/scorers/apg4recsim/scorer.py` | `APG4RecSimScorer`, grouped native ranking, row-aligned scores, trace collection |
| `runners/in_distribution/interaction_prediction/methods/apg4recsim.py` | Temporal-task checks, metadata/profile preparation, combined evaluation, Qwen smoke/full wrappers |
| `tests/test_apg4recsim_*.py`, `docs/scorers/apg4recsim.md` | Fake-client verification, implemented status, examples, artifact contract and limitations |

Use the standard scorer `fit`/`score` interface. `fit` prepares permitted training evidence; an explicit staged profile-build step creates profiles for the frozen cohort. `score` consumes label-free grouped candidate rows and never mutates fitted profiles. Keep initialization outcomes separate from group-response failures. Reuse timestamp recovery from `_llmsrec_syn_data.prepare_training` only after verifying the entire training projection; do not rewrite shared splitters or candidates. Reuse pure item-statistic, summary-provenance, parser, and metric helpers where their contracts match.

Use existing five-attempt retry semantics for invalid/failed stage operations and group responses. Resume initialization from successful immutable stage artifacts; a group retry restarts that group from its frozen profile, without running profile extraction again. An unavailable profile leaves that user's groups missing; failures must not become valid zero scores or synthetic rankings.

**Current runner caveat:** `_listwise.evaluate_listwise_scorer` selects thresholds on successful validation rows and raises before writing combined outputs if a split has no valid scores. It also exposes successful-only and failure-inclusive summaries separately. Do not describe unmodified reuse as satisfying every manuscript failure rule.

Use an APG-local orchestration adapter around existing metric primitives, without changing existing runners' defaults: retain raw missing scores/labels, report failed groups as zero utility for the fixed-denominator ranking summary, and use all-negative predictions for failed groups only in the strict classification summary and validation-threshold objective. Preserve successful-only diagnostics and coverage separately. Evaluate threshold candidates from valid scores, with explicit all-positive/all-negative endpoints and the existing deterministic tie convention, over the **same full validation groups**. With no successful validation group, record threshold unavailable and do not fabricate test classification labels; save coverage/status and any available ranking outputs. A wholly failed test split must still leave artifacts. This is integration with existing failure accounting, not a new failure-recovery subsystem. In the intended zero-failure run the strict and successful-only calculations coincide.

Only additive exports, registrations, and APG-specific code/configuration are expected. No changes to other methods' prompts, histories, profiles, thresholds, default method rosters, candidate sets, shared splits, or metrics. Behavioral isolation is an **acceptance criterion checked against the current workspace**, not an assumption. Preserve concurrent work in other scorers; do not reset unrelated modifications.

## Reproducibility and inference budget

Checkpoint the three raw profiles, consolidated traits, frozen probes/path, every comparison, and final mapped profile. Fingerprints include the permitted training projection and selected events, catalog/description/statistic hashes, model and serving revision, decoding settings, all prompt/schema/path versions, perturbation policy, probe seed/count, repeat identity, and task definition. Never key profiles solely by user ID. Scope profiles to evaluation seed/configuration; they may be reused across ratios only when all profile inputs/configuration are identical. Candidate/threshold artifacts remain ratio-specific.

Independent extraction and stability-repeat identities must remain distinct cache entries. Completing one baseline call cannot satisfy both stability repeats. Reuse a completed individual call on resume, but never reuse an unperturbed response for an intervention. Store seeds and whether the serving client actually honors them. No cache can make three independently sampled profiles deterministically distinct.

Proposed response budgets are 4,096 tokens for extraction/consolidation and 1,024 for the three-node ranking JSON, with thinking disabled on configured Qwen clients. These are engineering limits, not paper settings. Verify context fit before calls; preserve all required metadata and audit any explicit versioned truncation policy rather than silently shortening descriptions or traits. The same model is used throughout a primary configuration unless a separately named mixed-model experiment is introduced later.

For `T` consolidated traits, two probes and two repeats, the nominal initialization cost is **4 + 4 + 4T = 8 + 4T calls per user**: three extractions, one consolidation, four baselines, and four intervention calls per trait. Expert path, synthetic exemplar, deterministic negation, and final attachment need no LLM calls. Add one successful response call per validation/test group. Retries increase this total; pre-existing item-description preparation is separate. Record actual calls, tokens, failures, and cache hits. This is an accounting formula for the proposed implementation, not a cost reported in the source paper. No inference is authorized by this planning task.

## Verification and acceptance

Use fake clients only during implementation verification; do not launch experiments or paid calls.

1. **Evidence boundaries:** changing held-out ratings, targets, sampled flags, or held-out membership leaves selected histories, initialization prompts, probes, profile artifacts, and native predictions unchanged when training/catalog/candidates are fixed. Changing test labels cannot change the validation threshold. Changing validation labels may change the threshold, never profile generation or ranking scores.
2. **Three-stage architecture:** three separate samples feed one union/consolidation; schema/example grounding is present; runtime uses the frozen task profile. Independent samples/repeats survive caching and resume. No numerical Agent4Rec trait recipe or memory loop is imported.
3. **Real counterfactuals:** fake outputs demonstrate influence at one node, several nodes, no nodes, and unstable nodes. Assert exactly one trait statement changes, all candidate bytes stay fixed, stable repeats are actual calls, and every intervention uses the same unmodified baseline. Confirm no verbal-mapping fallback and no pruning after inconclusive comparisons.
4. **Profile lifecycle:** scoring groups in different orders leaves profile hashes and prompts unchanged; profiles are not regenerated by group retries; changed training evidence, models, metadata, paths, prompts, or probe policy invalidates relevant cached state.
5. **Metadata/temporal behavior:** most recent 15 with ties resolved deterministically; rating-bucket fields and all source item fields included; means/counts train-only; missing means distinct from zero; demographics and candidate ratings excluded. Reject unsupported/randomized tasks and incomplete required metadata before inference.
6. **Outputs and call accounting:** with a prebuilt profile, one validation and one test group require two successful calls total, each returning all three node orders. Assert native row mapping, `n-r` scores, thresholded labels from those scores, NDCG@5/Recall@5, unequal group sizes, strict parser behavior, and the `8+4T` successful initialization-call budget on stable fake data.
7. **Failures:** failed initialization/group calls retain missing raw outputs, fixed denominators and coverage; no-success validation records unavailable threshold; no-success test still writes outputs. Do not add post-response cohort substitution.
8. **Behavioral isolation:** run focused APG tests and relevant existing LLMRank, LLMSRec-Syn, RecAgent, Agent4Rec, listwise-runner, threshold, and ranking-metric regressions against the current workspace. Compare existing fake-client prompts/predictions/artifact contracts where touched. Record pre-existing failures separately. A shared helper change is acceptable only if its old defaults and callers remain behaviorally unchanged and are tested.

## Departures to disclose in documentation and the eventual manuscript

| Source method or uncertainty | Proposed implementation | Classification |
| --- | --- | --- |
| Three independent extractions and overcomplete trait union (§3.2) | Preserve with distinct sampling/cache identities | Core mechanism retained |
| Context-grounded consolidation (§3.3); missing executable driver; source prompt prematurely task-specific | Implement domain consolidation before influence mapping | Reconstruction, manuscript prioritized |
| Actual trait perturbation and agent rerun (§3.4.2) vs one verbal refinement call in released `causal_merge` | Actual reruns, as chosen by the user | Paper mechanism retained; deliberate difference from release |
| Expert or generated task path; exact usable path absent | Fixed three-node expert path with ranking-valued node outputs | Permitted branch plus explicit instrumentation choice |
| Unspecified probe/repeat/perturbation/deprioritization policies | Two train-only probes, one negation per trait, duplicate-call stability checks, lower-priority no-change traits | Reconstructed algorithmic details, not exact source settings |
| Per-user training-derived exemplar; released sampling excludes full observed history | Frozen synthetic schema/ranking example; initialization probes never inspect held-out membership | Example and leakage-boundary adaptation |
| Earliest 80%/last 20%, most recent window with main length unspecified | Shared 70/10/20; latest 15 from training, release default | Protocol adaptation and declared default |
| Source item descriptions and aggregate provenance unspecified | Project summaries plus training-only means/counts; all source MovieLens fields retained | Metadata provenance/statistic adaptation |
| Main GPT-4o-mini, temperature 0.1 | Configurable Qwen backbones; 0.1 extraction/merge, 0 response; explicit budgets | Backbone/decoding adaptation |
| Single-positive ranking; fixed-count discrimination; rating also evaluated | Shared potentially multi-positive ranking groups, cap20 and four ratios | Task/candidate-distribution adaptation; other native branches omitted |
| nDCG@5/@10 and HR@3, five repetitions | NDCG@5, Recall@5, shared aggregation, three seeds | Evaluation adaptation |
| Released DCG helper has equal weights at ranks 1 and 2; its use in paper results is unverified | Standard shared `1/log2(rank+1)` discount | Preserve common metric definition; disclose source-code discrepancy |
| Native ranking output | Rank-to-binary validation F1 threshold on the same scores | Additional evaluation view; not absolute willingness |
| Free-text profiles; final ranking prompt/parser absent | Structured traits, observable node permutations, validated labels, resumable caches/retries | Engineering reconstruction with disclosed prompt/interface effects |

APG4RecSim adds no fine-tuning, embeddings, retrieved demonstrations, numerical trait-weight training, persistent behavioral memory, action loop, recommender policy, or real-time learning. Its generated profiles and observed sensitivity are simulator mechanisms, not validated psychological traits. Different history lengths, metadata, profiling procedures, and initialization costs across methods remain part of the configured-method comparison; results cannot isolate architecture alone.

The original paper's RecAgent and Agent4Rec baselines are restricted profile-generation adaptations; do not use its reported comparisons as evidence that APG4RecSim outperforms the full versions implemented in this project. All performance claims await common-protocol experiments.

## Implementation order

1. Add typed inputs/artifacts, metadata allowlists, deterministic training/probe preparation, and versioned prompts/path/example.
2. Implement three-sample extraction, consolidation, staged caching/resume, and fake-client tests.
3. Implement actual single-trait interventions, stability comparisons, and frozen node mappings; verify call accounting.
4. Add native grouped ranking and the combined-output runner, registrations, coverage/threshold integration, and regression checks.
5. Document the implemented settings and every departure above. Update the manuscript only in a separately requested editing step, after verifying implementation state. No experiment launch is part of implementation verification.
