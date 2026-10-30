# SimUSER text-only zero: implementation plan

Status: implemented, 2026-10-29; see [implementation documentation](simuser.md) and the linked scorer/runner/tests for evidence. The sections below retain the agreed design specification. Fake-client verification does not establish experimental results or simulator quality.

## 1. Scope and supported claim

Implement **SimUSER(zero, text-only)** as a MovieLens-1M interaction scorer. The user explicitly chose history-initialized `zero` and an explicitly labeled text-only variant. Preserve persona matching, the history-derived profile, episodic memory, graph memory, self-ask retrieval, and iterative watch/skip refinement. Use the same final decisions for classification and ranking evaluation. Do not add a separate ranking scorer.

The primary source is Bougie and Watanabe, *SimUSER: Simulating User Behavior with Large Language Models for Recommender System Evaluation*. Read the [requested April 2025 arXiv v1 PDF](../../../papers/SimUSER%20Simulating%20User%20Behavior%20with%20Large%20Language%20Models%20for%20Recommender%20System%20Evaluation.pdf), particularly §§3.1–3.3, §4.1, Appendix A, Figure 2, and Algorithm 1. The [supplied architecture report](../../../docs/short_paper_selection/architecture_implementation_2026_10_25/reports/simuser.md) covers the [ACL publication](https://aclanthology.org/2025.acl-industry.5/), DOI `10.18653/v1/2025.acl-industry.5`. A [fresh independent reading](../../../docs/paper_readings/simuser_implementation_2026_10_29/reports/simuser25.md) and version comparison found the relevant method parameters agree. Keep the two source versions identifiable in provenance.

Supported claim after implementation and fixture verification:

> We implemented a text-only, history-initialized adaptation of SimUSER's persona-matching, dual-memory retrieval, and iterative watch/skip response architecture. We reconstructed underspecified retrieval and prompting details and evaluated independent candidate groups under our interaction-occurrence protocol. The same native binary decisions supply classification predictions and tied ranking scores.

Do **not** claim exact reproduction, reproduction of the complete interactive simulator, reproduction of Table 1's specific configuration, or that changing the evaluation interface is the only departure. Removing visual perception is an architectural ablation; graph retrieval also requires explicit reconstruction. `Zero` means no simulated warm-up, not no history, persona inference, or LLM calls. There is no simulator LLM finetuning.

Out of scope: Steam/other domains; image acquisition/captioning; simulated training sessions; recommender-policy generation; rating, navigation, fatigue, reflection, and exit experiments; hyperparameter/backbone-search orchestration; matched-history-budget experiments; experimental results. The later implementation request authorized the code; manuscript edits remain a separate follow-up.

## 2. Shared evaluation contract

Follow the current [manuscript protocol](../../../paper_beyond_click_sim/www_short/main.tex), the materialized interaction `Task`, and existing metrics:

- Require explicit per-user chronological training/validation/test splits with the primary fractions 70/10/20. Never silently substitute a random split. Every recorded ML-1M rating is a positive **interaction**, including ratings 1–3. Historical liking and disliking remain useful profile evidence but do not redefine the target.
- Use the same preselected cohort and exact candidate rows/order as the other methods. The primary design targets 1,000 users, at most five groups per user, at most 20 candidates per group, and ratios 1:1, 1:3, 1:9, 1:19. Smaller groups and multiple positives are valid. The scorer must not generate recommendations, prune unfamiliar items, or replace candidates.
- Keep target labels, sampled-negative flags, held-out ratings/timestamps, split names, sampling ratio, and true positive counts out of every model-visible prompt and retrieval query. Candidate metadata and opaque local labels are allowed. Do not enforce a selection quota.
- Build all profiles, memories, graphs, contrasts, and behavioral statistics from permitted **training** interactions. Validation/test labels are available only to the evaluator. Validation outcomes never enter the test snapshot. Sampling logic may exclude known positives without exposing that information to the scorer.
- Freeze the same initialized user snapshot for every validation/test group. Within-group retrieval and refinement are allowed; output from one group must not affect another group, including groups for the same user. Do not update the graph with predicted test interactions.
- WATCH/yes maps to score `1.0` and binary label `True`; SKIP/no maps to `0.0` and `False`. No star-rating threshold, F1-selected threshold, probability conversion, or separate ranking inference is used.
- Compute classification metrics and **NDCG@5 / Recall@5** from these exact saved outputs, using existing expected-average-tie handling and group-then-user aggregation. Ranking has two score levels and cannot order candidates within a level. Do not invent confidence scores or a tie-breaking model.
- Backbone selection, when orchestrated externally, remains independent per dataset/ratio/seed using validation NDCG@5. Expose validation outputs for that purpose; do not add a search controller in this implementation. Use seeds 0/1/2 when called by the common repetitions setup.
- Retain existing failure accounting and the five-attempt group policy. Preflight and cohort preparation avoid predictable missing-data failures but cannot guarantee successful generations. Never replace users or drop failed groups after observing responses. Missing outputs remain missing in artifacts; strict classification uses the established all-negative failure summary, while ranking assigns failed groups zero metric utility directly, not a tied zero-score ranking.

The intended experiments use a preselected cohort with no failed groups. Existing successful-output metric fields may be used for validation selection and final reporting once the relevant runs confirm complete outputs for every requested group and row, including users requiring initialization. With 100% coverage and identical group-then-user aggregation, successful-output and failure-inclusive metrics are equal. Cohort eligibility alone is not that confirmation. Retain the current coverage fields and failure-inclusive outputs; if failures remain, use the existing full-population rules above rather than treating successful-output metrics as equivalent. This adds no new failure-handling infrastructure or search orchestration.

Population training evidence is the complete training partition permitted by the materialized task, not just the users currently being scored. It is also not permission to import otherwise excluded canonical users. As in the shared static offline protocol, other users' training events can occur after the target user's cutoff. This plan does not add LLMSRec's method-specific target-time restriction to SimUSER or claim global historical consistency. A global cutoff would need to cover graph edges, contrasts, aggregates, and all comparison methods in a separate protocol.

Unobserved negatives remain unverified rejections. The output task measures recovery of logged interactions under the specified candidate sampling; it does not validate causal exposure-conditioned willingness or the full interactive simulator.

## 3. Training inputs and metadata

Reuse `require_temporal_task` and `prepare_training` from `runners/in_distribution/interaction_prediction/methods/_llmsrec_syn_data.py` unchanged where compatible. They verify the materialized training projection and align stable event IDs/timestamps. Join ratings and permitted metadata by verified row/item identities; the helper's event table itself contains IDs/timestamps, not ratings. Do not rematerialize evaluation groups.

Create an explicit scorer input allowlist. Whole task frames may contain demographics, sampling flags, or labels and must never be serialized generically. Fit accepts observed training rows only; it must reject sampled training negatives as historical experiences.

| Field | Planned treatment | Source fidelity and restriction |
| --- | --- | --- |
| Historical item title and 1–5 rating | Keep all permitted events for initialization; retain rating in episodic records | §§3.1–3.2.3, Appendix A. Ratings are feedback, not interaction labels |
| Item title, genres, short summary | Include in candidate and retrieved-item descriptions; use titles/genres/ratings in sampled history prompts | Text descriptions and categories occur in the method; Figure 2 lists a GPT-3.5 summary. Exact prompt placement is reconstructed |
| Quality | Use mean observed rating in the permitted population training partition | Figure 2 names quality but does not specify its calculation; this train-only definition is a reconstruction |
| Popularity | Use observed training interaction count | Figure 2 names popularity without defining it. Call the field training interaction count, not all-time popularity |
| User age, occupation, personality | Infer candidate personas; never condition on the target user's actual demographic row | Appendix A excludes supplied user-specific features from ordinary initialization |
| Names and gender | Omit | Source initialization omission, not missing metadata |
| Images and derived captions | Omit the complete thumbnail-perception path | User-selected text-only ablation; never manufacture a visual caption from the plot summary |
| Cast/actor relations | Not present in the initial graph | The source gives an actor-sharing path example, not a complete mandatory schema or supplied cast data. Our genre/interaction graph is an explicit reconstruction, not an exact copy of that example |
| Review text and review interventions | Do not add | Positive/negative review conditions are separate source experiments. ML-1M has no such review text |

The current canonical ML-1M manifest records summaries for 3,883/3,883 items from `resources/agent4rec/ml-1m/movies_augmentation.csv`, pinned to source repository commit `f4ee73e4fc686ef6177e9554d96ecf50add952b0` and SHA-256 `b8e791dc6e77245f6d29825ee08c0916a8a55344ef04ee76ef5686138f7d96d5`. Revalidate that provenance at implementation time. Reusing those frozen textual summaries is a declared metadata-source substitution; they are not SimUSER's captions or verified author artifacts for SimUSER. Do not copy quality/popularity values from an external augmentation file: recompute them from permitted training data.

For a candidate with no training observations, show count zero and quality unavailable. Do not use held-out ratings to fill it. Catalog-only title/genre/summary data may describe such an item. Missing required titles, ratings, genres, or summaries must be reported during preparation, before scoring; do not silently degrade into a title-only method or select a different cohort for SimUSER. If a frozen task lacks static fields already present in the canonical catalog, an isolated ID-checked enrichment of scorer input copies may add them without changing task rows or other methods.

Record distinct evidence budgets: 50 sampled events for preference/persona prompts, all permitted user training events for memories/habits, and the permitted population training partition for graph/contrasts/item statistics. Equalizing these budgets with other methods is deferred.

## 4. Persona matching and fixed user snapshot

Implement the source's two-stage persona construction (§3.1), followed by history-derived attributes (Appendix A).

1. With a user-local deterministic RNG, sample 50 training events without replacement, or all events if fewer than 50. Sampling is random, not a recent-20 window. Render the selected events in stable chronological order after selection. Generate a short taste summary, tagging ratings ≥4 liked, ≤2 disliked, and 3 neutral.
2. Generate exactly five distinct candidate personas from that summary and the same sampled history. Each contains age, occupation, and Big Five integer scores 1–3. No name or gender. Version the allowed menus. Proposed age menu: `{16,18,25,35,45,50,56,65}`; occupations: the public ML-1M occupation categories, including other/unspecified. These menus are our choices because the author lists are absent; using the public category vocabulary does not allow reading the target's true age/occupation.
3. Score each persona with the target-versus-other-history contrast of Eq. 1. Proposed operational defaults: `j=3`; `rho=min(10, number of distinct target training items, number of eligible contrast items)`. Construct three target subsets and three equally sized contrast subsets, shared across all five personas. Contrast events come from other users' permitted training rows, excluding items already in the target's training history. Sample without replacement within each subset, with unique item IDs; overlap across subsets is allowed. Do not exclude items using target validation/test membership. Save the sampled event IDs and RNG identity.
4. For each persona/subset pair, present the target and contrast items in a deterministic mixed order with opaque labels, without observed ratings, provenance tags, or target/other membership. Ask for a hypothetical rating 1–5 per item. Compute `s(p,u) = sum_b sum_target r_hat(p,i) - sum_b sum_contrast r_hat(p,i)`. The highest score wins; ties choose the earliest generated persona. Equal-size summed scores and their mean-scaled version have identical selection. The paper does not specify itemwise versus subset-level ratings, `j`, the main `rho`, contrast exclusion, or ties; these are reconstructions, not claimed source settings.
5. Derive attributes from the full permitted user history: engagement `N_u`; conformity `mean_i (r_ui - R_i_train)^2`; variety = number of distinct canonical genres; pickiness = not picky if mean rating ≥4.5, moderately picky if ≥3.5, otherwise extremely picky. Preserve the formula including the target's own permitted rating in item means. Report raw habit values with definitions, rather than importing Agent4Rec's percentile bins. A larger conformity statistic means greater deviation, not greater conformity.
6. Store the selected persona, all candidate scores, taste summary, attributes, sampled IDs, initialized memory references, and complete provenance as an immutable user snapshot.

Other users' observed items are a contrast distribution, not known dislikes of the target. Persona selection is an inference heuristic, not training on ground-truth demographics or proof that inferred traits are psychologically accurate. The taste-summary and consistency samples may overlap because both are training-only initialization evidence; neither is held-out validation.

If there is no eligible contrast evidence or training rating information, report the preparation problem. Do not bypass matching by substituting a random persona or copying Agent4Rec's profile. Five valid, distinct persona candidates are required; duplicates or malformed values use the declared initialization retry policy.

## 5. Episodic memory and self-ask retrieval

Initialize one record per permitted training event, retaining its event ID and timestamp for provenance. Follow Appendix A's literal zero-initialization templates: rating **>4** liked; **≤2** disliked; otherwise neutral. This deliberately preserves the source inconsistency for rating 4: it is liked in the summary and general graph preference relation, but neutral in the episodic sentence. Keep the actual rating visible, test this boundary, and disclose it; do not silently harmonize the rules. All recorded ratings remain interactions in the evaluator.

Use `text-embedding-3-small` and cosine similarity as the source specifies. A different embedding model is a named substitution, never an automatic fallback. Embedding clients are injectable for tests. Reuse the existing `CachedEmbeddings` utility from `scorers/history/llmsrec_syn_retrieval.py` only with a SimUSER-specific cache path and effective input/version namespace; do not edit its LLMSRec behavior or misdescribe its title-serialization version as SimUSER provenance.

Proposed self-ask reconstruction:

- Form one original query from the full candidate group's titles, genres, descriptions, and decision task; generate exactly two short follow-up retrieval questions using that query and the fixed profile. Never include evaluation outcomes.
- Search the target user's episodic store separately for the original query and each follow-up. Start with `k1=5` **per query**, merge by record ID, retain maximum cosine similarity, and sort ties by stable record ID. This can yield up to 15 initial records. The paper does not specify the merge budget or query count; record this interpretation explicitly.
- Hold these queries and their ranked retrieval lists fixed within the group. Expansion to `k1=7,9` takes larger prefixes of each list and therefore retains prior evidence. Use all available records if a list is shorter; finite memory is not a retrieval failure.
- Present first-person memory text with actual historical rating and an evidence ID. Do not add RecAgent importance scores, decay, forgetting, reflection, or memory consolidation; those mechanisms are not specified for SimUSER.

No candidate responses are added to episodic memory. A query or evidence cache may be reused, but its key must cover all prompt-visible inputs and it must not introduce cross-group behavioral state.

## 6. Graph memory and executable retrieval reconstruction

Retain a real training-derived graph rather than replacing this component with semantic search alone. The paper gives PathSim, `alpha=.8`, embedding weight `.25`, and initial `k2=3`, but not sufficient graph/path details to reproduce the exact retrieval. Use the following fully specified **first reconstruction**, identified as `simuser_graph_v1`:

- Nodes: permitted users, catalog items, and canonical genres. Edges: observed user–item training interaction with its rating/event provenance, liked for rating ≥4, disliked for rating ≤2, and item–genre membership. Rating 3 has an observed relation without a preference edge. Static metadata edges may exist for items with no training interactions. Never construct a target's held-out user–item edge.
- Let `B[u,i]` be binary observed-training membership and `C[i,g]` binary genre membership. Item path-count matrices are `M_UI=B.T @ B` for item–user–item and `M_G=C @ C.T` for item–genre–item. For each channel use the source normalization `S_M(x,y)=2*M[x,y]/(M[x,x]+M[y,y])`, with zero for a zero denominator. Use sparse or on-demand operations, not all simple-path enumeration.
- Set `S_item(x,y)=.5*S_UI(x,y)+.5*S_G(x,y)`. This path catalogue and equal channel weights are our choices. Likes/dislikes remain typed evidence, but are not silently treated as the same as interaction occurrence or used as a new learned similarity.
- Define `S_user(u,y)=mean_{h in unique training items of u} S_item(h,y)`. This explicitly substitutes a history projection for the unspecified heterogeneous user–item PathSim term. It uses user–item–user–item and user–item–genre–item evidence paths. It is **not** claimed to be the author's unrecoverable heterogeneous normalization.
- Embed item title/genres/frozen short summary and set `S_sem(x,y)=(1+cos(E(x),E(y)))/2`. Rank retrieved item `y` for query item `x` by `R(u,x,y)=.75*(.8*S_item(x,y)+.2*S_user(u,y))+.25*S_sem(x,y)`. The `.8` and `.25` weights come from the source; the convex mixing, cosine rescaling, and embedding text are our explicit choices because the complete source expression is absent.
- For each supplied candidate `x`, retrieve the top `k2=3` catalog items `y != x`. Other group candidates are not specially excluded. Expand to 5 and 7 during refinement. Deduplicate display records across candidates while retaining query-to-evidence mappings; ties use stable typed item IDs. If fewer items exist, use what is available.
- Return title, genres, summary, training quality/count, target historical rating when actually observed, and up to two deterministic structural witness paths per selected item for each nonzero structural channel. Distinguish target ratings from population mean ratings. Include the user-to-history prefix for user-conditioned paths. Do not fabricate a path for a semantic-only match; mark that evidence explicitly as semantic-only. Retain full path counts/scores and selected witness IDs in diagnostics.

When a candidate has no collaborative edges, genre and semantic channels can still operate. Do not silently rescale the missing channel's weight or borrow held-out edges. Retaining genre paths while omitting the illustrated actor paths is a documented metadata/path-schema choice, not something the protocol requires.

The source's sentence excluding relationships from “training/validation and test sets” conflicts with history-based initialization. Our rule is unambiguously **training interaction edges only**, plus outcome-independent catalog metadata. Disclose that resolution rather than attributing it verbatim to the paper.

## 7. Group response and bounded refinement

Keep the source's watch/skip interface and pickiness sentence, `You are {pickiness} about movies`. Add the common task definition: predict interaction occurrence for the supplied candidates; a recorded low rating still represents an interaction. Do not turn this into a rating-≥4 prediction task. The exact §4.1 experimental prompt is absent, so this is a documented prompt reconstruction.

Render the same interaction-target definition in the initial watch prompt, every consistency-check prompt, and every revision prompt. State that a user can interact with an item and subsequently dislike it: anticipated dislike or a low predicted rating is not, by itself, a logical contradiction of WATCH, and WATCH does not require an expected rating of at least 4. Preferences remain evidence for predicting interaction, and refinement may still change WATCH to SKIP when warranted; there is no rule preserving the initial decision. Keep actual candidate held-out ratings hidden. This clarification applies to response/refinement stages and does not replace the source's hypothetical-rating task used for persona matching or its historical liking categories.

One group attempt performs:

1. Restore the immutable user snapshot, form the self-ask queries, and retrieve both memories at initial limits.
2. Ask for a complete watch/skip vector with a short explanation and optional evidence IDs for each supplied candidate. Use strict JSON with unique local candidate labels, Boolean `watch`, brief `reason`, and `evidence_ids`. No candidate may be absent, duplicated, or added; no output quota or ranking permutation is requested.
3. In a separate consistency-check call, supply the shared interaction-target definition, profile, candidate metadata, current decisions, and retrieved evidence. Request per-candidate contradiction/insufficient-evidence flags and concise reasons. This check assesses the interaction prediction's consistency with supplied evidence, not accuracy against hidden labels or satisfaction of a liking threshold.
4. If the check requests revision, increase both limits by `delta_k=2`, supply the same interaction-target definition, previous decisions and check plus the expanded evidence, and request a complete revised vector; then check it again under that same target. Allow at most **two refinement rounds** after the initial decision. Preserve original candidate order throughout.
5. Stop when the check requests no revision, when no additional evidence exists, or after two refinements. Save the last complete valid vector. Record unresolved consistency flags and the stopping reason; do not convert subjective uncertainty into a technical failure or force all-no answers. Malformed responses or nonexistent cited evidence IDs are technical validation failures.

These call boundaries, two follow-ups, increment, round cap, stopping rules, and output schema are reconstructed engineering choices. They preserve multi-round watch refinement while bounding cost. They do not reproduce the separate causal **navigation** refinement in Appendix A.5.

Omit subsequent rating/feeling generation, navigation, causal action selection, reflection, memory updates, and exit interviews: they occur after watch selection in the described pipeline and have no later response to condition in this independent one-step setting. No simulated training warm-up is run, as the user selected zero. Do not claim the omitted stages were tested, or that they would be unnecessary for a trajectory study.

## 8. Models, cost boundaries, retries, and caches

Register Qwen3-8B and Qwen3.6-27B smoke/full configurations to match the current project integration pattern; retain injectable clients and an explicit configurable GPT-4o-mini reference setting. Each configuration uses its declared backbone for taste, persona generation/scoring, self-ask, decisions, and consistency checks. Do not silently use a different model for initialization. The source default is GPT-4o-mini; Qwen is a substitution. No finetuning is added.

Proposed decoding: temperature 0, provider seed where supported, no hidden model-thinking mode in the Qwen wrappers, and per-stage maximum response tokens: taste 1,024; five personas 2,048; consistency ratings 1,024; self-ask 512; watch/revision 2,048; consistency check 2,048. These values are not supplied by the paper. Record actual model/serving revisions, effective options, context limits, and usage. Bound short descriptions/reasons with versioned rendering rules; never silently drop candidates or retrieved evidence to fit a context window. Preflight should detect deterministic context-capacity problems before experimental calls.

Separate retries from semantic refinement. Use at most five attempts per initialization stage; retain completed valid stages. At group level, reuse `_score_groups` with at most five whole-pipeline attempts, restoring the same snapshot each time. Avoid nested five-by-five stage retries inside group attempts. Follow the existing transport/parser failure semantics; do not retry because a valid decision is negative, surprising, or inaccurate. Use fake clients for verification only.

With `j=3`, successful uncached initialization requires one taste call, one five-persona call, and 15 persona-rating calls: **17 LLM calls/user**, plus embedding preparation. A successful group uses one self-ask call and one to three decision/check pairs: **3–7 LLM calls/group**, plus query embeddings. Classification and ranking add **zero** extra generation calls. Report retries, actual calls, tokens, embedding batches/cache hits, and elapsed time separately; do not present the paper's inconsistent dollar-cost figures as our budget estimate.

Provide a no-network `preflight_only` path that reports evidence/metadata coverage, intended groups, source fingerprints, and uncached call bounds. Prompt lengths depending on generated text may require conservative schema/field bounds until cached initialization exists; distinguish bounds from measurements.

Use atomic, versioned JSON/Parquet checkpoints rather than unsafe object serialization. Cache keys cover effective permitted evidence and ratings, sampled event IDs, metadata hashes, exact rendered messages, prompt/schema/graph versions, model/embedding identities, decoding settings, user-local seeds, and snapshot fingerprint. Group caches additionally cover ordered candidates, queries, retrieval/refinement configuration and snapshot identity, never hidden labels. Keep validation/test outcomes out of cache values used for inference. Keep full source hashes as provenance separately from effective-evidence identities where necessary.

Use local RNGs derived from run seed, typed user/group IDs, and stage. Sequential, parallel, resumed, and reordered evaluation must not change deterministic inputs or cached outputs. Live hosted model generations need not be bitwise reproducible even at temperature zero; record that practical limit rather than asserting otherwise.

## 9. Code boundaries and artifacts

Add an isolated package under `src/beyond_click_sim/scorers/simuser/`:

| File/component | Responsibility |
| --- | --- |
| `config.py`, `state.py` | Validated settings, immutable snapshots, schema versions, stage provenance |
| `profiles.py` | Taste, five-persona matching, contrasts, source habit arithmetic |
| `episodic.py` | History templates, embeddings, self-ask retrieval and stable expansion |
| `graph.py` | Training graph, path counts, specified fusion and witness paths |
| `prompts.py` | Explicit allowed inputs, stage rendering and strict response validation |
| `scorer.py` | `SimUSERYesNoScorer.fit`, explicit `build_agents`, grouped `score`, diagnostics/checkpoints |

Maintain the public `Scorer.fit(X,y) -> self` and `score(X) -> Series` contract; optional training-event/configuration inputs follow the existing RecAgent pattern. `fit` performs deterministic validation/preparation; explicit `build_agents` performs inference-dependent initialization. `score` returns index-aligned binary numeric scores and raises the existing compatible group error on invalid output. Do not expose a separate ranking model.

Add `runners/in_distribution/interaction_prediction/methods/simuser_yes_no.py`. It should initialize once for the union of requested validation/test users, score each supplied split from the same snapshots, and reuse existing prediction/coverage/classification/ranking helpers. A test-only invocation remains possible with an externally fixed configuration. Do not copy RecAgent's cognition or assume its current test-only runner supplies validation selection automatically.

Add scorer exports and opt-in registrations in `methods/__init__.py`, for example `simuser_zero_text_only_litellm_qwen3_8b_{smoke,full}` and `simuser_zero_text_only_litellm_qwen36_27b_{smoke,full}`. Keep `zero` and `text_only` in public method identities and manifests. Do not add SimUSER to default expensive sweeps or change another method's registrations/defaults.

Retain existing artifact names and common columns:

- `predictions.parquet`: exact original row/user/item/group identity, split, evaluator-owned target, native binary score and prediction, with missing outputs preserved. Scoring has no access to the target column written by the evaluator.
- `metrics.json` and `metrics_ranking.json`: corresponding split metrics, group/user aggregation, primary F1 and NDCG@5/Recall@5, successful coverage and existing strict failure-inclusive views.
- `manifest.json`, `preflight.json`, `llm_errors.jsonl`: task/source/configuration provenance, method status, costs, initialization/group failures, and coverage.
- Additional SimUSER artifacts: versioned user snapshots and `simuser_trace.jsonl` with stage/attempt IDs, candidate/evidence identities, selected persona scores, retrieval terms/path witnesses, refinement flags, stop reasons and usage. Short evidence-based explanations are sufficient; do not require unrestricted private reasoning traces.

Reuse shared code unchanged when its semantics fit. If an essential capability is missing, prefer a small SimUSER-local adapter; do not refactor global evaluators, task builders, or other scorers as incidental cleanup. The implementation acceptance criterion is **behavioral isolation verified against the then-current workspace**, not merely new filenames. Existing Agent4Rec work is in progress and must not be reset or overwritten. Record the starting workspace state and confine edits to this package, its runner/tests/docs, and additive imports/registrations.

## 10. Implementation sequence and acceptance tests

1. Freeze the input contract and `simuser_graph_v1`/prompt/config versions; add deterministic fixtures, fake LLMs, and fake embeddings. Inspect the current workspace again before modifying code.
2. Implement training validation, metadata mapping, profile arithmetic, and complete persona matching with checkpoints and cost preflight.
3. Implement episodic and graph retrieval, including exact toy path-count tests and evidence provenance.
4. Implement bounded watch/check/refine scoring and immutable group resets.
5. Integrate the opt-in runner, combined artifacts/metrics, and smoke/full wrappers.
6. Run focused and relevant existing-method regression checks; document invocation examples, model roles, data flow, known adaptations and actual implementation status. Only then describe implemented behavior in the manuscript when requested; compile its existing source with the built-in compiler after such an edit.

Acceptance criteria:

- **Evidence boundaries:** with fixed materialized candidates and training evidence, changing validation/test labels, held-out ratings/timestamps, sampled flags, actual demographics, or evaluation order does not affect prompts, snapshots, retrieval, decisions, or persona selection. Other users' permitted training evidence can affect graph/contrasts as intended. Test that the population source does not shrink when scoring fewer users.
- **Sampling and profile arithmetic:** a history longer than 50 verifies random-50 summary/persona inputs while older/unselected training events remain in habits and memories. Check five personas, shared contrast draws, `j/rho`, score mapping, ties, pickiness boundaries, conformity and distinct-genre variety. Do not accidentally interpret synthetic contrast membership as evaluation negatives.
- **Source inconsistency:** explicitly test rating 4 as liked in summary/graph and neutral in initial episodic text, with rating 4 and low ratings all still positive observed interactions.
- **Graph fidelity to our specification:** hand-computed fixtures verify both PathSim channels, zero denominators, user projection, `.8/.2` and `.75/.25` fusion, normalized semantic similarity, stable ties, real witness paths, query exclusions, and train-only rating/quality/popularity provenance. Include an item without training edges. No held-out edge may appear in the graph.
- **Self-ask and refinement:** verify per-query retrieval, merge/deduplication, monotonic evidence expansion, no-refinement and two-refinement paths, memory exhaustion, unresolved flags, and exactly the declared stopping behavior. Malformed/unknown candidate or evidence IDs fail through the existing retry path. Semantic disagreement alone does not trigger technical retries.
- **Target consistency through refinement:** inspect rendered initial/check/revision prompts to verify that every stage receives the interaction-occurrence definition and the interaction-versus-liking distinction. A fake-client fixture must permit a complete WATCH decision whose explanation anticipates dislike, without parser rejection, a hard-coded rating cutoff, or automatic conversion to SKIP. Also verify that supported revisions remain possible and actual held-out ratings never enter these prompts. These tests verify instructions and control flow, not a guarantee that a live LLM always reasons correctly.
- **Fixed-state behavior:** using deterministic clients, group order, worker count, cache hits/misses, and interrupted/resumed initialization preserve outputs; scoring never changes the saved snapshot or graph. Record calls to show 17 successful initialization calls at `j=3` and 3–7 per group without extra ranking calls.
- **Combined contract:** a fixture with both yes and no decisions verifies that saved scores equal saved labels mapped to 0/1, and both metric files consume those scores. Include ties at the cutoff; do not resolve ties by label or order. Validation metrics use the same group/user aggregation as test. A complete-coverage fixture verifies that successful-output and failure-inclusive metrics agree; coverage is measured against the original requested groups and rows.
- **Failures:** initialization and exhausted-group failures leave missing row outputs and appear in coverage and strict summaries. A failed ranking receives zero utility, whereas a successful all-no group receives the expected ranking value of a tied list. Neither is silently dropped.
- **Isolation:** compare representative existing `llm_yes_no`, LLMRank, LLMSRec-Syn, RecAgent, Agent4Rec and popularity fixtures against the current baseline. Existing method names, prompts, defaults, artifacts, task candidates, registry entries and metric semantics stay unchanged; imports make no network calls. Run relevant existing scorer/runner, registry, temporal-training and ranking-metric tests, including tests affected by additive exports.
- **Documentation:** all reconstructed defaults and omitted components below are discoverable from documentation and run manifests. No fixture verification is described as empirical validation of simulator quality.

Use fake clients only for this implementation verification; no experiments, paid calls, package installs, or model downloads are required.

## 11. Required divergence disclosures and manuscript follow-up

| Difference from source | Reason/classification | Consequence for claims |
| --- | --- | --- |
| No thumbnails, GPT-4o captions, atomic visual checks or MLLM | Explicit user-selected architectural ablation | Name the method text-only; no visual-perception or full-SimUSER claim |
| Zero initialization; no simulated training pages | Source-supported variant chosen by user | Do not equate with `sim` or assert Table 1 used zero |
| Only independent watch/skip response, without later ratings/navigation/reflection/memory growth | One-step protocol adaptation | Does not test full simulator trajectories, dynamic adaptation, exit or policy outcomes |
| 70/10/20 per-user split; fixed training snapshot | Shared protocol versus source 80/10/10 temporal split with incomplete granularity | No direct numerical reproduction of source tables or global time-consistency claim |
| Shared groups up to 20, up to five/user, extra 1:19 condition, supplied ordering | Shared protocol versus native 20-item alignment and four-item interactive pages | Do not paginate into a simulated session or infer a natural acceptance rate |
| Native 0/1 scores also used for tied NDCG@5/Recall@5 | Additional evaluation of the same binary response | Ranking evaluates coarse decisions, not a native full-order ranking or calibrated propensity |
| Graph schema, actor-path omission, user projection, semantic mixing | Source underspecification plus available metadata | A declared graph-retrieval reconstruction; not exact author PathSim behavior |
| Train-only graph and aggregates | Protocol boundary and resolution of ambiguous source exclusion text | No held-out outcomes; shared static temporal limitation remains |
| Persona menus, contrast construction, itemwise ratings and sampling sizes | Unreported source implementation details | Record as defaults, not source hyperparameters; no demographic-ground-truth conditioning |
| Literal rating-4 episodic asymmetry | Conflicting source definitions | Preserve and disclose in fidelity notes; do not alter interaction labels |
| Summary artifact reuse; train-only quality/popularity; canonical genres | Metadata provenance/operational reconstruction | Include source-supported text fields but do not claim identical metadata to authors |
| Self-ask merge rule, finite refinement, strict JSON, retries, caches and token budgets | Reproducible operational reconstruction | Prompt/call/compute differences must be reported |
| Qwen response backbones, temperature 0 | Project model configuration; source default GPT-4o-mini, decoding unspecified | Configured-method comparison, not isolated architecture comparison |

The manuscript already has citation key `bougie2025simuser` in `agents.bib`; use that original-work reference, not the unrelated `xiang2024simuser` mobile-usability paper. When implementation is verified and a manuscript update is requested, replace the planned SimUSER placeholder with the supported claim in Section 1, cite the original work, and name the text-only zero variant in tables. Put essential adaptations in the method paragraph and mathematical reconstruction/defaults in reproducibility details. Continue disclosing sampled unobserved negatives and unequal histories, metadata, pretraining and call budgets across methods.

There are no remaining user-preference questions: both initial scope choices are resolved. The previously proposed explicit defaults are now recorded in the implementation configuration. They remain operational reconstructions, not facts recovered from the source. See [simuser.md](simuser.md) for the implemented contract and invocation examples.
