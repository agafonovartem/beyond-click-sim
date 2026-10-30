# Agent4Rec implementation plan

**Status: implemented on 2026-10-29** as `agent4rec_offline_v2`.
See [implementation details and invocation examples](agent4rec.md).
Verification: 506 fake-client/fixture regression tests passed; the edited manuscript
compiled successfully with the desktop editor's compiler. No experiments or paid
calls were run. Real title-generated item metadata remains an experiment preparation
step; the implementation provides generation/cache tooling and refuses incomplete
primary MovieLens profiles at scoring time.

Implementation detail: some shared temporal task tables omit the timestamp column
while preserving chronological training order. Agent4Rec preserves that order rather
than sorting those histories by item ID. Coat alone uses the documented deterministic
item-ID subset. Shared task builders and evaluators were not modified.

Revise the offline Agent4Rec response component to retain the paper's taste, rating-pattern, social-trait, and item-profile mechanisms within the project's evaluation protocol. The initial paper comparison concerns MovieLens 1M. Preserve each task's targets, materialized candidates, splits, and metric definitions.

This plan incorporates the decisions following the [Compare Agent4Rec method discussion](thread://01a18470-7f65-7b31-a66d-51d2a59d81dd?hostId=local). For ordinary MovieLens tasks, traits use all permitted training interactions, while taste and rating-pattern generation use up to 20 recent training interactions. Agent4Rec, LLMSRec-Syn, and RecAgent retain their distinct profile construction.

Sources: [On Generative Agents in Recommendation](https://arxiv.org/abs/2310.10108), especially Sections 2.1.1, 2.2, and 3.1; the [local implementation audit](../../../docs/paper_readings/agent4rec_implementation_audit_2026_10_29/comparison.md); and the [active manuscript](../../../paper_beyond_click_sim/www_short/main.tex). [agent4rec.md](agent4rec.md) describes the existing implementation separately.

## Scope and agreed boundaries

- Keep up to 20 training interactions for taste and rating-pattern generation. Select the most recent interactions in chronological order on timestamped MovieLens tasks. Compute MovieLens social traits from the full permitted training history before this taste-window selection. For Coat, select a reproducible subset of up to 20 permitted observations and record the rule; Coat has no timestamps. Preserve cold-start's existing k = 1, 3, 5 histories.
- Equalizing history budgets across all methods is deferred. The present comparison includes LLMSRec-Syn's 50-item history and RecAgent's full-history default; this plan makes no equal-budget claim.
- Backbone selection and validation-search orchestration are outside this implementation scope. Keep the backbone configurable and record both the response model and the separate taste-generation model. The manuscript's selection procedure remains a later experiment prerequisite.
- The experiment will select a common eligible user cohort before response scoring, checking required history and metadata to avoid predictable failures. Zero failed groups is the intended run condition. Eligibility cannot guarantee successful model or transport responses; retain existing retry and failure behavior, with no new failure-handling subsystem in this task. Do not silently replace users because their generated responses failed or were inaccurate.
- Changes across Agent4Rec interaction, preference, regression, cold-start, and policy-ranking variants are intended. The planned Coat direct-history baseline change from 24 to 20 observations is also accepted. Other methods' configurations and shared task semantics remain unchanged.
- Do not add interactive sessions, evolving emotional memory, policy retraining, or LLM-based item pruning.

## Taste profiles and target prompts

- For MovieLens and Coat taste generation, group historical ratings 3–5 as liked and 1–2 as disliked, while preserving the numerical ratings for rating-pattern extraction.
- Keep task labels unchanged. In interaction prediction, every recorded rating remains a positive interaction, including ratings 1 and 2. A separate preference task retains its rating >=4 target.
- Carry the generated positive and negative rating-pattern summaries into scoring alongside taste. Extend `Agent4RecUserProfile` with optional pattern fields and render them consistently in interaction, preference, ranking, and regression scorers. Store generated reasons as provenance.
- Preserve Steam's playtime interpretation and label its summaries as playtime patterns. Require both pattern fields in newly generated profiles, permitting an explicit insufficient-evidence response when the history contains no examples for one side.
- Correct itemwise preference prompts to include the same target definition as batch prompts. Route cold-start yes/no tasks by their target metadata and reject unsupported combinations. Listwise scorers must reject itemwise mode.
- Use taste plus the retained social-trait components for the primary Agent4Rec configuration. Keep taste-only and traits-only variants as named ablations. Remove the extra Coat demographic persona; the paper intentionally omits personal identifiers such as age, gender, and occupation from Agent4Rec's user profile.

## Social traits and training evidence

Preserve the paper's distinct mechanisms: activity counts observed interactions, conformity averages squared rating deviations from item mean ratings, and diversity counts the union of observed genres. Replace the current 80%-genre-mass diversity calculation with that distinct-genre count. Steam continues to omit rating-based conformity where only playtime is available.

Keep the existing percentile choices, which the paper does not specify. Make tier boundaries deterministic: values at or below the lower cutoff enter tier 1, remaining values at or below the upper cutoff enter tier 2, and larger values enter tier 3. Retain the proposed handling of constant populations and missing required inputs, recording any omitted trait and its reason. Compute population statistics independently of which users are selected for scoring.

For ordinary MovieLens tasks, use the full permitted training frame for trait construction and select the taste window separately:

- Retain activity, using each user's full permitted training interaction count rather than their capped taste-history length, subject to the constant-population and missing-evidence rules above.
- Compute conformity over each user's full training ratings. Derive item mean ratings from all fitted users' permitted training interactions, before selecting taste windows or evaluation users.
- Compute diversity as the union of genres across each user's full permitted training history.
- Derive trait tier cutoffs from all fitted training users, independently of the evaluation cohort. Selecting fewer users for scoring must not redefine the reference population.
- Use only the selected 20 recent interactions for taste and rating-pattern generation. The 20-item limit is a taste-generation budget, not a limit on the complete Agent4Rec profile's information.

Never use validation/test outcomes or full-lifetime counts extending beyond permitted training. Keep the same training-derived profile for validation and test; do not append validation outcomes to the test profile. Restricted cold-start and Coat retain their task-specific evidence boundaries and disabled activity: their observation counts are imposed by the evaluation or collection design and do not establish natural activity. Do not import additional user history to restore that component in those conditions.

Preserve the source conformity formula without leave-one-user-out means, shrinkage, or a new minimum-other-rater rule. For example, an item's sole permitted rating of 5 gives an item mean of 5 and zero squared deviation for its rater. This is a valid result of the source formula, not independent evidence of psychological conformity. Describe the quantities as Agent4Rec's history-derived trait measures and disclose that interpretation limit.

## Item metadata and protocol restrictions

Include all item-profile fields described in the original paper whenever permitted by the task. The primary MovieLens variant includes titles, quality, popularity, genres, and summaries. This supersedes the earlier plan to omit summaries and popularity.

| Source field or operation | Planned MovieLens behavior | Restriction or disclosed adaptation |
| --- | --- | --- |
| Title | Show the item title with an unambiguous local candidate identifier | Candidate identifiers are an output-parsing adaptation |
| Quality | Show mean rating computed from permitted population training interactions | Never use the target's held-out rating or aggregates computed from held-out feedback |
| Popularity | Show the number of permitted training reviews/ratings for the item | Training count replaces any source statistic that includes validation/test events; label it as a training count |
| Genre | Retain the paper's title-based LLM genre classification for the item profile; keep canonical genres separately for comparison and trait provenance | Reuse verified generated genre artifacts or add the preparation/cache path. Canonical genres alone are a substitution, not a reproduction of the generation step |
| Summary | Show a title-derived LLM item summary | Reuse verified existing artifacts; record source, content hashes, generation model and prompt where available. Do not generate summaries from held-out user feedback |
| Genre-based item pruning | Omit | Agent4Rec-specific item removal would change the shared candidate population. Genre disagreement may be recorded, but must not remove or replace a test candidate |

The complete item profiles are exposed in candidate-response prompts. Retain the source-supported taste-generation inputs and document metadata placement separately; the paper's item-profile description is not evidence that every item field was inserted into every taste-generation prompt. Additional history-summary variants remain named ablations unless the inspected source taste path supports their inclusion in the primary method.

Freeze item descriptions independently of target outcomes and use identical item metadata for a given configuration across validation and test. Reuse the shared training-statistic implementation where applicable, without changing other methods' visible fields or rematerializing different candidate groups. Items with no permitted training observations have count zero and unavailable mean rating; do not fill that mean from held-out data or present zero as a measured rating.

For ordinary MovieLens tasks, conformity item means and candidate item quality/popularity use the same full permitted population training partition, without the 20-item taste cap. Record their source partition and any task-specific population differences in extensions. Per-user temporal splits do not make these population aggregates globally time-consistent; this remains the shared static-offline protocol's disclosed limitation.

For extensions outside the initial MovieLens comparison, keep a field-by-field record of omissions and substitutions. Steam playtime is not a movie rating and must not be presented as such. Coat supplies clothing attributes rather than movie genres or plot summaries. Its existing reranking condition deliberately excludes item mean ratings derived from self-selected observations to avoid feeding that condition's biased item-quality signal into the simulator. Preserve that condition and explicitly document the omission; adding such statistics belongs in a separately identified condition. Check popularity against that condition's declared information policy as well, rather than silently adding a selection-frequency signal. Missing artifacts alone are an implementation prerequisite, not evidence that the evaluation protocol forbids a field.

Keep explicit metadata ablations, including existing no-summary variants, under descriptive names. The primary MovieLens method must not silently fall back to an incomplete item profile. Required metadata and its provenance are checked during the agreed cohort/data preparation.

## Combined classification and ranking contract

- For the primary binary variant, generate one unrestricted yes/no decision per candidate from the same group response. Do not reveal or enforce the true positive count.
- Save each native decision and its corresponding score: yes = 1, no = 0. Classification labels equal those decisions; no validation threshold is fitted for this binary method.
- Compute NDCG@5 and Recall@5 directly from those same binary scores using the existing average-tie evaluation. Preserve ties and use the existing group-then-user aggregation.
- Produce both classification and ranking artifacts from the same generation pass, with no extra ranking call and no hidden tie-breaking model.
- Keep the full-list ranking prompt as an explicitly named output-interface diagnostic, separate from the primary binary method. Hold its response backbone, user evidence, cached profiles, item metadata, and candidate groups fixed to the corresponding binary configuration. Record differences in output instructions and inference budgets.
- Preserve missing outputs and existing failure flags; they are not generated no decisions. Retain the established failure-inclusive metric definitions without expanding failure-handling implementation scope here.

## Cache compatibility and provenance

- Replace the taste cache's user/item-ID identity with a versioned fingerprint covering the ordered selected-history inputs, rendered generation messages, prompt kind/version, model/client identity, and generation settings.
- Restore the complete taste-and-pattern profile on cache hits. Fingerprint every metadata field actually used in its generation; changed ratings or visible summaries must not reuse incompatible tastes.
- Introduce new profile, prompt, and cache versions. Preserve existing caches but prevent them from satisfying the new version.
- Record the taste-history selection and 20-item budget separately from the full permitted trait history, including per-user evidence counts and the trait statistic reference population. Also record genre source, diversity definition, tier rules, omitted traits, visible item metadata, target description, and response/profile/metadata-generation models and known revisions.
- Record item-statistic training provenance and generated-metadata artifact hashes. Version item-generation caches independently of user-profile caches.
- Give corrected runs distinct identities and prevent resume from reusing earlier implementations' results. Preserve historical results and their configuration meaning.
- Preserve public `fit`, `build_taste`, and `score` interfaces. API compatibility does not imply unchanged Agent4Rec predictions; the intended behavioral changes span its affected runners and the agreed Coat baseline adjustment.

## Verification

Use deterministic fixtures and fake clients. Do not launch experiments or paid metadata generation as part of implementation verification.

- Verify that ordinary MovieLens taste and rating-pattern generation use the selected 20 recent training interactions, while traits consume the full permitted training history. A fixture with more than 20 interactions must show that older training events remain eligible for trait statistics without appearing as events in the taste prompt.
- Verify that conformity item means and trait tier cutoffs use the full permitted training population and remain unchanged when the scoring-user subset changes. Held-out outcomes must not affect profiles or item statistics, and validation scoring must not extend the evidence used for test profiles.
- Verify consistent 20-observation selection for the two Coat methods and use of only the permitted k observations in cold-start profiles.
- Verify distinct-genre diversity, deterministic tiers, and the agreed handling of constant populations and missing inputs.
- Verify rating 2 versus 3 taste categories, unchanged interaction labels, and unchanged preference targets.
- Verify that both rating-pattern fields reach all scorers on cache hits and misses, and that itemwise/batch target prompts agree.
- Verify all required MovieLens item-profile fields are rendered, statistics use training only, and generated descriptions do not use held-out feedback. Verify genre disagreement never removes candidates. Test unavailable quality without inventing a rating.
- Verify both metric views use the exact saved binary scores, preserve average ties, and require no extra LLM calls. Verify the optional listwise diagnostic uses the same profiles and declared inputs.
- Verify cache invalidation for changed evidence, models, prompts, or generation settings, and reuse for identical requests.
- Verify candidate identities, labels, evaluation rows, and metric definitions remain unchanged; exercise the accepted Agent4Rec and Coat changes and regress unrelated methods.

Run affected Agent4Rec, cold-start, preference, regression, policy-ranking, Coat, and ranking-metric tests. Inspect representative prompts and manifests. Update the scorer documentation to distinguish primary configurations, adaptations, and ablations.

## Required manuscript updates after implementation

Update the existing Agent4Rec subsection, retaining its original-paper citation. State that MovieLens activity, conformity, and diversity use the full permitted training history, while taste and rating-pattern generation use up to 20 recent interactions. Describe the task-specific restricted-history exceptions, complete item metadata, training-only statistics, generated-metadata provenance, model roles, combined binary/tied-ranking output, and separate listwise diagnostic.

Explicitly explain every protocol-driven omission or substitution, including no item pruning on shared candidates, excluded held-out outcomes, any domain-specific unavailable fields, and the declared Coat information restriction if that condition is reported. Distinguish these from optional ablations and unresolved metadata preparation. Report any remaining substitution of canonical genres for generated genres rather than calling that step reproduced.

Use the claim: an offline adaptation of Agent4Rec's profile-conditioned response component. Do not claim reproduction of the complete interactive simulator or equal history/information budgets across the comparison. Backbone selection remains outside this implementation task, and no experimental results follow from fixture tests.
