# Agent4Rec offline response component (v2)

Implemented from the [agreed plan](agent4rec_plan.md). Source: Zhang et al.,
[On Generative Agents in Recommendation](https://arxiv.org/abs/2310.10108),
Sections 2.1.1 and 2.2. This is an **offline adaptation of Agent4Rec's
profile-conditioned response component**, not a reproduction of its complete
interactive simulator. No experimental results follow from the fixture tests.

## Profiles and permitted evidence

The primary MovieLens configuration uses `profile_components=("traits", "taste")`.
The public `fit`, `build_taste`, and `score` interfaces are unchanged. Traits-only,
taste-only, no-summary, and history-summary configurations remain explicit ablations;
the generic low-level constructor still defaults to traits-only for compatibility.

- **Traits:** ordinary MovieLens uses the full permitted training frame, before
  selecting scoring users or taste windows. Activity counts training interactions;
  conformity averages squared deviations from population training item means;
  diversity counts the union of canonical genres. The former 80%-genre-mass rule
  is removed. Item means include the target user's training ratings, as in the
  source formula; no leave-one-out correction, shrinkage, or minimum-rater rule is added.
- **Tiers:** retain activity p60/p90, conformity p25/p80, diversity p33/p66. Values
  `<= low` enter tier 1; remaining values `<= high` enter tier 2; others enter tier 3.
  Constant populations and users missing required evidence receive no corresponding
  trait sentence; manifests record the omission. These are relative history-derived
  measures, not independently validated psychological traits. A sole rating has zero
  deviation from its own item mean.
- **Taste and rating patterns:** at most 20 recent training events, presented in
  chronological order, instead of the source's 25 randomly selected events. Available
  timestamps are sorted stably. Shared temporal task builders omit timestamps in
  some materialized tables but preserve chronological training order; that order is
  retained. Untimestamped custom frames must therefore already have their intended
  order. Ratings 3–5 are liked and 1–2 disliked for taste generation. Individual
  numerical ratings remain visible. Both HIGH and LOW pattern fields are mandatory;
  explicit insufficient evidence is allowed. Both fields reach every response
  interface and are restored on cache hits. Reasons are stored as provenance.
- **Targets:** every observed rating remains an interaction, including ratings 1–2.
  Preference prediction still targets rating >=4. Itemwise preference prompts now
  include that target. Cold-start selects interaction (`target_interact`) or
  preference (`target_like_ge4`) from task metadata and rejects unsupported targets.
  Validation outcomes never extend test profiles.

Cold-start uses only each cold user's permitted k observations (normally 1, 3, or 5)
for traits and taste; activity and external/lifetime activity counts are disabled.
Cold-start candidate statistics use warm training rows, while conformity means and
tier populations use the supplied cold-user initial histories. Coat uses the same
reproducible subset of at most 20 permitted observations for traits and taste;
selection takes the final 20 in stable string item-ID order. The Coat direct-history
baseline uses that same selection and cap. Coat activity and demographic persona
injection are disabled. Steam keeps playtime patterns and omits rating conformity.

## Item profiles and preparation

`runners/in_distribution/agent4rec_items.py` enriches an **Agent4Rec-only task copy**.
It preserves row identities, candidate membership, labels, and the original task.
MovieLens candidate cards expose title, training mean rating (quality), training
rating count (popularity), title-generated genres, and title-generated plot summary.
Canonical genres remain separate and supply the diversity statistic. Generated/canonical
genre disagreement never removes or replaces a candidate.

Mean/count features use the full permitted observed training population, not the
taste window. An unseen item has count zero and explicitly unavailable quality;
held-out feedback is never used to fill it. Per-user temporal partitions still do
not create one global time cutoff for these shared population statistics.

Preparation generates genres and summaries together from title-only messages.
It ignores ratings, labels, and other item/user fields; the artifact records the
model/client identity, sanitized endpoint, settings, exact messages, cache key,
and content hash. It is independently versioned as `agent4rec_title_metadata_v1`.
Scoring reads frozen artifacts only and fails before response calls when required
metadata is missing. It does not silently use canonical genres instead. The existing
vendored summary CSV remains available to other methods; its genre column is not
assumed to have verified generation provenance.

Example preparation (this invokes the chosen metadata model; **not run by tests**):

```bash
python -m runners.in_distribution.agent4rec_items \
  --items data/canonical/ml-1m/v1/items.parquet \
  --client openai_vk_proxy --model gpt-4o-mini \
  --output outputs/agent4rec_item_profiles/ml-1m_v1.jsonl
```

Supply the actual canonical item parquet path. The loader accepts `item_title` or
`title`. Scoring defaults to the output path above; set `AGENT4REC_ITEM_PROFILE_PATH`
for another prepared artifact. Prepare/copy that artifact separately before remote
runs; historical pod bundles do not substitute for it. No paid metadata generation
is launched implicitly by a scorer. A new metadata model/settings/title invalidates
its item cache entry.

For the primary method, generated descriptions appear in candidate cards; taste
uses titles and ratings only. `summary_usage="profile"`/`"both"` is a separate
history-summary ablation. `summary_usage="none"` is the named no-summary ablation.
Both binary and listwise interfaces share the same prepared metadata and profiles.

Domain extensions are not complete movie-profile reproductions. Steam substitutes
game genres/tags and playtime, supplies training interaction counts, and has no
movie-quality or generated movie-plot field. Coat substitutes clothing attributes;
its reranking condition omits both self-selected item-quality and selection-frequency
signals from candidate cards. It still computes its conformity trait from permitted
ratings. These restrictions are recorded in task provenance.

## Combined outputs and diagnostic

The primary binary scorer requests an unrestricted yes/no decision for every supplied
candidate, without revealing or enforcing the true positive count. `predictions.parquet`
stores native labels and identical scores (`yes=1`, `no=0`). `metrics.json` evaluates
classification; `metrics_ranking.json` evaluates the **same scores**, preserving ties
with the existing average-tie, group-then-user aggregation. Report NDCG@5 and Recall@5;
the shared evaluator also retains its existing auxiliary cutoffs. There is no
threshold fit, additional ranking call, or hidden tie breaker.

The separate listwise scorer is an output-interface diagnostic. It requests a complete
candidate permutation using the same profile, item evidence, and response backbone;
its existing diagnostic runner may additionally calibrate rank scores. It does not
replace the binary method. Listwise scorers reject itemwise mode. The default response
budget remains 1,000 tokens at temperature 0 in registered runners; named taste
wrappers retain their separate `gpt-4o-mini` client/model and temperature 0. Model
selection and cross-run search orchestration are outside this implementation.

Existing retry/failure handling is retained. Failed response groups have missing
outputs, with coverage and failure-inclusive metrics where already supported. A profile
failure or complete scoring failure can still abort a runner. Preselect eligibility
before response generation; never silently replace failed users. Eligibility is not
a guarantee against transport/model failures.

Example primary interaction invocation (requires prepared data and metadata):

```bash
python -m runners.in_distribution.interaction_prediction.run \
  --tasks ml-1m_item_stats_cap20_eval_users1000_cg5_m1_seed0 \
  --methods agent4rec_yes_no_litellm_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_smoke
```

The Qwen3.6-27B counterpart and no-summary variants remain registered. These commands
are examples, not experiments executed during implementation.

## Versioning, provenance, and boundaries

Implementation version: `agent4rec_offline_v2`; taste cache version:
`taste_patterns_v2`; prompt versions: `agent4rec_modify_v2`,
`agent4rec_playtime_v2`, and `agent4rec_coat_v2`.

Taste fingerprints include ordered evidence and all visible inputs, rendered messages,
prompt kind/version, model/client/endpoint identity, temperature, and token budget.
Old caches are preserved but cannot satisfy new keys. Manifests separately record
trait/taste evidence budgets and populations, trait omissions/thresholds, item-statistic
partition/hash, item artifact hashes, and profile/response/metadata model roles.
Model aliases do not establish an exact weight revision: record known snapshots when
running experiments. Reusing a taste profile across response backbones is deliberate.

Corrected runners refuse to overwrite existing result directories. Resume checks
reject earlier Agent4Rec implementation versions; fixed-directory convenience runners
use a v2 suffix. Keep historical outputs and caches. Other methods' configurations
are unchanged apart from the agreed Coat direct-history selection/cap adjustment.
No shared evaluator or task-builder target/split logic was changed.

The simulator omits the source's emotional memory, evolving sessions, exit/return loop,
recommender retraining, and LLM-based item pruning. Different history/metadata budgets
across LLMSRec-Syn, RecAgent, LLMRank, and Agent4Rec remain intentional pending later
harmonization; this is a comparison of configured methods, not architecture alone.

Verification uses fake clients: `tests/test_agent4rec_protocol_v2.py` covers full trait
versus capped taste evidence, population invariance, chronological order, cache
invalidation, complete patterns, metadata provenance, target routing, exact combined
outputs/ties, held-out-label independence, and old-run rejection. Existing Agent4Rec,
cold-start, preference, regression, listwise, policy, Coat, and ranking tests remain
part of regression verification. No live calls or experimental claims are involved.
