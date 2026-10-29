# RecAgent: offline profile–memory–action adaptation

Implemented in `src/beyond_click_sim/scorers/recagent/`, with the isolated runner
`runners/in_distribution/interaction_prediction/methods/recagent_yes_no.py`.
MovieLens-1M interaction prediction only. No fine-tuning is performed.

The defensible claim is:

> We reimplemented RecAgent's profile–memory–action response architecture using
> available author-code details and adapted it to offline interaction prediction.

This is not an exact reproduction of the published discrimination experiment or
of the complete interactive simulator. Original paper: Wang et al., *User Behavior
Simulation with Large Language Model based Agents*,
[arXiv:2306.02552](https://arxiv.org/abs/2306.02552).
Code reference: [RUC-GSAI/YuLan-Rec, revision
fff935570079502d5a17509b0baf590becd1eaab](https://github.com/RUC-GSAI/YuLan-Rec/tree/fff935570079502d5a17509b0baf590becd1eaab).
That code revision is not established as the published experiment revision.
The source-by-source audit is in the workspace's
`docs/paper_readings/recagent_plan_audit_2026_10_29/comparison.md` (outside this Python repository).

## Running and preflight

Registered methods:

- `recagent_yes_no_litellm_qwen3_8b_smoke`
- `recagent_yes_no_litellm_qwen3_8b_full`
- `recagent_yes_no_litellm_qwen36_27b_smoke`
- `recagent_yes_no_litellm_qwen36_27b_full`

Smoke limits **test groups to five**, not users, histories, or memory operations.
Full evaluates every group in the supplied task. Both initialize only evaluated
users. No validation configuration search, threshold selection, or separate
ranking scorer is used. The fixed adaptation does not select a backbone across
seeds; run/backbone selection for a paper remains an external experimental step.

Examples from the Python repository (choose a registered ML-1M task):

```bash
# No LLM client construction, embedding loading, or weight downloads:
.venv/bin/python -m runners.in_distribution.interaction_prediction.methods.recagent_yes_no \
  --task ml-1m_cap20_eval_users1000_cg5_m1_seed0 \
  --output-dir outputs/recagent_preflight --preflight --max-groups 5

# After explicitly provisioning dependencies, local embedding weights and serving:
.venv/bin/python -m runners.in_distribution.interaction_prediction.run \
  --tasks ml-1m_cap20_eval_users1000_cg5_m1_seed0 \
  --methods recagent_yes_no_litellm_qwen3_8b_smoke
```

List exact task names in `TASK_BUILDERS`; the examples above are checked registry
entries. Task materialization must include the existing candidate summaries and
recorded demographics for the defaults. The dedicated runner also exposes
`--no-candidate-summaries`, `--no-demographics`, `--max-history-items`,
`--max-groups`, `--cache-dir`, `--model`, `--model-revision`, and
`--embedding-revision`. These explicit variants are recorded in artifacts.

The optional `recagent` dependency group supplies Sentence Transformers. Provision
it with `uv sync --extra recagent` when preparing an inference environment; no
packages or weights were installed by the implementation tests. MPNet loads only
from the local Hugging Face cache and fails if the requested snapshot is missing.
The scorer never initiates weight downloads. Normal imports and preflight do not
require Sentence Transformers. Inject an `EmbeddingProvider` for tests.

Before persistent reuse/resume, supply the actual serving checkpoint revision
through `model_revision`, `--model-revision`, or
`BEYOND_CLICK_SIM_RECAGENT_MODEL_REVISION`. This is a caller assertion, not a
server attestation. Without it, caches use a private session namespace and cannot
silently reuse another process's initialization. Provide a stable cache directory
and revision to resume an interrupted run; the frozen initialized state is then
reused rather than regenerated. The scorer records the embedding's resolved
snapshot commit, encoding settings, and library versions, never just `main`.

Preflight reports selected users, events, profile blocks, groups, rows, metadata,
source hashes, and candidate checkpoint-file count. It does not promise valid
cache hits before resolving the embedding revision. Its call estimate is
`profile_blocks + 2 * events + 2 * groups`, excluding promotions, reflections and
retries. The two per-group calls are summary and action; memory reads may cause
additional calls. Full-history initialization can be expensive.

## Training evidence and initialized state

The existing `_llmsrec_syn_data.prepare_training` helper is reused unchanged to
recover event timestamps and occurrence identities. It verifies the complete
canonical training projection against the materialized task. Unsupported datasets,
random splits, different canonical versions, source changes and inconsistent
projections fail before constructing an LLM client. Only verified **training**
events leave that helper; test candidates are never regenerated.

`RecAgentYesNoScorer.fit(X, y, profile_user_ids=...)` validates and retains inputs
without inference. The direct scorer API requires index-aligned `training_events`;
the runner supplies the verified mapping. Observed training interactions must all
have target 1. `build_agents()` performs/resumes initialization. `score(X)` accepts
one user and candidate group and returns one aligned float 0/1 per row.

- All training events are used by default. The optional latest-20 variant must be
  explicitly selected. Stable input order resolves equal timestamps and repeated
  item occurrences retain distinct event identities.
- Recorded gender, age **category**, and occupation initialize a neutral named
  profile. ZIP, fabricated behavioral classes, relationships, and personality
  fields are excluded. A demographics-off option is available. Evidence-grounded
  character insights are allowed in memory; they do not overwrite recorded facts.
- Consecutive blocks of at most 20 events update a summary bounded to 120 words.
  Each chosen event is separately replayed through sensory memory using its title,
  genres and recorded rating. Observations describe interactions, without invented
  reviews or feelings. Processing every event does not guarantee its retention.
- Replay uses historical Unix seconds for memory creation, promotion, access and
  forgetting. The decision clock is the final complete training event +1 second,
  even with a shorter initialization history. No held-out event time is used.
- A separate recent-history buffer holds the last ten selected movie titles. It
  is not the full history or STM. No social history is fabricated.
- Candidate metadata defaults to titles, genres and the project's provenance-checked
  summaries. History/profile summaries of individual movies are disabled. Optional
  training-derived item means/counts are off by default. Enabling them imports the
  shared whole-training-population statistics and its absence of a global time
  cutoff; it is a distinct information-access variant.

## Memory algorithms and source mapping

The port removes the old LangChain/FAISS dependency chain while preserving the
audited operational rules. Source line numbers below refer to the original
`agents/recagent_memory.py` at the cited revision.

| Component | Implemented default | Source |
|---|---|---|
| Sensory | One observation, one compression call, integer importance call; importance = integer / 10 × 0.9; retain strictly >0.62 (7–10) | 72–176 |
| STM | Capacity 10; cosine similarity plus 0.1 for the final existing record; p=sigmoid(similarity); reinforce when p≥0.7 and local random draw≤p | 186–210, 338–367 |
| Promotion | After three supports, remove the STM record and store its original text plus a generated character insight; score the insight's importance | 225–292, 880–929 |
| Eviction | Protect the newly inserted record; evict by lowest importance, then fewest supports, then stable order; remove one matching support occurrence from other STM records | 294–322 |
| LTM relevance | Exact float32 squared L2 search; closest 100 receive `1 − squared_L2 / sqrt(2)`; no normalization added | retriever + `simulator.py` 132–165 |
| LTM ranking | Every stream record participates in recency/importance; add `0.99^hours_since_access + importance` to any relevance; select five active records with stable ties and refresh their access times | 17–63 |
| Reads | Retrieved LTM reinforces STM without sensory insertion; include all surviving STM; save promotions, then reflect/forget when applicable | 473–507, 823–838, 880–894 |
| Reflection | Accumulated LTM importance strictly >10; one question from last 50 stream records, one insight grounded in retrieved IDs; merge cited parents when sigmoid(cosine)≥0.72 | 550–649, 714–742; simulator threshold setting |
| Reflection importance | Mark merged parents with importance 1 and current access time; then average cited parents' importance for the new insight; no recursive reflection; reset accumulation | 597–649, 714–742 |
| Forgetting | r=0.99^hours; q=1−max(r^1.5,0.01)(importance+r)/2; normalize across the stream; independently sample each eligible record; protect access within 24 hours inclusive | 654–742 |

Deleted/merged records remain as tombstones with their original indexed vectors;
they can consume a top-100 relevance slot and participate in forgetting-weight
normalization. Their marker, importance 1 and refreshed access time follow the
source. They cannot appear as active retrieved memories.

Embeddings are CPU `sentence-transformers/all-mpnet-base-v2`, 768 dimensions,
float32, batch size 1, without added vector normalization. MPNet's 384-token encoder
limit is explicit and recorded: embedding inputs follow the encoder's normal
truncation, while original memory text and LLM prompts remain intact. NumPy
implements the source flat-distance calculation; retrieval does not substitute
cosine for L2. Cosine remains the reinforcement/merging similarity.

## Action, prompts, and isolation

For each candidate group the scorer clones the initialized profile, memory and
clock and derives a local RNG stream from its frozen state, seed, group identity
and ordered visible candidates. First it loads memory for the profile-summary
chain (preserving side effects even though the summary prompt itself uses only the
profile). It then loads memory again for the final action. Read-time reinforcement,
promotion, reflection and forgetting occur only in that private copy. Nothing is
written back after predicting. Whole-group retries restart from the same state
and random stream. No process-global RNG or group-order-dependent agent state is
used; one worker is the default, and scoring supports multiple workers.

Versioned prompts are in `recagent/prompts.py`. Changes from author prompts:

- Sensory instructions retain movie-focused one-sentence compression, with an
  explicit prohibition on inventing feelings/reviews from rating events.
- Importance rules retain the distinction between generic navigation/chat and
  specific movie information, with stricter integer-only validation.
- Promotion preserves inferred character insights and adds explicit evidence
  grounding. Reflection preserves question→evidence→insight and uses validated
  bracketed source IDs.
- Interest initialization is our rolling update prompt; the paper does not
  disclose its historical profile/memory initialization procedure.
- Profile-summary and action composition follow the inspected chain. The action
  asks for unrestricted `Ck: yes/no`, replacing navigation/fixed-count selection.
  Supplied history/memory defines prior experience; candidate descriptions may
  describe unfamiliar movies and general knowledge may interpret them without
  claiming prior exposure. The source's restrictive knowledge sentence is adapted.
- Recorded demographics remain fixed; the summary is not written back to them.

Candidates have an explicit item-metadata whitelist. Targets, held-out ratings,
held-out timestamps, sampled flags and positive counts never enter prompts,
initialization keys, or group seeds. Group input/row order is preserved.

All LLM stages use the selected frozen Qwen3-8B or Qwen3.6-27B through the existing
`litellm_local` client, temperature 0 and thinking disabled. Prose/action budgets
are 256 tokens; importance uses 16. The author-code generic budget was 1,500, and
the paper names ChatGPT without a fully specified per-stage checkpoint. Changing
the backbone or prompts may change behavior even without fine-tuning.

## Persistence, repairs, and failures

Agent snapshots use checksummed JSON with numeric vectors; embeddings use numeric
`.npy` files with `allow_pickle=False`. Writes use temporary files and atomic
replacement. No pickle/dill is loaded. Checkpoint keys cover ordered events,
visible history content, demographics, clocks, prompt/source versions, model
configuration/revision, embedding identity/settings, memory parameters and seed.

Each profile block and each replay event is a transaction. On failure, retry from
the previous state and RNG up to five times; only success advances a checkpoint.
An exhausted user remains unavailable, with missing outputs for all that user's
groups. No generic fallback profile is substituted. The existing `_score_groups`
helper retries the complete summary/memory/action operation up to five times.
LLM-client retries are not added inside individual stages.

Explicit robustness repairs relative to inspected code:

- Stable record/support identities replace text-keyed eviction and duplicate-text
  ambiguity. Evicting one occurrence only removes one support with that identity.
- Valid 1–10 importance is required; malformed output fails instead of silently
  becoming zero. Reflection must cite actual retrieved records; no fallback ID 0.
- Empty retrieval/forgetting stores are valid. All-zero forgetting weights are a
  no-op. Invalid, zero-norm or non-finite vectors/distances fail explicitly.
- Reflection skips when no active evidence remains. Tombstones use explicit
  status rather than recognizing marker strings inside natural-language content.
- Historical time is passed explicitly rather than using wall-clock fallbacks.
- Bounded profile responses are validated, not silently sliced. Server context
  errors, empty replies, output-length truncation and malformed labels fail the
  transaction/group. No LLM history/memory truncation is performed by this port.

`recagent_trace.jsonl` streams stage/user/group/attempt, prompts, replies, reported
token usage, elapsed times, cache hits, retrieved IDs and memory transitions.
`drain_diagnostics()` returns pending records for direct scorer use; when a sink is
provided, records are sent directly instead of retained. The runner streams traces
so interruptions do not erase earlier successful-call diagnostics.

Artifacts:

- `predictions.parquet`: original test rows, target, float `score`, nullable binary
  `prediction`; both are missing for failures.
- `metrics.json`: parsed-output classification, coverage, and the existing
  `test_failure_as_negative` convention.
- `metrics_ranking.json`: those **same binary scores**, average ties, NDCG@5 and
  Recall@5 (also existing @1/@3/@10); failed groups receive zero utility in
  `test_failure_as_zero_group`, not chance credit from tied-zero scores.
- `manifest.json`: configuration, revisions/source hashes, initialization/cache
  coverage, verified training provenance, task manifest and method claim.
- `preflight.json`, `llm_errors.jsonl`, `recagent_trace.jsonl`, `recagent_cache/`:
  setup counts, errors, detailed trace and resumable state.

If all groups fail, artifacts and failure-inclusive metrics are saved before the
runner raises. Parsed-only metrics are null in that case. The normal output
directory is covered by existing ignore rules; bulky caches/traces are not paper
results. No experiments or paid LLM calls were launched during implementation.

## Experimental boundaries and verification

The protocol uses per-user temporal fractional splits, shared multi-positive
groups and sampled unobserved negatives. The source discrimination experiment
held out the last one/three events and requested the known number of positives.
Our unrestricted yes/no interface removes that count, uses different cohort/slate
sizes and allows both classification and tied ranking evaluation. Binary outputs
provide no ordering within yes or no tiers and are not calibrated probabilities.
Unobserved negatives are not known rejections. Rating timestamps are not verified
watch/exposure times; all targets are predicted from the fixed training-end state.

The full social network, navigation, activity scheduler, executed watch actions,
generated post-action experiences, policy learning, interventions and human role
playing are omitted. Their omission narrows scope to independent one-step
prediction. Historical profile compression/replay, clock policy, metadata,
backbones, prompts, budgets and code-based stochastic memory rules are substantive
reconstruction choices, not consequences of changing an output format alone.

Fake-client tests in `test_recagent_memory.py`, `test_recagent_scorer.py` and
`test_recagent_runner.py` cover numeric memory rules, duplicate identities,
reflection/forgetting, full/latest-20 initialization, two memory loads, malformed
responses, rollback/resume, cache invalidation, held-out-label invariance, worker
and group-order independence, shared binary/ranking artifacts and missing-output
coverage. These establish executable behavior, not predictive quality or parity
with unpublished original experimental settings.

Isolation is checked against a fingerprint of the **current pre-implementation
workspace**, including pre-existing uncommitted changes. Existing scorer/task/
evaluator/client implementations are unchanged; only public export and runner
registry additions touch existing Python files. Optional dependency metadata and
lock entries are additive, with all prior locked packages and versions retained.
Relevant existing-method tests are run alongside the new tests. The pre-existing
Agent4Rec runner test expecting no `errors: 0` field in taste-cache statistics is
reported separately and is not repaired by this change.

Verification on 2026-10-29: **73 new RecAgent tests pass**. The combined command
below reports **290 passed, one pre-existing failure** in
`test_agent4rec_yes_no_runner_writes_taste_manifest_and_cache` (unexpected
`errors: 0` in its expected dictionary). This same failure was recorded before
implementation. Ruff and `uv lock --check --offline` pass. Of 245 fingerprinted
pre-existing source/documentation/configuration files, 242 are byte-identical; the
three exceptions are additive scorer exports, runner registration and optional
dependency metadata. All pre-existing lockfile package records are unchanged
except the root project's added optional-dependency declaration. The manuscript
implementation-status update compiles with the built-in LaTeX compiler.

```bash
.venv/bin/python -m pytest -q \
  tests/test_recagent_memory.py tests/test_recagent_scorer.py tests/test_recagent_runner.py \
  tests/test_llmrank_scorer.py tests/test_llmrank_runner.py \
  tests/test_llmsrec_syn_scorer.py tests/test_llmsrec_syn_runner.py \
  tests/test_llm_yes_no_runner.py tests/test_llm_scorers.py \
  tests/test_agent4rec_yes_no_scorer.py tests/test_agent4rec_yes_no_runner.py \
  tests/test_ranking_evaluation.py
```
