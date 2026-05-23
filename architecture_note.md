# Architecture for Codex: Offline Response Prediction First

Working title: **Beyond Click Prediction: A Multi-Task Evaluation of LLM User Simulators for Recommendation**

This is a short implementation note for Codex. The goal is to explain the project direction, not to specify every interface.

We build the project in this order:
1. First implement **offline LLM predicts interaction**.
2. Build the reusable layer: `DatasetAdapter` + `ScenarioBuilder` + `TaskBuilder` + `CandidateSampler` + `ResponsePredictor`.
3. Reuse this layer later for static recommender policy ranking and, if needed, full trajectory simulation: `policy ranks -> user responds -> state updates -> repeat`.

The first paper should not start by building a full Agent4Rec-like simulator. Most existing alignment tables in Agent4Rec, SimUSER, and AgentRecBench are one-step candidate evaluation tasks, not true trajectory simulations. Therefore the first implementation target is an offline response prediction benchmark.

The code will eventually be public. We want clean research code: simple, readable, reproducible.

---

## 0. Core distinction

### 0.1 Offline response prediction

First-stage scope:

```text
user history + item/candidate set -> predicted response or ranking
```

Examples: will the user interact with this item; will the user like it enough; what rating/playtime/dwell time will the user produce; which candidate items will the user choose.

This does **not** require a simulator loop. It only requires a one-step response model.

### 0.2 Trajectory simulation

Later-stage extension:

```text
policy recommends slate_t -> user simulator responds -> state updater modifies user state -> policy recommends slate_{t+1} -> ...
```

This is needed only for path-dependent outcomes: fatigue, saturation, trust degradation, session exit, return probability, filter bubbles, long-term policy effects.

Do not implement this first. Design offline components so they can be reused later.

---

## 1. Project stages

### Stage 1: Offline response prediction benchmark

Implement: `DatasetAdapter`, `ScenarioBuilder`, `TaskBuilder`, `CandidateSampler`, `ResponsePredictor`, `OfflineEvaluator`.

Main tasks: interaction prediction, positive preference prediction, intensity regression.

### Stage 2: Static policy ranking agreement

Add later: `Policy`, `StaticPolicyEvaluator`, `PolicyRankingAgreementMetrics`.

No simulator loop yet. Process: policy recommends fixed top-k/slates; response predictor estimates outcomes; aggregate value per policy; compare simulated policy ranking with real held-out policy ranking.

### Stage 3: Optional trajectory simulation

Add only if needed: `RolloutEnvironment`, `UserState`, `StateUpdater`, `SequentialUserSimulator`, `ScenarioTransforms`, `TrajectoryEvaluator`.

---

## 2. Implementation style

Separate code into files and folders, but do not over-engineer. The first useful version should be small and easy to change.

Avoid starting with a large framework, complex inheritance, a full Agent4Rec clone, multi-agent environment, memory/persona system, or trajectory simulator.

---

## 3. Canonical data layer

### Goal

Convert each dataset into one common representation so that the same scenario builders, task builders, samplers, predictors, and evaluators work across datasets.

Start with three logical entities: users, items, and interactions. They are rows in validated tables, not Python objects for every row.

The canonical data layer should answer: who are the users; what are the items; what interactions happened; what raw signals and metadata are available.

Materialize canonical datasets on disk:

```text
data/canonical/{dataset}/{version}/
  manifest.json
  users.parquet
  items.parquet
  interactions.parquet
```

`CanonicalDataset` should be a small descriptor over these files plus schema/version metadata. Use dataclasses for descriptors, manifests, configs, and small task instances, not for every user/item/interaction row.

### DatasetAdapter responsibility

Each adapter should convert one raw/source dataset into canonical tables with minimal cleaning and a manifest.

The adapter should not build splits, candidates, targets, train models, or run metrics.

---

## 4. Scenario and split layer

A scenario defines the evaluation setting: eligibility, support/held-out visibility, train/validation/test split policy, user/domain partitions, and seed. The split is part of the scenario.

A scenario may use event filters to define eligible users/items/interactions or domains. However, the final prediction label belongs to TaskBuilder, not ScenarioBuilder.

---

## 5. Task construction layer

A dataset is not a task. The same dataset can produce multiple tasks: interaction prediction, positive preference prediction, intensity regression, candidate ranking, policy evaluation, OOD split evaluation.

`TaskBuilder` defines the target and label for each task instance within a scenario. General behavior: get user context; get held-out interactions; define target label; ask `CandidateSampler` to produce candidates; return task instances for predictors/evaluators.

### Core tasks

**Task I: interaction prediction.** Question: will the user interact with this item? This is closest to Agent4Rec alignment and AgentRecBench-style candidate evaluation.

**Task II: positive preference prediction.** Question: will the user like it enough? This is different from interaction. Possible labels: high rating, long playtime, like/share, long watch time.

**Task III: intensity regression.** Question: how much will the user like or consume this item? Examples: rating, playtime, dwell time, watch time.

`TaskBuilder` must be model-agnostic. It should not care whether the downstream model is LLM, CatBoost, BPR, LightGCN, SASRec, popularity, hybrid, or a future simulator.

---

## 6. Candidate sampling layer

Evaluation depends heavily on candidate construction. A model does not perform well or badly on a dataset in isolation. It performs well or badly under a specific target, candidate set, and negative sampling strategy.

Candidate sampling modes must be explicit, configurable, deterministic by seed, and logged.

Possible modes: `1:1`, `1:5`, `1:20`, random negatives, popularity-matched negatives, category/genre-matched negatives, hard negatives, observed-only candidates, observed + unobserved candidates, full-catalog ranking.

Start with a small subset and extend later.

Leakage rules: exclude train interactions when needed; exclude validation/test leakage; exclude future interactions for temporal splits; exclude target user's held-out positives from negatives; log negative sampling strategy in metadata.

`CandidateSampler` is universal. In Stage 1, it creates candidate sets for offline response prediction. In Stage 2, it can create candidate pools for static policy ranking. In Stage 3, it may support candidate pools for rollout simulation. Do not implement rollout-specific behavior now.

---

## 7. Response prediction layer

This is the main modeling layer for Stage 1.

A `ResponsePredictor` takes a task and returns scores, labels, ratings, or rankings for candidate items.

Conceptually:

```text
user history + candidate item(s) -> predicted response
```

The response predictor can be random baseline, popularity baseline, classical recommender model, supervised ML model, LLM prompt, Agent4Rec-style prompt, or hybrid model.

This is why we do not start with a full simulator. Most current alignment experiments can be tested through a one-step predictor.

First predictors: random baseline, popularity baseline, simple similarity baseline, LLM response predictor.

Later predictors: BPR/MF, LightGCN, SASRec, CatBoost/XGBoost, text embedding baselines, hybrid models.

### LLM predictor

The LLM predictor is just another `ResponsePredictor`. It receives user history, candidate item(s), task instruction, and available metadata. It outputs something comparable to other models: candidate scores, candidate ranking, yes/no labels, rating or intensity prediction.

Prompt variants should be configurable: history length, positive-only history vs positive+negative history, with/without item summaries, with/without popularity/average rating, with/without dataset name, with/without user/item ids.

Important distinction: `ResponsePredictor` answers how the user would respond; `Policy` answers what to show. Do not mix these roles in the code.

Candidate-free LLM recommenders from the prototype are closer to `Policy` or full-catalog ranking baselines: they produce a top-k list without seeing an explicit candidate set, then evaluation maps generated titles back to catalog items and logs match diagnostics.

---

## 8. Offline evaluation layer

The first evaluator checks whether predictors recover held-out real user outcomes.

For interaction / positive preference: accuracy, precision, recall, F1, AUC, HR@K, NDCG@K, MRR, calibration if probabilities are available.

For intensity regression: MAE, RMSE, correlation, ranking metrics after sorting by predicted intensity.

Every result should log dataset, task type, target definition, candidate sampling strategy, negative sampling strategy, predictor name, LLM prompt/config if applicable, random seed, metrics.

This logging is essential because the paper argues that results depend on target, candidate construction, and negative sampling.

---

## 9. Static policy ranking extension

Stage 2 question:

```text
Can a response predictor rank recommender policies in the same order as real held-out data?
```

Process: train or define several recommender policies; for each user, each policy recommends fixed top-k items/slates; response predictor estimates outcomes for those recommendations; aggregate predicted value for each policy; compare simulated policy ranking with real held-out policy ranking.

This still does not require a simulator loop.

Key distinction: static policy ranking is many independent user-item/user-slate evaluations; trajectory-aware policy ranking means recommendations change user state and future responses.

Start with static policy ranking. Add trajectory-aware ranking only if useful.

---

## 10. Trajectory simulation extension

Stage 3. Do not implement first.

Needed only when previous recommendations affect future behavior.

Process:

```text
policy ranks candidates / chooses slate -> user simulator responds -> state updater updates user state -> policy observes updated state -> repeat
```

Use for questions like: does bad early recommendation quality reduce later trust; does repeated exposure cause fatigue or saturation; does a policy create a filter bubble over time; does item order change session length or exit probability; does a user return after a bad session.

Future experiment: **same items, different order**. Take the same set of items for a user and change only order or grouping: best-first, worst-first, random, category-blocked, diverse interleaving, popularity-first, novelty-first. If static evaluation says these are equivalent but rollout evaluation produces different outcomes, then trajectory matters.

---

## 11. Minimal first milestone

First milestone: one dataset adapter; one task type, interaction prediction; two candidate samplers, `1:1` and `1:20`; two baselines, random and popularity; one LLM predictor; one offline evaluator; one reproducible experiment script.

Then extend: more datasets, more targets, more candidate samplers, more baselines, memorization controls, OOD splits, static policy ranking, trajectory simulation.

---

## 12. Research principle

The implementation should make the paper argument easy to test:

```text
Most existing LLM user-simulator alignment evaluations are one-step candidate evaluation tasks.
They do not require a full simulator loop.
Therefore we first evaluate LLMs as offline response predictors across many targets and candidate settings.
Only after that do we ask when trajectory simulation is necessary.
```

Short version:

```text
Offline first. Trajectory later. Reuse the same data/task/candidate layers for both.
```
