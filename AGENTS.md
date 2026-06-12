# AGENTS.md

## Project

This repository is the clean codebase for the research project currently titled:

**Beyond Click Prediction: A Multi-Task Evaluation of LLM User Simulators for Recommendation**

The goal is to write a scientific paper about how useful (or not) LLM-based user simulators are for recommender-system evaluation. The project is mostly experimental, so claims must be backed by reproducible code, explicit setup definitions, and verifiable outputs.

## Research Framing

The paper studies user simulators for recommendation beyond a single click-prediction setup.

Use a broad definition of a simulator: any model that predicts a user response for explicitly provided user-item task rows can be treated as a user-response simulator in the offline setting. This includes LLM agents, LLM prompt-based predictors, classical recommender models used as scorers, tabular models, popularity baselines, and other response models. The important question is not whether something is called an agent or simulator, but which user-response scenarios it can model well.

Do not mix response prediction with recommendation policy. A scorer/simulator predicts how a user would respond to provided candidates. A recommender or policy decides which candidates to show.

The central claim is:

> Simulator quality is evaluation-setup-specific.

The working hypothesis is that LLMs may have useful language/world-knowledge priors for OOD or out-of-sample simulation, while standard ID-based collaborative recommender baselines usually have limited extrapolation capacity outside the observed user-item distribution. This must be tested, not assumed.

A model does not perform well or badly "on MovieLens" or "on Steam" in general. It performs well or badly under a specific target definition, split, candidate construction, negative sampling strategy, metadata visibility, prompt/model configuration, and metric.

Main paper scope:

- **Memorization tests:** check whether LLMs know recommendation datasets rather than infer user preferences.
- **In-distribution multi-target prediction:** test whether simulators predict held-out user outcomes such as interaction, positive preference, rating, playtime, or other intensity signals.
- **Pointwise vs ranking evaluation:** distinguish reproducing individual user outcomes from reproducing preferences over candidate sets.
- **Policy-ranking agreement:** test whether simulated responses rank recommender policies in the same order as real held-out data.
- **Offline-constructed OOD shifts:** test extrapolation across shifts built from historical logs, such as cold-start, temporal, semantic, feature-based, or domain-like splits.
- **Behavioral extrapolation:** test effects that are not directly captured by static logs, such as choice overload, framing, anchoring, social proof, repeated exposure, fatigue, novelty seeking, trust degradation, filter bubbles, and session exit.

A key distinction: many Agent4Rec, SimUSER, and AgentRecBench-style "alignment" evaluations are one-step offline prediction tasks, not full simulator loops. They take user history and candidate items, then ask a model to predict a response or ranking.

Full simulation loops are only necessary when the interaction trajectory matters: previous recommendations must affect later user state, later responses, or future policy behavior. Examples include fatigue, trust degradation, filter bubbles, session length, exit, return probability, and same-items-different-order experiments.

Therefore, do not assume that a full Agent4Rec-style loop is required. Start from the scientific question: if independent user-item or user-slate responses are enough, use offline evaluation; if history changes future outcomes, use trajectory simulation.

## Authority and Evidence

This is an active research project. Sources can be incomplete, outdated, or exploratory.

User instructions in chat are authoritative for task scope, priorities, naming, and implementation preferences. They are not automatically authoritative for factual claims, empirical results, dataset statistics, citations, or what the code currently does.

For implementation state and empirical claims, use this evidence order:

1. Local code, configs, and tests for what the implementation currently does.
2. Reproducible experiment outputs, run manifests, and output metadata for empirical claims, if they match the intended setup.
3. Papers, official documentation, or dataset documentation for external factual claims.
4. Local markdown files as research notes, plans, ideas, issue logs, and notebook-like thinking.

Markdown files are useful for understanding ideas and paper framing, but they may be outdated. Do not treat them as automatically correct implementation specifications.

If sources disagree about facts, results, experiment design, or implementation details, do not silently resolve the conflict. State the conflict and which source you used.

For factual or empirical claims, cross-check against papers, code, documentation, or actual experiment outputs when possible.

## Working Principles

- Prefer simple, accurate, readable, reproducible scientific code over over-engineered architecture.
- Do not invent metrics, results, citations, dataset statistics, or experimental conclusions.
- Be critical about leakage, memorization, unfair candidate sets, weak baselines, popularity artifacts, negative sampling artifacts, and target-definition drift.
- Keep pointwise prediction, candidate ranking, static policy evaluation, and trajectory simulation conceptually separate.
- Every experiment result should document dataset, split, target definition, candidate construction, seed, model/prompt version, metadata visibility, sample sizes, metrics, and known caveats.
- Treat exploratory notebooks and notes as sources of ideas to extract and simplify, not structures to copy blindly.

## Experiment Artifacts

`final_results.md` is the current registry of final results. Treat runs listed there as the result set intended for reporting, not as a scratch notebook index.

For every run listed in `final_results.md`, commit the compact provenance files:
- `manifest.json`
- the relevant `metrics*.json`

Do not list uncommitted, server-only, or locally missing runs as final results. Final results are defined by explicit user instructions, not inferred from notebooks or output directories.

Do not commit row-level or large artifacts. If a notebook depends on ignored row-level artifacts, mark it briefly: local parquet required.
