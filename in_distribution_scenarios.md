### TaskBuilder for in-distribution

Preprocessing after canonization should have 3 steps:
1. **Filtering.** I can filter any dataset by any rules I like. It should be simply `DatasetFilter` class with method `filter`. That's it. It does not care about target. For `DatasetFilter` it is just filtering condition.
2. **Splitters:**
    - RandomFractionSplitter: train/val/test by fractions, e.g. 70/10/20
    - StratifiedRandomSplitter: random split preserving label proportions.
    - GlobalTemporalSplitter: shared time cutoffs across users.
    - LeaveNOutSplitter: hold out fixed counts per user, e.g. val_n=1, test_n=1. [MAYBE LATER]
    - TemporalLeaveNOutSplitter: hold out last n interactions per user, only when timestamps are meaningful. [MAYBE LATER]
3. **CandidateSampler.**  

Examples:
1. *Split*: RandomFractionSplitter. *Target*: Interaction.
    - Filter: just want to ensure that the user has at least $n$ interactions
    - Split: $x \%$ on train, $y \%$ on test, $(100 - x - y)\%$ on val. 
    - Sampler: randomly sample negatives from non-interactions. It could be different strategies: $1:1$, $1:5$, $1:20$, full ranking, or zero-shot LLM suggestions. 
2. *Split*: GlobalTemporalSplitter. *Target*: Interaction.
    - Filter: want to do proper timesplit, to ensure that many users user has enough interactions before and after cutoff.
    - Split: by timestamp cutoff. 
    - Sampler: randomly sample negatives from non-interactions. It could be different strategies: $1:1$, $1:5$, $1:20$, full ranking, or zero-shot LLM suggestions. 
3. *Split*: StratifiedRandomSplitter. *Target*: Preference.
    - Filter: choose users with at least $x$ positives, $y$ negatives.
    - Split: train-test-val split for classification with stratification by preference. 
    - Sampler: no need for classification metrics. For ranking metrics we then want to combine test pos/neg in groups like $1:1$, $1:5$, $1:20$.
4. *Split*: GlobalTemporalSplitter. *Target*: Preference.
    - Filter: want to do proper timesplit, to ensure that many users user has enough interactions before and after cutoff.
    - Split: by timestamp cutoff + maybe check stratification. 
    - Sampler: no need for classification metrics. For ranking metrics we then want to combine test pos/neg in groups like $1:1$, $1:5$, $1:20$.
5. *Split*: RandomFractionSplitter. *Target*: Regression.
    - Filter: just want to ensure that the user has at least $n$ interactions
    - Split: $x \%$ on train, $y \%$ on test, $(100 - x - y)\%$ on val.  
    - Sampler: no need for regression.
6. *Split*: GlobalTemporalSplitter. *Target*: Regression.
    - Filter: want to do proper timesplit, to ensure that many users user has enough interactions before and after cutoff.
    - Split: by timestamp cutoff
    - Sampler: no need for regression.

### Problem Formulation for Multi-Target In-Distribution User Outcome Prediction
When we simulate users in recommender systems, or build recommender systems in general, we have many datasets and can define many tasks on them:
1. **Interaction:** will the user interact with this item?
2. **Positive preference:** will the user like it enough?
3. **Intensity regression:** how much will the user like or consume this item?

Note, that all of these problems are user-level.

The first two targets can be evaluated from two perspectives: pointwise outcome prediction, similar to alignment-style tables in Agent4Rec and SimUSER, and candidate ranking, similar to AgentRecBench-style recommendation evaluation.

From the model perspective, both settings reduce to scoring. The model assigns a numeric score to each user-item row. Then we can treat them in different ways according to evaluation protocol.

- Pointwise / Alignment / Classification (Tasks 1 and 2):
    - input: test set (candidates list)
    - model: calculates scores. They could be different: LLM just says yes/no, while classical ML methods may give probabilities, ratings, logits.
    - evaluation: convert scores into binary predictions (per candidate group if needed) using a threshold or another protocol, then compute accuracy, precision, recall, F1, AUC / PR-AUC, ...

    This setting asks whether the model can distinguish positive and negative user outcomes.

- Ranking / Recommendation (Tasks 1 and 2)
    - input: candidates list
    - model: assigns one score to each candidate row
    - evaluation: sort candidates by score within each candidate group, then compute HR@K, NDCG@K, MRR, and related ranking metrics.

    In this setting, the model does not directly output ranks. It outputs relevance scores; ranks are derived by the evaluator.

- Regression / Intensity Prediction (Task 3)
    - input: test set
    - model: calculates scores
    - evaluation: computes mse, mae

This leads to the core Stage-1 abstraction: every user simulator or response model is a `Scorer`. Conceptually, a scorer estimates `score_target(user, item, ...)`.

The score is not necessarily a calibrated probability. For binary tasks, it is a positive-class score. For ranking tasks, it is a relevance or utility score, and ranks are obtained by sorting scores inside candidate groups. For regression tasks, the score is the predicted numeric value. Therefore the minimal `Scorer` interface should have two methods: `fit()` and `score()`.

### Pointwise Decision Protocol
For pointwise alignment/classification, scorers may output arbitrary real-valued scores, so binary metrics require an explicit decision rule.

For score-based methods (Popularity, ItemKNN, MF/BPR, tabular ML), choose the threshold on validation and keep it fixed on test. Do not tune thresholds on test.

For direct LLM yes/no protocols, no threshold is needed: the prompt/parser already defines binary predictions. Prompt and parser choices should still be fixed before test evaluation.

For pointwise threshold selection, validation should use the same candidate construction as test.

Ranking is a separate protocol: sorting candidates within `candidate_group` and reporting HR@K/NDCG@K/MRR should not be silently mixed with pointwise alignment metrics.
