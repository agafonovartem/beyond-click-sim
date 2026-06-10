### TaskBuilder for In-Distribution Tasks

Preprocessing after canonization should have 3 steps:
1. **Filtering.** I can filter any dataset by any rules I like. It should be simply `DatasetFilter` class with method `filter`. That's it. It does not care about target. For `DatasetFilter` it is just filtering condition.
2. **Splitters:** A splitter creates train/validation/test interaction splits. Examples:
    - RandomFractionSplitter: train/val/test by fractions, e.g. 70/10/20
    - StratifiedRandomSplitter: random split preserving label proportions.
    - GlobalTemporalSplitter: shared time cutoffs across users.
    - leave-n-out variants may be added later.
3. **CandidateSampler.**  It constructs explicit candidate rows when the task needs them. This is independent of whether we later compute pointwise metrics or ranking metrics. The same candidate table can be flattened for binary classification metrics or grouped by candidate group for ranking metrics.
While canonicalization creates standard target columns, candidate samplers may add rows that are not canonical interactions such as sampled non-interactions; in that case the sampler is responsible for assigning `target` and documenting the assumption.

Task families:

1. **Interaction task.**
    Question: will the user interact with this item?
    - Filter: usually ensure each user has enough observed interactions before splitting.
    - Split: random or temporal split over observed interactions.
    - Sampler: required for explicit negative examples. We sample unobserved items to provide negative candidates for positive observed interactions. For sampled non-interactions, `target=0` is part of the interaction-task assumption.

2. **Preference task.**
    Question: will the user like this interacted item enough?
    - Filter: usually ensure each user has enough observed positives and observed negatives under the selected preference target.
    - Split: random, stratified random, or temporal split over observed interactions.
    - Sampler: optional for observed-only pointwise classification, but required for candidate-set evaluation. When sampling candidates, we should usually sample observed unliked items to provide negative candidates for positive liked items. This keeps preference labels grounded in observed feedback.

3. **Regression task.**
    Question: what numeric intensity will the user produce?
    - Filter: usually ensure enough observed interactions with valid numeric target values.
    - Split: random or temporal split over observed interactions.
    - Sampler: no negative sampling is needed. For standard RMSE/MAE, evaluate on observed held-out rows with known numeric labels. In the SimUSER paper, rating prediction uses an 80/10/10 time-based split without stratification or fixed-size candidate sets.  

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
    - evaluation: convert scores into binary predictions (per candidate group if needed) using a decision rule, such as a validation-selected threshold, then compute accuracy, precision, recall, F1, AUC / PR-AUC, ...

    This setting asks whether the model can distinguish positive and negative user outcomes.

- Ranking / Recommendation (Tasks 1 and 2)
    - input: candidates list
    - model: assigns one score to each candidate row
    - evaluation: sort candidates by score within each candidate group, then compute HR@K, NDCG@K, MRR, and related ranking metrics.

    In this setting, the model does not directly output ranks. It outputs relevance scores; ranks are derived by the evaluation protocol.

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

For candidate-grouped pointwise tasks, aggregate the headline metric as
`macro_by_user_group_mean`: compute the metric inside each candidate group, average groups
within the same user, then average users equally. Keep `macro_by_group` and `micro` as
diagnostics, not as the headline. Threshold selection for score-based methods should use
the same user-group aggregation as the reported test metric.

Threshold and hyperparameter selection belongs to the experiment loop, not to metric functions.

Ranking is a separate protocol: sorting candidates within `candidate_group` and reporting HR@K/NDCG@K/MRR should not be silently mixed with pointwise alignment metrics.
