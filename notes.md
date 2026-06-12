1. 
    For LLM we need to extend history by adding different targets. E.g. it may need rating for better understanding of user, to predict interaction/preference. We actually can do this for some classical models by providing avg. rating feature or something like this.

    **Answer:** Mostly implemented for LLMs: `history_context_columns` add train-only feedback such as `rating` to user history, while val/test candidates keep these columns missing. Classical avg-rating features are not implemented yet. For LLM numeric regression, out-of-range outputs should be treated as parse failures rather than clamped into the target range; e.g. MovieLens rating predictions must parse as a bare integer in `{1, 2, 3, 4, 5}`.
2. 
    Alignment Task. We ask LLM to score watch/no watch for each candidate groups. We may provide or not information how many likes it should put (like 1 always in our 1:m val). Check if Agent4Rec or SimUser do it? It makes it closer to ranking evaluation. If we want to apply Popularity (or maybe other metrics) for such problem, we need to find threshold on val set, than apply it on test set? For classic ML its simpler, as we can just use standard threshold. 

    **Answer:** For pointwise alignment, score-based methods require a validation-calibrated decision rule. Direct yes/no LLM protocols do not need threshold calibration, but prompt/parser choices must be fixed before test and should be developed on validation only. *We need to carefully choose the size of val set!*

3.
    Think about creating a separate base class for LLMScorers. `client`, `model`, `max_tokens`, `temperature`, `max_history_items` are their common attributes.

4.
    Current Agent4Rec-style candidate construction is `cap20`, not always `fixed20`. For `m=9` and `m=19`, candidate groups are effectively fixed at 20 items. For `m=1` and `m=3`, group size may be smaller when a user has too few held-out positives; for `m=2`, strict `1:2` ratio means the maximum group size is 18. Compare scorers within the same `m`; comparisons across different `m` also change candidate-group size and positive prevalence.

    **Answer:** This is the current default behavior of `CappedUserInteractionCandidateSampler`, so no implementation change is needed. Keep it as an evaluation caveat and report candidate-group size/positive prevalence in manifests/results.

5.
    LLM simulator vs classic RS baselines is an asymmetric information regime by design. LLM sees one user's history and item metadata, but also has a pretrained world prior and possible memorized knowledge. Classic RS methods use train interactions from many users as their natural population-level signal. Therefore, limiting evaluation to 1000 users can mean two different questions:

    - **Simulator question:** train classic RS on full train, let LLM use per-user history, and evaluate all methods on the same deterministic 1000-user candidate subset. Here 1000 users is an evaluation budget that should be implemented in `CandidateSampler`, not a dataset filter.
    - **Agent4Rec/SimUSER-style reproduction:** sample 1000 eligible users before split and build the whole task only on them. This would be a `DatasetFilter` before splitting. It is closer to the "1000 agents" setup, but it intentionally limits classic RS training signal.

    We should not silently mix these protocols. If both are useful, name them explicitly, e.g. `full_train_eval1000` and `sampled_users_train_eval1000`.
6.
    **Validation-budget question.** Current `eval_users1000` implementation applies the same user budget to both validation and test candidate construction because the sampler is called separately for `val` and `test`. This is fine for a quick sanity check, but it mixes two decisions: test evaluation budget (needed for LLM cost / 1000-agent protocol) and validation budget (used only for threshold or hyperparameter selection). For score-based baselines, full validation with `test_eval1000` may give a more stable threshold without leaking test information. For hard yes/no LLM scoring, validation may be unnecessary unless we select prompt/config variants. Consider explicit protocol parameters such as `max_val_users=None` and `max_test_users=1000`.

7.
    **Idea: select LLM scorer config on validation.** Treat LLM scorer choices as hyperparameters tuned on the validation split, with the test split evaluated only once at the end: `max_history_items` (history length), prompt wording/template, history composition (positive-only vs positive+negative), and metadata visibility (titles/genres/popularity/avg rating). "Tuning" here means config/prompt selection — optionally prompt or prefix tuning, or an automated prompt-search loop — not weight training. This matters once we compare prompt/config variants; otherwise we risk selecting on test. The yes/no scorer needs no threshold, so without such selection no val set is required.

8.
    **Idea: use `logprob(yes)` as a continuous LLM score.** The yes/no scorer returns hard 0/1, which already supports pointwise metrics and (coarse, tie-heavy) ranking. Reading the probability of the `yes` token from `top_logprobs` (supported by vLLM and the OpenAI API) would give a continuous score, enabling finer-grained ranking (NDCG/HR), threshold-free metrics (AUC/PR-AUC), and cleaner comparability with score-based baselines (popularity, MF, ...). Caveat: not all endpoints expose logprobs, and the values need not be calibrated.

9.
    **Ranking output protocol: pointwise-score ranking vs direct re-ranking.** Our ranking
    metrics (HR@k/NDCG@k) sort per-row scores within each `candidate_group`. For score-based
    methods (popularity) this is a genuine ordering. For the yes/no LLM the score is binary, so
    the ranking is tie-dominated and is a *coarse re-summary* of the same per-group yes/no
    decisions: the expected (tie-averaged) HR@k/NDCG@k of a group depends only on (group size,
    #positives, #predicted-yes, #true-positives-in-yes) — the same confusion counts behind
    precision/recall. Always report `groups_with_score_ties_fraction` next to the ranking headline.

    This is a ranking diagnostic derived from Agent4Rec-style binary per-item decisions; it is
    not a direct-ranking prompt, and the original Agent4Rec paper's headline is distribution
    alignment rather than NDCG (arXiv 2310.10108).
    **AgentRecBench instead uses a direct re-ranking protocol** (arXiv 2505.19623): the agent's
    terminal `CandidateRank` action emits a full ordering over the 20 candidates (1 positive : 19
    negatives), so there are no ties and HR@{1,3,5} measures a true ranking. **We do not currently
    cover this direct-re-ranking case.** For a genuine LLM ranking signal beyond a transform of the
    classification, the options are `logprob(yes)` as a continuous score (item 8) or an
    AgentRecBench-style direct "rank these candidates" prompt.

10.
    **LLM metric stability check.** Check whether LLM metrics are stable enough on smaller user samples. If bootstrap/subsampling shows acceptable uncertainty, 100 users may be enough for intermediate results instead of 1000 users.
