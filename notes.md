1. 
    For LLM we need to extend history by adding different targets. E.g. it may need rating for better understanding of user, to predict interaction/preference. We actually can do this for some classical models by providing avg. rating feature or something like this.
2. 
    Alignment Task. We ask LLM to score watch/no watch for each candidate groups. We may provide or not information how many likes it should put (like 1 always in our 1:m val). Check if Agent4Rec or SimUser do it? It makes it closer to ranking evaluation. If we want to apply Popularity (or maybe other metrics) for such problem, we need to find threshold on val set, than apply it on test set? For classic ML its simpler, as we can just use standard threshold. 

    **Answer:** For pointwise alignment, score-based methods require a validation-calibrated decision rule. Direct yes/no LLM protocols do not need threshold calibration, but prompt/parser choices must be fixed before test and should be developed on validation only. *We need to carefully choose the size of val set!*

3. Think about creating a separate base class for LLMScorers. `client`, `model`, `max_tokens`, `temperature`, `max_history_items` are their common attributes.

4. Current Agent4Rec-style candidate construction is `cap20`, not always `fixed20`. For `m=9` and `m=19`, candidate groups are effectively fixed at 20 items. For `m=1` and `m=3`, group size may be smaller when a user has too few held-out positives; for `m=2`, strict `1:2` ratio means the maximum group size is 18. Compare scorers within the same `m`; comparisons across different `m` also change candidate-group size and positive prevalence.

5.
    LLM simulator vs classic RS baselines is an asymmetric information regime by design. LLM sees one user's history and item metadata, but also has a pretrained world prior and possible memorized knowledge. Classic RS methods use train interactions from many users as their natural population-level signal. Therefore, limiting evaluation to 1000 users can mean two different questions:

    - **Simulator question:** train classic RS on full train, let LLM use per-user history, and evaluate all methods on the same deterministic 1000-user candidate subset. Here 1000 users is an evaluation budget, not a dataset filter.
    - **Agent4Rec/SimUSER-style reproduction:** sample 1000 eligible users before split and build the whole task only on them. This is closer to the "1000 agents" setup, but it intentionally limits classic RS training signal.

    We should not silently mix these protocols. If both are useful, name them explicitly, e.g. `full_train_eval1000` and `sampled_users_train_eval1000`.
