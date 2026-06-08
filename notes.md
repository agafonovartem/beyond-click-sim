1. 
    For LLM we need to extend history by adding different targets. E.g. it may need rating for better understanding of user, to predict interaction/preference. We actually can do this for some classical models by providing avg. rating feature or something like this.
2. 
    Alignment Task. We ask LLM to score watch/no watch for each candidate groups. We may provide or not information how many likes it should put (like 1 always in our 1:m val). Check if Agent4Rec or SimUser do it? It makes it closer to ranking evaluation. If we want to apply Popularity (or maybe other metrics) for such problem, we need to find threshold on val set, than apply it on test set? For classic ML its simpler, as we can just use standard threshold. 

    **Answer:** For pointwise alignment, score-based methods require a validation-calibrated decision rule. Direct yes/no LLM protocols do not need threshold calibration, but prompt/parser choices must be fixed before test and should be developed on validation only. *We need to carefully choose the size of val set!*

3. Think about creating a separate base class for LLMScorers. `client`, `model`, `max_tokens`, `temperature`, `max_history_items` are their common attributes.

4. Current Agent4Rec-style candidate construction is `cap20`, not always `fixed20`. For `m=9` and `m=19`, candidate groups are effectively fixed at 20 items. For `m=1` and `m=3`, group size may be smaller when a user has too few held-out positives; for `m=2`, strict `1:2` ratio means the maximum group size is 18. Compare scorers within the same `m`; comparisons across different `m` also change candidate-group size and positive prevalence.
