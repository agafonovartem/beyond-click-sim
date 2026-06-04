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