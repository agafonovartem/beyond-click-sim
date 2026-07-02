# In Distribution

Snapshot of local in-distribution final-result artifacts. Metrics are copied from compact JSON provenance under `outputs/in_distribution/`; this is a run registry, not a paper table.

Notes:
- Interaction prediction uses the reduced `eval_users1000_cg5` protocol: seeds 0-2, 1000 validation/test users per split, and up to 5 candidate groups per selected user.
- Regression prediction still uses the existing ML-1M `eval_users1000` rating protocol below.
- For every listed run, the compact `manifest.json` plus relevant `metrics*.json` files should be tracked; row-level `predictions.parquet` files are local artifacts and should stay untracked.
- Paper-table notebooks that still reference the older uncapped interaction protocol need refresh before being used for reporting.

## Regression Prediction

### ML-1M rating eval_1000 users

Protocol:
- Dataset: `ml-1m` canonical `v1`.
- Target: `target_rating`; `rating` is available as train history context and hidden on val/test rows.
- Split: `RandomFractionSplitter(0.7, 0.1, 0.2, group_column="user_id")`, seeds 0-4.
- Filter: `MinUserInteractionsFilter(10)`.
- Evaluation budget: post-split `PostSplitUserSampler(n_users=1000, seed=seed)`.
- Candidate construction: none; no negatives, no `candidate_group`, no `sampled`.
- Methods: `mode_regressor` fits the most frequent train rating and predicts a valid discrete rating; `mean_regressor` fits the continuous train target mean and is retained as a MAE/RMSE diagnostic. `user_mean_regressor` and `user_mode_regressor` use the same per-user last-20 train-history window in input order that is shown to `LLMRegressor`. `item_mean_regressor` and `item_mode_regressor` fit train-target statistics per item and use the global train mean/mode for cold items. Item-stats LLM variants additionally expose train-split-only item rating mean/count for history and candidate movies.
- Main metric: `test.macro_by_user_mean.mae`; micro MAE/RMSE are retained as secondary diagnostics.
- Scope: 6040 filtered/train users and 3883 items for all seeds; test users are capped to 1000 after splitting.

#### LLMRegressor

| seed | train rows | test rows | test users | llm errors | scored/requested | macro MAE | macro RMSE | micro MAE | micro RMSE | run |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 694335 | 34520 | 1000 | 0 | 34520/34520 | 0.7995 | 1.0804 | 0.8057 | 1.1167 | `20260615T103530Z_ml-1m_rating_eval_users1000_seed0_llm_regressor_vllm_llama33_70b_full` |

#### LLMRegressor with item stats

| seed | train rows | test rows | test users | llm errors | scored/requested | macro MAE | macro RMSE | micro MAE | micro RMSE | run |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 694335 | 34520 | 1000 | 0 | 34520/34520 | 0.7523 | 1.0277 | 0.7492 | 1.0600 | `20260616T141222Z_ml-1m_rating_item_stats_eval_users1000_seed0_llm_regressor_vllm_llama33_70b_with_item_stats_full` |

#### ModeRegressor

| seed | train rows | val rows | test rows | test users | mode | macro MAE | macro RMSE | micro MAE | micro RMSE | run |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 694335 | 17629 | 34520 | 1000 | 4.0000 | 0.8364 | 1.0932 | 0.8468 | 1.1665 | `20260612T152221Z_ml-1m_rating_eval_users1000_seed0_mode_regressor` |

#### ItemMeanRegressor

| seed | train rows | val rows | test rows | test users | cold test rows | macro MAE | macro RMSE | micro MAE | micro RMSE | run |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 694335 | 17629 | 34520 | 1000 | 5 | 0.7912 | 0.9560 | 0.7755 | 0.9710 | `20260618T084646Z_ml-1m_rating_eval_users1000_seed0_item_mean_regressor` |
| 1 | 694335 | 16574 | 32425 | 1000 | 5 | 0.7782 | 0.9397 | 0.7852 | 0.9826 | `20260618T084659Z_ml-1m_rating_eval_users1000_seed1_item_mean_regressor` |
| 2 | 694335 | 16731 | 32738 | 1000 | 5 | 0.7825 | 0.9429 | 0.7807 | 0.9783 | `20260618T084712Z_ml-1m_rating_eval_users1000_seed2_item_mean_regressor` |
| 3 | 694335 | 17388 | 34008 | 1000 | 7 | 0.7905 | 0.9543 | 0.7730 | 0.9692 | `20260618T084725Z_ml-1m_rating_eval_users1000_seed3_item_mean_regressor` |
| 4 | 694335 | 17564 | 34407 | 1000 | 8 | 0.7982 | 0.9642 | 0.7920 | 0.9925 | `20260618T084738Z_ml-1m_rating_eval_users1000_seed4_item_mean_regressor` |

Seed average over seeds 0-4:

| metric | mean | std |
|---|---:|---:|
| test.macro_by_user_mean.mae | 0.7881 | 0.0079 |
| test.macro_by_user_mean.rmse | 0.9514 | 0.0100 |
| test.micro.mae | 0.7813 | 0.0076 |
| test.micro.rmse | 0.9787 | 0.0094 |

#### ItemModeRegressor

| seed | train rows | val rows | test rows | test users | cold test rows | macro MAE | macro RMSE | micro MAE | micro RMSE | run |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 694335 | 17629 | 34520 | 1000 | 5 | 0.7782 | 1.0619 | 0.7644 | 1.0850 | `20260618T084647Z_ml-1m_rating_eval_users1000_seed0_item_mode_regressor` |
| 1 | 694335 | 16574 | 32425 | 1000 | 5 | 0.7621 | 1.0427 | 0.7729 | 1.0962 | `20260618T084700Z_ml-1m_rating_eval_users1000_seed1_item_mode_regressor` |
| 2 | 694335 | 16731 | 32738 | 1000 | 5 | 0.7630 | 1.0432 | 0.7725 | 1.0953 | `20260618T084713Z_ml-1m_rating_eval_users1000_seed2_item_mode_regressor` |
| 3 | 694335 | 17388 | 34008 | 1000 | 7 | 0.7716 | 1.0592 | 0.7619 | 1.0862 | `20260618T084726Z_ml-1m_rating_eval_users1000_seed3_item_mode_regressor` |
| 4 | 694335 | 17564 | 34407 | 1000 | 8 | 0.7797 | 1.0626 | 0.7790 | 1.1050 | `20260618T084739Z_ml-1m_rating_eval_users1000_seed4_item_mode_regressor` |

Seed average over seeds 0-4:

| metric | mean | std |
|---|---:|---:|
| test.macro_by_user_mean.mae | 0.7709 | 0.0082 |
| test.macro_by_user_mean.rmse | 1.0539 | 0.0101 |
| test.micro.mae | 0.7701 | 0.0069 |
| test.micro.rmse | 1.0935 | 0.0082 |

#### UserMeanRegressor

| seed | train rows | val rows | test rows | test users | max history | macro MAE | macro RMSE | micro MAE | micro RMSE | run |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 694335 | 17629 | 34520 | 1000 | 20 | 0.8418 | 1.0185 | 0.8375 | 1.0480 | `20260618T072749Z_ml-1m_rating_eval_users1000_seed0_user_mean_regressor` |
| 1 | 694335 | 16574 | 32425 | 1000 | 20 | 0.8429 | 1.0202 | 0.8560 | 1.0684 | `20260618T072804Z_ml-1m_rating_eval_users1000_seed1_user_mean_regressor` |
| 2 | 694335 | 16731 | 32738 | 1000 | 20 | 0.8371 | 1.0178 | 0.8370 | 1.0536 | `20260618T072817Z_ml-1m_rating_eval_users1000_seed2_user_mean_regressor` |
| 3 | 694335 | 17388 | 34008 | 1000 | 20 | 0.8425 | 1.0203 | 0.8377 | 1.0481 | `20260618T072832Z_ml-1m_rating_eval_users1000_seed3_user_mean_regressor` |
| 4 | 694335 | 17564 | 34407 | 1000 | 20 | 0.8531 | 1.0337 | 0.8573 | 1.0721 | `20260618T072846Z_ml-1m_rating_eval_users1000_seed4_user_mean_regressor` |

Seed average over seeds 0-4:

| metric | mean | std |
|---|---:|---:|
| test.macro_by_user_mean.mae | 0.8435 | 0.0058 |
| test.macro_by_user_mean.rmse | 1.0221 | 0.0066 |
| test.micro.mae | 0.8451 | 0.0106 |
| test.micro.rmse | 1.0580 | 0.0114 |

#### UserModeRegressor

| seed | train rows | val rows | test rows | test users | max history | macro MAE | macro RMSE | micro MAE | micro RMSE | run |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 694335 | 17629 | 34520 | 1000 | 20 | 0.8947 | 1.1894 | 0.8843 | 1.2388 | `20260618T072751Z_ml-1m_rating_eval_users1000_seed0_user_mode_regressor` |
| 1 | 694335 | 16574 | 32425 | 1000 | 20 | 0.8954 | 1.1901 | 0.9080 | 1.2665 | `20260618T072805Z_ml-1m_rating_eval_users1000_seed1_user_mode_regressor` |
| 2 | 694335 | 16731 | 32738 | 1000 | 20 | 0.8933 | 1.1943 | 0.8929 | 1.2615 | `20260618T072819Z_ml-1m_rating_eval_users1000_seed2_user_mode_regressor` |
| 3 | 694335 | 17388 | 34008 | 1000 | 20 | 0.8805 | 1.1812 | 0.8805 | 1.2498 | `20260618T072833Z_ml-1m_rating_eval_users1000_seed3_user_mode_regressor` |
| 4 | 694335 | 17564 | 34407 | 1000 | 20 | 0.9100 | 1.2080 | 0.9141 | 1.2749 | `20260618T072848Z_ml-1m_rating_eval_users1000_seed4_user_mode_regressor` |

Seed average over seeds 0-4:

| metric | mean | std |
|---|---:|---:|
| test.macro_by_user_mean.mae | 0.8948 | 0.0105 |
| test.macro_by_user_mean.rmse | 1.1926 | 0.0098 |
| test.micro.mae | 0.8960 | 0.0147 |
| test.micro.rmse | 1.2583 | 0.0142 |

#### MeanRegressor

| seed | train rows | val rows | test rows | test users | train mean | macro MAE | macro RMSE | micro MAE | micro RMSE | run |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 694335 | 17629 | 34520 | 1000 | 3.5812 | 0.9352 | 1.0809 | 0.9214 | 1.1009 | `20260612T121634Z_ml-1m_rating_eval_users1000_seed0_mean_regressor` |
| 1 | 694335 | 16574 | 32425 | 1000 | 3.5820 | 0.9330 | 1.0769 | 0.9352 | 1.1194 | `20260612T121644Z_ml-1m_rating_eval_users1000_seed1_mean_regressor` |
| 2 | 694335 | 16731 | 32738 | 1000 | 3.5795 | 0.9387 | 1.0832 | 0.9277 | 1.1107 | `20260612T121654Z_ml-1m_rating_eval_users1000_seed2_mean_regressor` |
| 3 | 694335 | 17388 | 34008 | 1000 | 3.5808 | 0.9394 | 1.0849 | 0.9246 | 1.1069 | `20260612T121705Z_ml-1m_rating_eval_users1000_seed3_mean_regressor` |
| 4 | 694335 | 17564 | 34407 | 1000 | 3.5804 | 0.9411 | 1.0860 | 0.9383 | 1.1219 | `20260612T121716Z_ml-1m_rating_eval_users1000_seed4_mean_regressor` |

Seed average over seeds 0-4:

| metric | mean | std |
|---|---:|---:|
| test.macro_by_user_mean.mae | 0.9375 | 0.0033 |
| test.macro_by_user_mean.rmse | 1.0824 | 0.0036 |
| test.micro.mae | 0.9294 | 0.0071 |
| test.micro.rmse | 1.1120 | 0.0087 |

## Interaction Prediction

Protocol:

- Dataset: `ml-1m` and `steam` canonical `v1`.

- Target: `target_interact`; train history may include dataset-native context columns, but the popularity baselines use only train interaction targets grouped by `item_id`.

- Split: `RandomFractionSplitter(0.7, 0.1, 0.2, group_column="user_id")`, seeds 0-2.

- Filter: `MinUserInteractionsFilter(10)`.

- Evaluation budget: post-split `CappedUserInteractionCandidateSampler(max_eval_users=1000, max_candidate_groups_per_user=5, total_items=20, seed=seed)`.

- Candidate construction: for ratio `m`, each candidate group uses `k = floor(20 / (m + 1))` held-out positives plus `k * m` sampled non-interactions when enough positives are available; `cg5` caps candidate groups per selected user, not items or positives directly.

- Methods currently final under this reduced protocol: `popularity_f1_threshold` and `popularity_ranking`. LLM/Agent4Rec interaction results from the older uncapped `eval_users1000` protocol are intentionally not listed here; current-protocol LLM runs are pending.

- Main pointwise metric: `test.macro_by_user_group_mean.f1`; threshold is selected on validation by `macro_by_user_group_mean_f1`.

- Main ranking metric: `test.macro_by_user_group_mean.ndcg@5`; HR@1 and score-tie fraction are retained as diagnostics.

- Scope: train uses all filtered users; validation/test evaluate deterministic 1000-user subsets for each split. Candidate counts differ by dataset, ratio, and seed because `cap20` is not fixed-20 for every ratio.

#### Pointwise F1 seed averages

| dataset | m | seeds | mean | std |
|---|---:|---:|---:|---:|
| ml-1m | 1 | 0,1,2 | 0.7914 | 0.0034 |
| ml-1m | 2 | 0,1,2 | 0.7009 | 0.0032 |
| ml-1m | 3 | 0,1,2 | 0.6360 | 0.0059 |
| ml-1m | 9 | 0,1,2 | 0.4574 | 0.0067 |
| ml-1m | 19 | 0,1,2 | 0.3286 | 0.0028 |
| steam | 1 | 0,1,2 | 0.8931 | 0.0025 |
| steam | 2 | 0,1,2 | 0.8445 | 0.0036 |
| steam | 3 | 0,1,2 | 0.8074 | 0.0020 |
| steam | 9 | 0,1,2 | 0.6898 | 0.0022 |
| steam | 19 | 0,1,2 | 0.5796 | 0.0077 |

#### Ranking NDCG@5 seed averages

| dataset | m | seeds | mean | std |
|---|---:|---:|---:|---:|
| ml-1m | 1 | 0,1,2 | 0.8871 | 0.0020 |
| ml-1m | 2 | 0,1,2 | 0.7819 | 0.0036 |
| ml-1m | 3 | 0,1,2 | 0.7038 | 0.0059 |
| ml-1m | 9 | 0,1,2 | 0.6544 | 0.0095 |
| ml-1m | 19 | 0,1,2 | 0.5863 | 0.0119 |
| steam | 1 | 0,1,2 | 0.9698 | 0.0025 |
| steam | 2 | 0,1,2 | 0.9272 | 0.0023 |
| steam | 3 | 0,1,2 | 0.8863 | 0.0014 |
| steam | 9 | 0,1,2 | 0.8810 | 0.0016 |
| steam | 19 | 0,1,2 | 0.8331 | 0.0009 |

### ML-1M eval_users1000_cg5

#### ml-1m popularity_f1_threshold

| m | seed | test groups | test rows | positives/group | F1 | micro F1 | threshold | run |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0 | 2867 | 50786 | 8.8570 | 0.7911 | 0.7930 | 153.0000 | `20260702T122721Z_ml-1m_cap20_eval_users1000_cg5_m1_seed0_popularity_f1_threshold` |
| 1 | 1 | 2733 | 47718 | 8.7300 | 0.7950 | 0.7935 | 139.0000 | `20260702T122734Z_ml-1m_cap20_eval_users1000_cg5_m1_seed1_popularity_f1_threshold` |
| 1 | 2 | 2763 | 48500 | 8.7767 | 0.7882 | 0.7854 | 97.0000 | `20260702T122747Z_ml-1m_cap20_eval_users1000_cg5_m1_seed2_popularity_f1_threshold` |
| 2 | 0 | 3528 | 58893 | 5.5643 | 0.7036 | 0.7043 | 197.0000 | `20260702T122800Z_ml-1m_cap20_eval_users1000_cg5_m2_seed0_popularity_f1_threshold` |
| 2 | 1 | 3386 | 56217 | 5.5343 | 0.7018 | 0.7009 | 176.0000 | `20260702T122814Z_ml-1m_cap20_eval_users1000_cg5_m2_seed1_popularity_f1_threshold` |
| 2 | 2 | 3456 | 57312 | 5.5278 | 0.6973 | 0.6947 | 188.0000 | `20260702T122827Z_ml-1m_cap20_eval_users1000_cg5_m2_seed2_popularity_f1_threshold` |
| 3 | 0 | 3734 | 70532 | 4.7223 | 0.6388 | 0.6405 | 221.0000 | `20260702T122841Z_ml-1m_cap20_eval_users1000_cg5_m3_seed0_popularity_f1_threshold` |
| 3 | 1 | 3624 | 67748 | 4.6736 | 0.6400 | 0.6391 | 236.0000 | `20260702T122855Z_ml-1m_cap20_eval_users1000_cg5_m3_seed1_popularity_f1_threshold` |
| 3 | 2 | 3672 | 69088 | 4.7037 | 0.6292 | 0.6294 | 231.0000 | `20260702T122909Z_ml-1m_cap20_eval_users1000_cg5_m3_seed2_popularity_f1_threshold` |
| 9 | 0 | 4641 | 91390 | 1.9692 | 0.4596 | 0.4663 | 419.0000 | `20260702T122926Z_ml-1m_cap20_eval_users1000_cg5_m9_seed0_popularity_f1_threshold` |
| 9 | 1 | 4629 | 91050 | 1.9669 | 0.4626 | 0.4660 | 408.0000 | `20260702T122939Z_ml-1m_cap20_eval_users1000_cg5_m9_seed1_popularity_f1_threshold` |
| 9 | 2 | 4621 | 90890 | 1.9669 | 0.4498 | 0.4482 | 362.0000 | `20260702T122952Z_ml-1m_cap20_eval_users1000_cg5_m9_seed2_popularity_f1_threshold` |
| 19 | 0 | 4979 | 99580 | 1.0000 | 0.3308 | 0.3307 | 431.0000 | `20260702T123006Z_ml-1m_cap20_eval_users1000_cg5_m19_seed0_popularity_f1_threshold` |
| 19 | 1 | 4988 | 99760 | 1.0000 | 0.3297 | 0.3369 | 483.0000 | `20260702T123021Z_ml-1m_cap20_eval_users1000_cg5_m19_seed1_popularity_f1_threshold` |
| 19 | 2 | 4982 | 99640 | 1.0000 | 0.3254 | 0.3266 | 462.0000 | `20260702T123035Z_ml-1m_cap20_eval_users1000_cg5_m19_seed2_popularity_f1_threshold` |

#### ml-1m popularity_ranking

| m | seed | test groups | test rows | NDCG@5 | HR@1 | score-tie groups | run |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0 | 2867 | 50786 | 0.8876 | 0.9374 | 0.3080 | `20260702T122721Z_ml-1m_cap20_eval_users1000_cg5_m1_seed0_popularity_ranking` |
| 1 | 1 | 2733 | 47718 | 0.8888 | 0.9368 | 0.3191 | `20260702T122734Z_ml-1m_cap20_eval_users1000_cg5_m1_seed1_popularity_ranking` |
| 1 | 2 | 2763 | 48500 | 0.8849 | 0.9381 | 0.3145 | `20260702T122747Z_ml-1m_cap20_eval_users1000_cg5_m1_seed2_popularity_ranking` |
| 2 | 0 | 3528 | 58893 | 0.7835 | 0.8731 | 0.3892 | `20260702T122801Z_ml-1m_cap20_eval_users1000_cg5_m2_seed0_popularity_ranking` |
| 2 | 1 | 3386 | 56217 | 0.7844 | 0.8696 | 0.3907 | `20260702T122814Z_ml-1m_cap20_eval_users1000_cg5_m2_seed1_popularity_ranking` |
| 2 | 2 | 3456 | 57312 | 0.7779 | 0.8600 | 0.3990 | `20260702T122828Z_ml-1m_cap20_eval_users1000_cg5_m2_seed2_popularity_ranking` |
| 3 | 0 | 3734 | 70532 | 0.7039 | 0.8078 | 0.5362 | `20260702T122841Z_ml-1m_cap20_eval_users1000_cg5_m3_seed0_popularity_ranking` |
| 3 | 1 | 3624 | 67748 | 0.7096 | 0.8179 | 0.5190 | `20260702T122855Z_ml-1m_cap20_eval_users1000_cg5_m3_seed1_popularity_ranking` |
| 3 | 2 | 3672 | 69088 | 0.6979 | 0.8030 | 0.5324 | `20260702T122909Z_ml-1m_cap20_eval_users1000_cg5_m3_seed2_popularity_ranking` |
| 9 | 0 | 4641 | 91390 | 0.6606 | 0.5684 | 0.6643 | `20260702T122926Z_ml-1m_cap20_eval_users1000_cg5_m9_seed0_popularity_ranking` |
| 9 | 1 | 4629 | 91050 | 0.6591 | 0.5662 | 0.6574 | `20260702T122939Z_ml-1m_cap20_eval_users1000_cg5_m9_seed1_popularity_ranking` |
| 9 | 2 | 4621 | 90890 | 0.6435 | 0.5620 | 0.6644 | `20260702T122952Z_ml-1m_cap20_eval_users1000_cg5_m9_seed2_popularity_ranking` |
| 19 | 0 | 4979 | 99580 | 0.5981 | 0.3754 | 0.7009 | `20260702T123006Z_ml-1m_cap20_eval_users1000_cg5_m19_seed0_popularity_ranking` |
| 19 | 1 | 4988 | 99760 | 0.5866 | 0.3662 | 0.7115 | `20260702T123021Z_ml-1m_cap20_eval_users1000_cg5_m19_seed1_popularity_ranking` |
| 19 | 2 | 4982 | 99640 | 0.5743 | 0.3619 | 0.7073 | `20260702T123035Z_ml-1m_cap20_eval_users1000_cg5_m19_seed2_popularity_ranking` |

### Steam eval_users1000_cg5

#### steam popularity_f1_threshold

| m | seed | test groups | test rows | positives/group | F1 | micro F1 | threshold | run |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0 | 2027 | 31428 | 7.7523 | 0.8909 | 0.8768 | 356.0000 | `20260702T123214Z_steam_cap20_eval_users1000_cg5_m1_seed0_popularity_f1_threshold` |
| 1 | 1 | 2002 | 31304 | 7.8182 | 0.8958 | 0.8849 | 289.0000 | `20260702T123355Z_steam_cap20_eval_users1000_cg5_m1_seed1_popularity_f1_threshold` |
| 1 | 2 | 2000 | 31222 | 7.8055 | 0.8926 | 0.8789 | 290.0000 | `20260702T123536Z_steam_cap20_eval_users1000_cg5_m1_seed2_popularity_f1_threshold` |
| 2 | 0 | 2661 | 41847 | 5.2420 | 0.8417 | 0.8270 | 468.0000 | `20260702T123714Z_steam_cap20_eval_users1000_cg5_m2_seed0_popularity_f1_threshold` |
| 2 | 1 | 2655 | 41727 | 5.2388 | 0.8486 | 0.8368 | 476.0000 | `20260702T123849Z_steam_cap20_eval_users1000_cg5_m2_seed1_popularity_f1_threshold` |
| 2 | 2 | 2676 | 41946 | 5.2250 | 0.8433 | 0.8301 | 534.0000 | `20260702T124029Z_steam_cap20_eval_users1000_cg5_m2_seed2_popularity_f1_threshold` |
| 3 | 0 | 2946 | 52484 | 4.4538 | 0.8088 | 0.7950 | 674.0000 | `20260702T124210Z_steam_cap20_eval_users1000_cg5_m3_seed0_popularity_f1_threshold` |
| 3 | 1 | 2929 | 52320 | 4.4657 | 0.8085 | 0.7975 | 714.0000 | `20260702T124350Z_steam_cap20_eval_users1000_cg5_m3_seed1_popularity_f1_threshold` |
| 3 | 2 | 2947 | 52840 | 4.4825 | 0.8051 | 0.7962 | 665.0000 | `20260702T124529Z_steam_cap20_eval_users1000_cg5_m3_seed2_popularity_f1_threshold` |
| 9 | 0 | 4161 | 80770 | 1.9411 | 0.6915 | 0.6872 | 1210.0000 | `20260702T124710Z_steam_cap20_eval_users1000_cg5_m9_seed0_popularity_f1_threshold` |
| 9 | 1 | 4149 | 80610 | 1.9429 | 0.6906 | 0.6802 | 1070.0000 | `20260702T124849Z_steam_cap20_eval_users1000_cg5_m9_seed1_popularity_f1_threshold` |
| 9 | 2 | 4188 | 81400 | 1.9436 | 0.6873 | 0.6872 | 1250.0000 | `20260702T125029Z_steam_cap20_eval_users1000_cg5_m9_seed2_popularity_f1_threshold` |
| 19 | 0 | 4730 | 94600 | 1.0000 | 0.5803 | 0.5832 | 1465.0000 | `20260702T125207Z_steam_cap20_eval_users1000_cg5_m19_seed0_popularity_f1_threshold` |
| 19 | 1 | 4707 | 94140 | 1.0000 | 0.5869 | 0.5913 | 1642.0000 | `20260702T125350Z_steam_cap20_eval_users1000_cg5_m19_seed1_popularity_f1_threshold` |
| 19 | 2 | 4754 | 95080 | 1.0000 | 0.5715 | 0.5696 | 1361.0000 | `20260702T125530Z_steam_cap20_eval_users1000_cg5_m19_seed2_popularity_f1_threshold` |

#### steam popularity_ranking

| m | seed | test groups | test rows | NDCG@5 | HR@1 | score-tie groups | run |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0 | 2027 | 31428 | 0.9669 | 0.9852 | 0.3320 | `20260702T123215Z_steam_cap20_eval_users1000_cg5_m1_seed0_popularity_ranking` |
| 1 | 1 | 2002 | 31304 | 0.9713 | 0.9882 | 0.3536 | `20260702T123356Z_steam_cap20_eval_users1000_cg5_m1_seed1_popularity_ranking` |
| 1 | 2 | 2000 | 31222 | 0.9711 | 0.9863 | 0.3430 | `20260702T123536Z_steam_cap20_eval_users1000_cg5_m1_seed2_popularity_ranking` |
| 2 | 0 | 2661 | 41847 | 0.9245 | 0.9662 | 0.4961 | `20260702T123714Z_steam_cap20_eval_users1000_cg5_m2_seed0_popularity_ranking` |
| 2 | 1 | 2655 | 41727 | 0.9289 | 0.9684 | 0.5205 | `20260702T123850Z_steam_cap20_eval_users1000_cg5_m2_seed1_popularity_ranking` |
| 2 | 2 | 2676 | 41946 | 0.9281 | 0.9660 | 0.4981 | `20260702T124029Z_steam_cap20_eval_users1000_cg5_m2_seed2_popularity_ranking` |
| 3 | 0 | 2946 | 52484 | 0.8847 | 0.9528 | 0.6511 | `20260702T124211Z_steam_cap20_eval_users1000_cg5_m3_seed0_popularity_ranking` |
| 3 | 1 | 2929 | 52320 | 0.8867 | 0.9564 | 0.6640 | `20260702T124351Z_steam_cap20_eval_users1000_cg5_m3_seed1_popularity_ranking` |
| 3 | 2 | 2947 | 52840 | 0.8874 | 0.9536 | 0.6620 | `20260702T124530Z_steam_cap20_eval_users1000_cg5_m3_seed2_popularity_ranking` |
| 9 | 0 | 4161 | 80770 | 0.8825 | 0.8470 | 0.8234 | `20260702T124711Z_steam_cap20_eval_users1000_cg5_m9_seed0_popularity_ranking` |
| 9 | 1 | 4149 | 80610 | 0.8814 | 0.8396 | 0.8373 | `20260702T124850Z_steam_cap20_eval_users1000_cg5_m9_seed1_popularity_ranking` |
| 9 | 2 | 4188 | 81400 | 0.8793 | 0.8371 | 0.8357 | `20260702T125029Z_steam_cap20_eval_users1000_cg5_m9_seed2_popularity_ranking` |
| 19 | 0 | 4730 | 94600 | 0.8325 | 0.6748 | 0.8837 | `20260702T125208Z_steam_cap20_eval_users1000_cg5_m19_seed0_popularity_ranking` |
| 19 | 1 | 4707 | 94140 | 0.8328 | 0.6745 | 0.8834 | `20260702T125351Z_steam_cap20_eval_users1000_cg5_m19_seed1_popularity_ranking` |
| 19 | 2 | 4754 | 95080 | 0.8341 | 0.6767 | 0.8814 | `20260702T125531Z_steam_cap20_eval_users1000_cg5_m19_seed2_popularity_ranking` |
