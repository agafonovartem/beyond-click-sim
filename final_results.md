# In Distribution

Snapshot of local `eval_users1000` in-distribution artifacts. Metrics are copied from compact JSON provenance under `outputs/in_distribution/`; this is a run registry, not a paper table.

Notes:
- Main pointwise metric: `test.macro_by_user_group_mean.f1` when available; otherwise `test.macro_by_group.f1` and marked as `group`.
- Main ranking metric: `test.macro_by_user_group_mean.ndcg@5` when available; HR@1 is included for quick sanity checks.
- Pop pointwise uses `popularity_f1_threshold`; pop ranking uses raw popularity scores via `popularity_ranking`.
- Paper-table notebook: `notebooks/compare_popularity_llm_eval1000.ipynb` renders four compact seed-averaged tables: ML-1M pointwise, Steam pointwise, ML-1M ranking, and Steam ranking.
- Raw-score pop ranking is included in the ranking tables from `metrics_ranking.json`.
- Regression notebook: `notebooks/compare_regression_mean_eval1000.ipynb` currently renders the ML-1M rating `MeanRegressor` seed table and seed-averaged summary. `ModeRegressor` is the primary discrete constant baseline for simulator-style rating prediction and is listed directly below.

## Regression Prediction

### ML-1M rating eval_1000 users

Protocol:
- Dataset: `ml-1m` canonical `v1`.
- Target: `target_rating`; `rating` is available as train history context and hidden on val/test rows.
- Split: `RandomFractionSplitter(0.7, 0.1, 0.2, group_column="user_id")`, seeds 0-4.
- Filter: `MinUserInteractionsFilter(10)`.
- Evaluation budget: post-split `PostSplitUserSampler(n_users=1000, seed=seed)`.
- Candidate construction: none; no negatives, no `candidate_group`, no `sampled`.
- Methods: `llm_regressor_vllm_llama33_70b_full` predicts a strict bare integer rating from train history and item metadata; `mode_regressor` fits the most frequent train rating and predicts a valid discrete rating; `mean_regressor` fits the continuous train target mean and is retained as a MAE/RMSE diagnostic.
- Main metric: `test.macro_by_user_mean.mae`; micro MAE/RMSE are retained as secondary diagnostics.
- Scope: 6040 filtered/train users and 3883 items for all seeds; test users are capped to 1000 after splitting.

#### LLMRegressor

| seed | train rows | test rows | test users | llm errors | scored/requested | macro MAE | macro RMSE | micro MAE | micro RMSE | run |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 694335 | 34520 | 1000 | 0 | 34520/34520 | 0.8043 | 1.0862 | 0.8096 | 1.1237 | `20260612T154600Z_ml-1m_rating_eval_users1000_seed0_llm_regressor_vllm_llama33_70b_full` |

#### ModeRegressor

| seed | train rows | val rows | test rows | test users | mode | macro MAE | macro RMSE | micro MAE | micro RMSE | run |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 694335 | 17629 | 34520 | 1000 | 4.0000 | 0.8364 | 1.0932 | 0.8468 | 1.1665 | `20260612T152221Z_ml-1m_rating_eval_users1000_seed0_mode_regressor` |

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

### ML-1M eval_1000 users

#### LLM yes no

| m | seed | status | llm_errors | groups scored/requested | pointwise F1 | pointwise scope | ranking NDCG@5 | HR@1 | run |
|---:|---:|---|---:|---:|---:|---|---:|---:|---|
| 1 | 0 | complete | 0 | 3887/3887 | 0.7023 | user_group | 0.6515 | 0.6352 | `20260610T063021Z_ml-1m_cap20_eval_users1000_m1_seed0_llm_yes_no_vllm_llama33_70b_full` |
| 2 | 0 | complete | 0 | 6173/6173 | 0.5827 | user_group | 0.4979 | 0.4679 | `20260610T081306Z_ml-1m_cap20_eval_users1000_m2_seed0_llm_yes_no_vllm_llama33_70b_full` |
| 3 | 0 | complete | 0 | 7293/7293 | 0.4968 | user_group | 0.4029 | 0.3689 | `20260611T180320Z_ml-1m_cap20_eval_users1000_m3_seed0_llm_yes_no_vllm_llama33_70b_full` |
| 9 | 0 | complete | 0 | 17517/17517 | 0.2683 | user_group | 0.3019 | 0.1669 | `20260611T211709Z_ml-1m_cap20_eval_users1000_m9_seed0_llm_yes_no_vllm_llama33_70b_full` |
| 19 | 0 | partial_non_final | 3 | 448/34520 | 0.1743 | user_group | n/a | n/a | `20260610T133914Z_ml-1m_cap20_eval_users1000_m19_seed0_llm_yes_no_vllm_llama33_70b_full` |

#### Pop

##### Pointwise threshold

| m | seed | F1 | micro F1 | threshold | run |
|---:|---:|---:|---:|---:|---|
| 1 | 0 | 0.7913 | 0.7900 | 153.0000 | `20260610T111838Z_ml-1m_cap20_eval_users1000_m1_seed0_popularity_f1_threshold` |
| 1 | 1 | 0.7955 | 0.7897 | 139.0000 | `20260610T111925Z_ml-1m_cap20_eval_users1000_m1_seed1_popularity_f1_threshold` |
| 1 | 2 | 0.7881 | 0.7841 | 97.0000 | `20260610T112012Z_ml-1m_cap20_eval_users1000_m1_seed2_popularity_f1_threshold` |
| 1 | 3 | 0.7964 | 0.7914 | 125.0000 | `20260610T112104Z_ml-1m_cap20_eval_users1000_m1_seed3_popularity_f1_threshold` |
| 1 | 4 | 0.7951 | 0.7903 | 121.0000 | `20260610T112157Z_ml-1m_cap20_eval_users1000_m1_seed4_popularity_f1_threshold` |
| 2 | 0 | 0.7019 | 0.6985 | 197.0000 | `20260610T112254Z_ml-1m_cap20_eval_users1000_m2_seed0_popularity_f1_threshold` |
| 2 | 1 | 0.7018 | 0.6964 | 176.0000 | `20260610T112328Z_ml-1m_cap20_eval_users1000_m2_seed1_popularity_f1_threshold` |
| 2 | 2 | 0.6986 | 0.6880 | 183.0000 | `20260610T112401Z_ml-1m_cap20_eval_users1000_m2_seed2_popularity_f1_threshold` |
| 2 | 3 | 0.6964 | 0.6949 | 161.0000 | `20260610T112438Z_ml-1m_cap20_eval_users1000_m2_seed3_popularity_f1_threshold` |
| 2 | 4 | 0.7008 | 0.6960 | 186.0000 | `20260610T112532Z_ml-1m_cap20_eval_users1000_m2_seed4_popularity_f1_threshold` |
| 3 | 0 | 0.6367 | 0.6361 | 221.0000 | `20260610T112631Z_ml-1m_cap20_eval_users1000_m3_seed0_popularity_f1_threshold` |
| 3 | 1 | 0.6409 | 0.6329 | 243.0000 | `20260610T112731Z_ml-1m_cap20_eval_users1000_m3_seed1_popularity_f1_threshold` |
| 3 | 2 | 0.6285 | 0.6223 | 231.0000 | `20260610T112829Z_ml-1m_cap20_eval_users1000_m3_seed2_popularity_f1_threshold` |
| 3 | 3 | 0.6385 | 0.6328 | 261.0000 | `20260610T112920Z_ml-1m_cap20_eval_users1000_m3_seed3_popularity_f1_threshold` |
| 3 | 4 | 0.6385 | 0.6339 | 243.0000 | `20260610T112954Z_ml-1m_cap20_eval_users1000_m3_seed4_popularity_f1_threshold` |
| 9 | 0 | 0.4593 | 0.4585 | 419.0000 | `20260610T113033Z_ml-1m_cap20_eval_users1000_m9_seed0_popularity_f1_threshold` |
| 9 | 1 | 0.4620 | 0.4554 | 408.0000 | `20260610T113114Z_ml-1m_cap20_eval_users1000_m9_seed1_popularity_f1_threshold` |
| 9 | 2 | 0.4514 | 0.4424 | 362.0000 | `20260610T113154Z_ml-1m_cap20_eval_users1000_m9_seed2_popularity_f1_threshold` |
| 9 | 3 | 0.4508 | 0.4533 | 407.0000 | `20260610T113236Z_ml-1m_cap20_eval_users1000_m9_seed3_popularity_f1_threshold` |
| 9 | 4 | 0.4572 | 0.4553 | 413.0000 | `20260610T113319Z_ml-1m_cap20_eval_users1000_m9_seed4_popularity_f1_threshold` |
| 19 | 0 | 0.3309 | 0.3330 | 432.0000 | `20260610T113410Z_ml-1m_cap20_eval_users1000_m19_seed0_popularity_f1_threshold` |
| 19 | 1 | 0.3350 | 0.3324 | 440.0000 | `20260610T113501Z_ml-1m_cap20_eval_users1000_m19_seed1_popularity_f1_threshold` |
| 19 | 2 | 0.3268 | 0.3255 | 464.0000 | `20260610T113553Z_ml-1m_cap20_eval_users1000_m19_seed2_popularity_f1_threshold` |
| 19 | 3 | 0.3330 | 0.3365 | 483.0000 | `20260610T113645Z_ml-1m_cap20_eval_users1000_m19_seed3_popularity_f1_threshold` |
| 19 | 4 | 0.3321 | 0.3360 | 477.0000 | `20260610T113750Z_ml-1m_cap20_eval_users1000_m19_seed4_popularity_f1_threshold` |

##### Raw-score ranking

| m | seed | NDCG@5 | HR@1 | run |
|---:|---:|---:|---:|---|
| 1 | 0 | 0.8880 | 0.9358 | `20260612T054530Z_ml-1m_cap20_eval_users1000_m1_seed0_popularity_ranking` |
| 1 | 1 | 0.8895 | 0.9378 | `20260612T054603Z_ml-1m_cap20_eval_users1000_m1_seed1_popularity_ranking` |
| 1 | 2 | 0.8837 | 0.9346 | `20260612T054635Z_ml-1m_cap20_eval_users1000_m1_seed2_popularity_ranking` |
| 1 | 3 | 0.8864 | 0.9332 | `20260612T054708Z_ml-1m_cap20_eval_users1000_m1_seed3_popularity_ranking` |
| 1 | 4 | 0.8883 | 0.9296 | `20260612T054741Z_ml-1m_cap20_eval_users1000_m1_seed4_popularity_ranking` |
| 2 | 0 | 0.7840 | 0.8684 | `20260612T054814Z_ml-1m_cap20_eval_users1000_m2_seed0_popularity_ranking` |
| 2 | 1 | 0.7858 | 0.8664 | `20260612T054849Z_ml-1m_cap20_eval_users1000_m2_seed1_popularity_ranking` |
| 2 | 2 | 0.7802 | 0.8578 | `20260612T054924Z_ml-1m_cap20_eval_users1000_m2_seed2_popularity_ranking` |
| 2 | 3 | 0.7859 | 0.8686 | `20260612T054959Z_ml-1m_cap20_eval_users1000_m2_seed3_popularity_ranking` |
| 2 | 4 | 0.7814 | 0.8538 | `20260612T055034Z_ml-1m_cap20_eval_users1000_m2_seed4_popularity_ranking` |
| 3 | 0 | 0.7069 | 0.8039 | `20260612T055110Z_ml-1m_cap20_eval_users1000_m3_seed0_popularity_ranking` |
| 3 | 1 | 0.7137 | 0.8169 | `20260612T055146Z_ml-1m_cap20_eval_users1000_m3_seed1_popularity_ranking` |
| 3 | 2 | 0.7023 | 0.8006 | `20260612T055222Z_ml-1m_cap20_eval_users1000_m3_seed2_popularity_ranking` |
| 3 | 3 | 0.7094 | 0.8125 | `20260612T055259Z_ml-1m_cap20_eval_users1000_m3_seed3_popularity_ranking` |
| 3 | 4 | 0.7082 | 0.8023 | `20260612T055336Z_ml-1m_cap20_eval_users1000_m3_seed4_popularity_ranking` |
| 9 | 0 | 0.6599 | 0.5631 | `20260612T055416Z_ml-1m_cap20_eval_users1000_m9_seed0_popularity_ranking` |
| 9 | 1 | 0.6622 | 0.5694 | `20260612T055509Z_ml-1m_cap20_eval_users1000_m9_seed1_popularity_ranking` |
| 9 | 2 | 0.6481 | 0.5559 | `20260612T055600Z_ml-1m_cap20_eval_users1000_m9_seed2_popularity_ranking` |
| 9 | 3 | 0.6532 | 0.5536 | `20260612T055651Z_ml-1m_cap20_eval_users1000_m9_seed3_popularity_ranking` |
| 9 | 4 | 0.6567 | 0.5593 | `20260612T055745Z_ml-1m_cap20_eval_users1000_m9_seed4_popularity_ranking` |
| 19 | 0 | 0.5917 | 0.3686 | `20260612T055846Z_ml-1m_cap20_eval_users1000_m19_seed0_popularity_ranking` |
| 19 | 1 | 0.5932 | 0.3739 | `20260612T060019Z_ml-1m_cap20_eval_users1000_m19_seed1_popularity_ranking` |
| 19 | 2 | 0.5792 | 0.3584 | `20260612T060146Z_ml-1m_cap20_eval_users1000_m19_seed2_popularity_ranking` |
| 19 | 3 | 0.5863 | 0.3648 | `20260612T060316Z_ml-1m_cap20_eval_users1000_m19_seed3_popularity_ranking` |
| 19 | 4 | 0.5891 | 0.3673 | `20260612T060459Z_ml-1m_cap20_eval_users1000_m19_seed4_popularity_ranking` |

### Steam eval_1000 users

#### LLM yes no

| m | seed | status | llm_errors | groups scored/requested | pointwise F1 | pointwise scope | ranking NDCG@5 | HR@1 | run |
|---:|---:|---|---:|---:|---:|---|---:|---:|---|
| 1 | 0 | complete | 0 | 2243/2243 | 0.7583 | user_group | 0.7365 | 0.7104 | `20260610T065422Z_steam_cap20_eval_users1000_m1_seed0_llm_yes_no_vllm_llama33_70b_full` |
| 2 | 0 | complete | 0 | 3340/3340 | 0.6448 | user_group | 0.5773 | 0.5319 | `20260610T093426Z_steam_cap20_eval_users1000_m2_seed0_llm_yes_no_vllm_llama33_70b_full` |
| 3 | 0 | complete | 0 | 3930/3930 | 0.5555 | user_group | 0.4799 | 0.4220 | `20260610T140623Z_steam_cap20_eval_users1000_m3_seed0_llm_yes_no_vllm_llama33_70b_full` |
| 9 | 0 | complete | 0 | 9086/9086 | 0.2992 | user_group | 0.3462 | 0.1854 | `20260610T183724Z_steam_cap20_eval_users1000_m9_seed0_llm_yes_no_vllm_llama33_70b_full` |
| 19 | 0 | complete | 0 | 17641/17641 | 0.1675 | user_group | 0.2728 | 0.0939 | `20260611T024141Z_steam_cap20_eval_users1000_m19_seed0_llm_yes_no_vllm_llama33_70b_full` |

#### Pop

##### Pointwise threshold

| m | seed | F1 | micro F1 | threshold | run |
|---:|---:|---:|---:|---:|---|
| 1 | 0 | 0.8914 | 0.8659 | 356.0000 | `20260610T114057Z_steam_cap20_eval_users1000_m1_seed0_popularity_f1_threshold` |
| 1 | 1 | 0.8955 | 0.8732 | 289.0000 | `20260610T114617Z_steam_cap20_eval_users1000_m1_seed1_popularity_f1_threshold` |
| 1 | 2 | 0.8925 | 0.8632 | 290.0000 | `20260610T115144Z_steam_cap20_eval_users1000_m1_seed2_popularity_f1_threshold` |
| 1 | 3 | 0.8875 | 0.8636 | 264.0000 | `20260610T115712Z_steam_cap20_eval_users1000_m1_seed3_popularity_f1_threshold` |
| 1 | 4 | 0.8984 | 0.8636 | 349.0000 | `20260610T120235Z_steam_cap20_eval_users1000_m1_seed4_popularity_f1_threshold` |
| 2 | 0 | 0.8421 | 0.8114 | 468.0000 | `20260610T120758Z_steam_cap20_eval_users1000_m2_seed0_popularity_f1_threshold` |
| 2 | 1 | 0.8488 | 0.8193 | 476.0000 | `20260610T121307Z_steam_cap20_eval_users1000_m2_seed1_popularity_f1_threshold` |
| 2 | 2 | 0.8422 | 0.8021 | 534.0000 | `20260610T120156Z_steam_cap20_eval_users1000_m2_seed2_popularity_f1_threshold` |
| 2 | 3 | 0.8373 | 0.7984 | 611.0000 | `20260610T120709Z_steam_cap20_eval_users1000_m2_seed3_popularity_f1_threshold` |
| 2 | 4 | 0.8418 | 0.8062 | 545.0000 | `20260610T121208Z_steam_cap20_eval_users1000_m2_seed4_popularity_f1_threshold` |
| 3 | 0 | 0.8086 | 0.7731 | 674.0000 | `20260610T121657Z_steam_cap20_eval_users1000_m3_seed0_popularity_f1_threshold` |
| 3 | 1 | 0.8112 | 0.7765 | 667.0000 | `20260610T122157Z_steam_cap20_eval_users1000_m3_seed1_popularity_f1_threshold` |
| 3 | 2 | 0.8042 | 0.7642 | 665.0000 | `20260610T122659Z_steam_cap20_eval_users1000_m3_seed2_popularity_f1_threshold` |
| 3 | 3 | 0.8096 | 0.7650 | 643.0000 | `20260610T123209Z_steam_cap20_eval_users1000_m3_seed3_popularity_f1_threshold` |
| 3 | 4 | 0.8114 | 0.7694 | 548.0000 | `20260610T123715Z_steam_cap20_eval_users1000_m3_seed4_popularity_f1_threshold` |
| 9 | 0 | 0.6886 | 0.6462 | 1210.0000 | `20260610T124246Z_steam_cap20_eval_users1000_m9_seed0_popularity_f1_threshold` |
| 9 | 1 | 0.6915 | 0.6450 | 1070.0000 | `20260610T124802Z_steam_cap20_eval_users1000_m9_seed1_popularity_f1_threshold` |
| 9 | 2 | 0.6859 | 0.6366 | 1250.0000 | `20260610T125316Z_steam_cap20_eval_users1000_m9_seed2_popularity_f1_threshold` |
| 9 | 3 | 0.6889 | 0.6361 | 1153.0000 | `20260610T125842Z_steam_cap20_eval_users1000_m9_seed3_popularity_f1_threshold` |
| 9 | 4 | 0.6887 | 0.6356 | 1322.0000 | `20260610T130445Z_steam_cap20_eval_users1000_m9_seed4_popularity_f1_threshold` |
| 19 | 0 | 0.5790 | 0.5393 | 1465.0000 | `20260610T131133Z_steam_cap20_eval_users1000_m19_seed0_popularity_f1_threshold` |
| 19 | 1 | 0.5823 | 0.5407 | 1642.0000 | `20260610T131823Z_steam_cap20_eval_users1000_m19_seed1_popularity_f1_threshold` |
| 19 | 2 | 0.5708 | 0.5340 | 1724.0000 | `20260610T132439Z_steam_cap20_eval_users1000_m19_seed2_popularity_f1_threshold` |
| 19 | 3 | 0.5774 | 0.5370 | 1665.0000 | `20260610T133124Z_steam_cap20_eval_users1000_m19_seed3_popularity_f1_threshold` |
| 19 | 4 | 0.5796 | 0.5294 | 1429.0000 | `20260610T133808Z_steam_cap20_eval_users1000_m19_seed4_popularity_f1_threshold` |

##### Raw-score ranking

| m | seed | NDCG@5 | HR@1 | run |
|---:|---:|---:|---:|---|
| 1 | 0 | 0.9671 | 0.9860 | `20260612T061031Z_steam_cap20_eval_users1000_m1_seed0_popularity_ranking` |
| 1 | 1 | 0.9715 | 0.9890 | `20260612T061522Z_steam_cap20_eval_users1000_m1_seed1_popularity_ranking` |
| 1 | 2 | 0.9710 | 0.9863 | `20260612T062016Z_steam_cap20_eval_users1000_m1_seed2_popularity_ranking` |
| 1 | 3 | 0.9695 | 0.9915 | `20260612T062509Z_steam_cap20_eval_users1000_m1_seed3_popularity_ranking` |
| 1 | 4 | 0.9724 | 0.9857 | `20260612T063002Z_steam_cap20_eval_users1000_m1_seed4_popularity_ranking` |
| 2 | 0 | 0.9264 | 0.9665 | `20260612T063455Z_steam_cap20_eval_users1000_m2_seed0_popularity_ranking` |
| 2 | 1 | 0.9293 | 0.9677 | `20260612T063949Z_steam_cap20_eval_users1000_m2_seed1_popularity_ranking` |
| 2 | 2 | 0.9281 | 0.9644 | `20260612T064444Z_steam_cap20_eval_users1000_m2_seed2_popularity_ranking` |
| 2 | 3 | 0.9243 | 0.9688 | `20260612T064935Z_steam_cap20_eval_users1000_m2_seed3_popularity_ranking` |
| 2 | 4 | 0.9291 | 0.9642 | `20260612T065429Z_steam_cap20_eval_users1000_m2_seed4_popularity_ranking` |
| 3 | 0 | 0.8870 | 0.9531 | `20260612T065925Z_steam_cap20_eval_users1000_m3_seed0_popularity_ranking` |
| 3 | 1 | 0.8888 | 0.9548 | `20260612T070419Z_steam_cap20_eval_users1000_m3_seed1_popularity_ranking` |
| 3 | 2 | 0.8885 | 0.9513 | `20260612T070915Z_steam_cap20_eval_users1000_m3_seed2_popularity_ranking` |
| 3 | 3 | 0.8837 | 0.9515 | `20260612T071411Z_steam_cap20_eval_users1000_m3_seed3_popularity_ranking` |
| 3 | 4 | 0.8927 | 0.9580 | `20260612T071906Z_steam_cap20_eval_users1000_m3_seed4_popularity_ranking` |
| 9 | 0 | 0.8828 | 0.8434 | `20260612T072401Z_steam_cap20_eval_users1000_m9_seed0_popularity_ranking` |
| 9 | 1 | 0.8822 | 0.8366 | `20260612T072859Z_steam_cap20_eval_users1000_m9_seed1_popularity_ranking` |
| 9 | 2 | 0.8798 | 0.8366 | `20260612T073400Z_steam_cap20_eval_users1000_m9_seed2_popularity_ranking` |
| 9 | 3 | 0.8803 | 0.8419 | `20260612T073857Z_steam_cap20_eval_users1000_m9_seed3_popularity_ranking` |
| 9 | 4 | 0.8818 | 0.8368 | `20260612T074358Z_steam_cap20_eval_users1000_m9_seed4_popularity_ranking` |
| 19 | 0 | 0.8313 | 0.6711 | `20260612T074858Z_steam_cap20_eval_users1000_m19_seed0_popularity_ranking` |
| 19 | 1 | 0.8321 | 0.6722 | `20260612T075411Z_steam_cap20_eval_users1000_m19_seed1_popularity_ranking` |
| 19 | 2 | 0.8316 | 0.6696 | `20260612T075919Z_steam_cap20_eval_users1000_m19_seed2_popularity_ranking` |
| 19 | 3 | 0.8325 | 0.6776 | `20260612T080431Z_steam_cap20_eval_users1000_m19_seed3_popularity_ranking` |
| 19 | 4 | 0.8361 | 0.6805 | `20260612T080946Z_steam_cap20_eval_users1000_m19_seed4_popularity_ranking` |
