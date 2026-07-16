from __future__ import annotations


POINTWISE_METRICS_FILENAME = "metrics.json"
RANKING_METRICS_FILENAME = "metrics_ranking.json"

POINTWISE_MAIN_METRIC = "test.macro_by_user_group_mean.f1"
RANKING_MAIN_METRIC = "test.macro_by_user_group_mean.ndcg@5"

RANKING_KS = (1, 3, 5, 10)
RANKING_TIE_POLICY = "average"
