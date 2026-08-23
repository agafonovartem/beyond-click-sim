from __future__ import annotations

from time import perf_counter
from typing import Any

import pandas as pd

from beyond_click_sim.data.canonical import CanonicalDataset
from beyond_click_sim.evaluation import evaluate_policy_recommendations
from beyond_click_sim.tasks.base import Task, TaskSchema
from beyond_click_sim.tasks.item_enrichment import item_enrichment_manifest
from beyond_click_sim.tasks.policy_ranking import (
    POLICY_EVAL_KS,
    PolicyRankingTaskBuilder,
    _policy_hyperparameters,
)


class PolicyRerankingTaskBuilder(PolicyRankingTaskBuilder):
    """Build a policy-ranking task where policies re-rank a fully-labelled pool.

    The difference from :class:`PolicyRankingTaskBuilder` is where the slate
    comes from. There, each policy retrieves its top-k from the whole catalog
    and most recommended items have no held-out label, so unlabelled slots are
    scored 0. Here each policy re-ranks *the user's own held-out pool*, so every
    slate slot carries a real label and the zero-fill disappears entirely.

    That matters on an MCAR evaluation set: pool membership was assigned at
    random, so a zero-filled free-retrieval utility would converge to the same
    value for every policy regardless of quality. Re-ranking the pool keeps the
    labels informative.

    The test frame holds `policy.k` rows per (user, policy), ranked 1..k, with
    the same extra `policy` / `rank` columns the parent emits.
    """

    def __init__(self, *args: Any, source_column: str = "source", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.source_column = source_column

    def build(self, dataset: CanonicalDataset) -> Task:
        canonical_manifest = dataset.load_manifest()
        users = dataset.load_users()
        items = dataset.load_items()
        interactions = dataset.load_interactions()
        task_item_enrichment = item_enrichment_manifest(canonical_manifest, items)

        self._require_columns(interactions, [self.user_column, self.item_column])
        self._require_columns(
            interactions,
            [self.target_source_column, *self.history_context_columns],
        )
        # The splitter partitions on provenance, so a filter that dropped the
        # column must fail here rather than deep inside split().
        self._require_columns(interactions, [self.source_column])
        self._require_columns(users, [self.user_column])
        self._require_columns(items, [self.item_column])

        users, items, interactions = self.dataset_filter.filter(
            users=users,
            items=items,
            interactions=interactions,
        )
        split = self.splitter.split(interactions)

        test_interactions, test_eval_summary = self._sample_eval_rows(
            split.test,
            train=split.train,
        )

        # Every pool pair carries a real label; a miss is a bug, not a negative.
        test_targets: dict[tuple[Any, Any], Any] = {
            (row[self.user_column], row[self.item_column]): row[self.target_source_column]
            for _, row in test_interactions.iterrows()
        }

        pool_by_user: dict[Any, list[Any]] = (
            test_interactions.groupby(self.user_column, sort=True)[self.item_column]
            .agg(list)
            .to_dict()
        )
        test_users = users[users[self.user_column].isin(pool_by_user)].copy()

        rec_frames: list[pd.DataFrame] = []
        policy_metrics: list[dict[str, object]] = []
        fallback_counts: dict[str, int] = {}

        for policy in self.policies:
            t0 = perf_counter()
            policy.fit(
                split.train,
                items=items,
                user_column=self.user_column,
                item_column=self.item_column,
            )

            rows: list[dict[str, Any]] = []
            fallbacks = 0
            for user_id in test_users[self.user_column]:
                pool = pool_by_user[user_id]
                ordering = policy.rank_candidates(
                    user_id,
                    pool,
                    train_interactions=split.train,
                    user_column=self.user_column,
                    item_column=self.item_column,
                )
                if sorted(map(str, ordering)) != sorted(map(str, pool)):
                    raise ValueError(
                        f"{policy.name}.rank_candidates did not return a permutation "
                        f"of the candidate pool for user {user_id!r}"
                    )
                if self._is_untied_fallback(policy, user_id, pool, ordering):
                    fallbacks += 1
                rows.extend(
                    policy._build_recommendation_rows(
                        user_id,
                        list(ordering)[: policy.k],
                        user_column=self.user_column,
                        item_column=self.item_column,
                    )
                )

            recs = pd.DataFrame(
                rows, columns=[self.user_column, self.item_column, "policy", "rank"]
            )
            missing = [
                key
                for key in zip(recs[self.user_column], recs[self.item_column], strict=True)
                if key not in test_targets
            ]
            if missing:
                raise ValueError(
                    "Re-ranking slates must be fully labelled, but "
                    f"{len(missing)} (user, item) pairs had no held-out label "
                    f"for policy {policy.name!r}; first few: {missing[:5]}"
                )
            recs[self.target_column] = [
                test_targets[(u, i)]
                for u, i in zip(recs[self.user_column], recs[self.item_column], strict=True)
            ]

            fallback_counts[policy.name] = fallbacks
            policy_metric = evaluate_policy_recommendations(
                recs,
                targets=recs[self.target_column],
                user_column=self.user_column,
                policy_name=policy.name,
                k=policy.k,
                ks=POLICY_EVAL_KS,
                fit_recommend_seconds=perf_counter() - t0,
            )
            policy_metric["rank_candidates_fallback_users"] = fallbacks
            policy_metrics.append(policy_metric)
            print(
                f"Policy {policy.name}: "
                f"slate_positive_rate={policy_metric['mean_hit_rate']:.4f} "
                f"users={policy_metric['n_users']} "
                f"rows={policy_metric['n_recommendations']} "
                f"fallback_users={fallbacks} "
                f"({policy_metric.get('fit_recommend_seconds', 0.0):.3f}s)",
                flush=True,
            )
            rec_frames.append(recs)

        all_recs = pd.concat(rec_frames, ignore_index=True)

        # Mask history context columns on test rows (no leakage).
        for col in self.history_context_columns:
            all_recs[col] = pd.NA

        items, item_feature_manifest = self._enrich_items_from_train(
            items=items,
            train_interactions=split.train,
        )

        user_features = self._prefixed_features(users, self.user_column, "user")
        item_features = self._prefixed_features(items, self.item_column, "item")
        feature_columns = self._feature_columns(
            user_features,
            item_features,
            user_column=self.user_column,
            item_column=self.item_column,
        )

        test_task_columns = self._task_columns(feature_columns) + ["policy", "rank"]
        test = self._with_features(
            rows=all_recs,
            user_features=user_features,
            item_features=item_features,
            columns=test_task_columns,
            user_column=self.user_column,
            item_column=self.item_column,
        )
        train = self._with_features(
            rows=self._target_rows(split.train),
            user_features=user_features,
            item_features=item_features,
            columns=self._task_columns(feature_columns),
            user_column=self.user_column,
            item_column=self.item_column,
        )
        val = pd.DataFrame(columns=train.columns)

        schema = TaskSchema(
            target_column=self.target_column,
            feature_columns=feature_columns,
            id_columns=(self.user_column, self.item_column),
            candidate_group_column=None,
            sampled_column=None,
            history_context_columns=self.history_context_columns,
        )

        return Task(
            name=self.name,
            train=train,
            val=val,
            test=test,
            schema=schema,
            manifest={
                "protocol": "policy_reranking",
                "candidate_construction": {
                    "kind": "held_out_pool_rerank",
                    "description": (
                        "Each policy re-ranks the user's own held-out pool and "
                        "emits its top-k. Every slate slot carries a real label, "
                        "so no zero-filling is applied."
                    ),
                    "k": int(self.policies[0].k),
                    "pool_sizes": sorted(
                        {int(len(pool)) for pool in pool_by_user.values()}
                    ),
                },
                "dataset": dataset.name,
                "dataset_version": dataset.version,
                "target_source_column": self.target_source_column,
                "target_column": self.target_column,
                "feature_columns": list(feature_columns),
                "history_context_columns": list(self.history_context_columns),
                "item_enrichment": task_item_enrichment,
                "item_feature_builder": item_feature_manifest,
                "policies": [
                    {
                        "class": policy.__class__.__name__,
                        "name": policy.name,
                        "k": policy.k,
                        "seed": policy.seed,
                        "positive_feedback_column": getattr(
                            policy, "positive_feedback_column", None
                        ),
                        "hyperparameters": _policy_hyperparameters(policy),
                        "hyperparameter_selection": getattr(
                            policy, "hyperparameter_selection", None
                        ),
                        "fit_rows": getattr(policy, "fit_row_count_", None),
                        "fit_users": getattr(policy, "fit_user_count_", None),
                        "rank_candidates_fallback_users": fallback_counts[policy.name],
                    }
                    for policy in self.policies
                ],
                "filter": self._component_manifest(self.dataset_filter),
                "splitter": self._component_manifest(self.splitter),
                "sampler": None,
                "eval_sampler": None,
                "eval_user_selection": {
                    "kind": "post_split",
                    "test": test_eval_summary,
                },
                "policy_recommendation_metrics": {
                    "protocol": "rerank_pool",
                    "k": int(self.policies[0].k),
                    "target_source_column": self.target_source_column,
                    "ks": list(POLICY_EVAL_KS),
                    "ranking_score": "negative_rank",
                    "aggregation_headline": "macro_by_user_group_mean",
                    "caveat": (
                        "mean_hit_rate and every ndcg@*/hit_rate@* here are over "
                        "k-item slates drawn from a held-out pool, not over a top-k "
                        "retrieved from the full catalog. They are NOT comparable "
                        "with the free-retrieval policy-ranking families."
                    ),
                    "policies": policy_metrics,
                },
                "rows": {
                    "train": int(len(train)),
                    "val": 0,
                    "test": int(len(test)),
                },
                "users": {
                    "filtered": int(interactions[self.user_column].nunique()),
                    "train": int(split.train[self.user_column].nunique()),
                    "test": int(test_interactions[self.user_column].nunique()),
                },
                "items": int(items[self.item_column].nunique()),
            },
        )

    @staticmethod
    def _is_untied_fallback(policy, user_id, pool, ordering) -> bool:
        """True when the ordering is indistinguishable from the seeded shuffle.

        A policy with no representation for a user falls back to that shuffle, so
        counting matches tells us how often a policy was effectively Random. It is
        a heuristic: a policy could coincidentally reproduce the shuffle.
        """

        try:
            shuffled = policy._seeded_permutation(user_id, pool)
        except Exception:
            return False
        return list(map(str, shuffled)) == list(map(str, ordering))
