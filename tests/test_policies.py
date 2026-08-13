from __future__ import annotations

import pandas as pd
import pytest

from beyond_click_sim.tasks.policies import ALSPolicy, BPRPolicy, ItemKNNPolicy, LightGCNPolicy, PopularityPolicy, RandomPolicy


def _items(*ids):
    return pd.DataFrame({"item_id": list(ids)})


def _interactions(*rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id"])


def _preference_interactions(*rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "feedback"])


# ---------------------------------------------------------------------------
# Policy base: invalid k
# ---------------------------------------------------------------------------

def test_policy_rejects_non_positive_k():
    with pytest.raises(ValueError):
        RandomPolicy(k=0)
    with pytest.raises(ValueError):
        PopularityPolicy(k=0)
    with pytest.raises(ValueError):
        ALSPolicy(k=0)
    with pytest.raises(ValueError):
        ItemKNNPolicy(k=0)
    with pytest.raises(ValueError):
        BPRPolicy(k=0)
    with pytest.raises(ValueError):
        LightGCNPolicy(k=0)


def test_policy_name_defaults_to_class_name():
    assert RandomPolicy(k=1).name == "RandomPolicy"
    assert ALSPolicy(k=1).name == "ALSPolicy"


def test_policy_name_override():
    policy = ALSPolicy(k=1, n_factors=32, name="als_f32")
    assert policy.name == "als_f32"
    assert ItemKNNPolicy(k=1, name="knn_10").name == "knn_10"
    assert BPRPolicy(k=1, name="bpr_a").name == "bpr_a"
    assert LightGCNPolicy(k=1, name="lgcn_a").name == "lgcn_a"


# ---------------------------------------------------------------------------
# RandomPolicy
# ---------------------------------------------------------------------------

class TestRandomPolicy:
    def setup_method(self):
        self.items = _items("i1", "i2", "i3", "i4", "i5")
        self.train = _interactions(
            ("u1", "i1"),
            ("u1", "i2"),
            ("u2", "i3"),
        )
        self.users = pd.DataFrame({"user_id": ["u1", "u2"]})

    def test_output_columns(self):
        recs = RandomPolicy(k=2, seed=0).fit(self.train, items=self.items).recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        assert set(recs.columns) == {"user_id", "item_id", "policy", "rank"}

    def test_at_most_k_per_user(self):
        recs = RandomPolicy(k=2, seed=0).fit(self.train, items=self.items).recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        for user_id in self.users["user_id"]:
            user_recs = recs[recs["user_id"] == user_id]
            assert len(user_recs) <= 2

    def test_excludes_train_items(self):
        recs = RandomPolicy(k=10, seed=0).fit(self.train, items=self.items).recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        u1_items = set(recs[recs["user_id"] == "u1"]["item_id"])
        assert "i1" not in u1_items
        assert "i2" not in u1_items
        u2_items = set(recs[recs["user_id"] == "u2"]["item_id"])
        assert "i3" not in u2_items

    def test_rank_starts_at_1(self):
        recs = RandomPolicy(k=3, seed=0).fit(self.train, items=self.items).recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        for user_id in self.users["user_id"]:
            user_recs = recs[recs["user_id"] == user_id].sort_values("rank")
            if not user_recs.empty:
                assert user_recs["rank"].iloc[0] == 1

    def test_policy_column_equals_class_name(self):
        recs = RandomPolicy(k=2, seed=0).fit(self.train, items=self.items).recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        assert (recs["policy"] == "RandomPolicy").all()

    def test_stable_across_calls(self):
        policy = RandomPolicy(k=3, seed=42).fit(self.train, items=self.items)
        r1 = policy.recommend(self.users, train_interactions=self.train, items=self.items)
        r2 = policy.recommend(self.users, train_interactions=self.train, items=self.items)
        pd.testing.assert_frame_equal(r1, r2)

    def test_user_with_all_items_seen_gets_no_recs(self):
        all_items = _items("i1", "i2")
        train = _interactions(("u1", "i1"), ("u1", "i2"))
        users = pd.DataFrame({"user_id": ["u1"]})
        recs = RandomPolicy(k=2, seed=0).fit(train, items=all_items).recommend(
            users, train_interactions=train, items=all_items
        )
        assert recs[recs["user_id"] == "u1"].empty


# ---------------------------------------------------------------------------
# PopularityPolicy
# ---------------------------------------------------------------------------

class TestPopularityPolicy:
    def setup_method(self):
        # i1: 2 interactions, i2: 1 interaction, i3: 0, i4: 3
        self.items = _items("i1", "i2", "i3", "i4", "i5")
        self.train = _interactions(
            ("u1", "i1"),
            ("u2", "i1"),
            ("u1", "i2"),
            ("u3", "i4"),
            ("u4", "i4"),
            ("u5", "i4"),
        )
        self.users = pd.DataFrame({"user_id": ["u1", "u2"]})

    def test_output_columns(self):
        recs = PopularityPolicy(k=3, seed=0).fit(self.train, items=self.items).recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        assert set(recs.columns) == {"user_id", "item_id", "policy", "rank"}

    def test_most_popular_unseen_is_rank1_for_u1(self):
        # u1 has seen i1, i2. Popularity order: i4(3) > i1(2) > i2(1) > i3=i5(0)
        # So for u1, first unseen in popularity order is i4.
        recs = PopularityPolicy(k=1, seed=0).fit(self.train, items=self.items).recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        u1_recs = recs[recs["user_id"] == "u1"]
        assert len(u1_recs) == 1
        assert u1_recs.iloc[0]["item_id"] == "i4"
        assert u1_recs.iloc[0]["rank"] == 1

    def test_excludes_train_items(self):
        recs = PopularityPolicy(k=10, seed=0).fit(self.train, items=self.items).recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        u1_items = set(recs[recs["user_id"] == "u1"]["item_id"])
        assert "i1" not in u1_items
        assert "i2" not in u1_items
        u2_items = set(recs[recs["user_id"] == "u2"]["item_id"])
        assert "i1" not in u2_items

    def test_at_most_k_per_user(self):
        recs = PopularityPolicy(k=2, seed=0).fit(self.train, items=self.items).recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        for user_id in self.users["user_id"]:
            assert len(recs[recs["user_id"] == user_id]) <= 2

    def test_policy_column_equals_class_name(self):
        recs = PopularityPolicy(k=2, seed=0).fit(self.train, items=self.items).recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        assert (recs["policy"] == "PopularityPolicy").all()

    def test_zero_interaction_items_included_as_fallback(self):
        # i3 has 0 interactions. For u1 with small k it may not be in top-k,
        # but with large k it must appear.
        recs = PopularityPolicy(k=10, seed=0).fit(self.train, items=self.items).recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        u1_items = set(recs[recs["user_id"] == "u1"]["item_id"])
        assert "i3" in u1_items or "i5" in u1_items


# ---------------------------------------------------------------------------
# ALSPolicy
# ---------------------------------------------------------------------------

class TestALSPolicy:
    def setup_method(self):
        # Strong CF signal: u2..u5 all interacted with A and C.
        # u1 has only seen A — ALS should place C at rank 1 for u1 since it
        # co-occurs with A across many users. B and D only appear together via u6.
        self.items = _items("A", "B", "C", "D")
        self.train = _interactions(
            ("u1", "A"),
            ("u2", "A"), ("u2", "C"),
            ("u3", "A"), ("u3", "C"),
            ("u4", "A"), ("u4", "C"),
            ("u5", "A"), ("u5", "C"),
            ("u6", "B"), ("u6", "D"),
        )
        self.users = pd.DataFrame({"user_id": ["u1", "u2"]})
        self.policy = ALSPolicy(k=3, n_factors=2, iterations=30, seed=0)

    def _fit(self):
        return self.policy.fit(self.train, items=self.items)

    def test_output_columns(self):
        recs = self._fit().recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        assert set(recs.columns) == {"user_id", "item_id", "policy", "rank"}

    def test_rank_1_is_highest_scored_item(self):
        recs = self._fit().recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        u1_rank1 = recs[(recs["user_id"] == "u1") & (recs["rank"] == 1)]
        assert len(u1_rank1) == 1
        assert u1_rank1.iloc[0]["item_id"] == "C"

    def test_excludes_seen_items(self):
        recs = self._fit().recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        u1_items = set(recs[recs["user_id"] == "u1"]["item_id"])
        assert "A" not in u1_items
        u2_items = set(recs[recs["user_id"] == "u2"]["item_id"])
        assert "A" not in u2_items
        assert "C" not in u2_items

    def test_at_most_k_per_user(self):
        recs = self._fit().recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        for user_id in self.users["user_id"]:
            assert len(recs[recs["user_id"] == user_id]) <= self.policy.k

    def test_policy_name(self):
        recs = self._fit().recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        assert (recs["policy"] == "ALSPolicy").all()

    def test_cold_start_user_gets_no_recs(self):
        unknown = pd.DataFrame({"user_id": ["unknown"]})
        recs = self._fit().recommend(
            unknown, train_interactions=self.train, items=self.items
        )
        assert recs.empty

    def test_deterministic(self):
        policy = self._fit()
        r1 = policy.recommend(self.users, train_interactions=self.train, items=self.items)
        r2 = policy.recommend(self.users, train_interactions=self.train, items=self.items)
        pd.testing.assert_frame_equal(r1, r2)

    def test_rejects_non_positive_n_factors(self):
        with pytest.raises(ValueError, match="n_factors"):
            ALSPolicy(k=5, n_factors=0)

    def test_rejects_non_positive_iterations(self):
        with pytest.raises(ValueError, match="iterations"):
            ALSPolicy(k=5, iterations=0)

    def test_regularization_and_alpha_exposed(self):
        policy = ALSPolicy(k=2, regularization=0.5, alpha=10.0)
        assert policy.regularization == 0.5
        assert policy.alpha == 10.0
        # Defaults reproduce the implicit library's own defaults.
        default = ALSPolicy(k=2)
        assert default.regularization == 0.01
        assert default.alpha == 1.0

    def test_rejects_invalid_regularization_and_alpha(self):
        with pytest.raises(ValueError, match="regularization"):
            ALSPolicy(k=5, regularization=0.0)
        with pytest.raises(ValueError, match="alpha"):
            ALSPolicy(k=5, alpha=-1.0)

    def test_fit_with_regularization_and_alpha(self):
        policy = ALSPolicy(
            k=3, n_factors=2, iterations=5, seed=0,
            regularization=0.5, alpha=10.0,
        )
        recs = policy.fit(self.train, items=self.items).recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        assert not recs.empty


# ---------------------------------------------------------------------------
# ItemKNNPolicy
# ---------------------------------------------------------------------------

class TestItemKNNPolicy:
    def setup_method(self):
        # Strong CF signal: u2..u5 all interacted with A and C.
        # u1 has only seen A — ItemKNN should place C at rank 1 for u1 because
        # A and C have very high cosine similarity (they co-occur across 4 users).
        # B and D only co-occur via u6, so they are dissimilar to A and C.
        self.items = _items("A", "B", "C", "D")
        self.train = _interactions(
            ("u1", "A"),
            ("u2", "A"), ("u2", "C"),
            ("u3", "A"), ("u3", "C"),
            ("u4", "A"), ("u4", "C"),
            ("u5", "A"), ("u5", "C"),
            ("u6", "B"), ("u6", "D"),
        )
        self.users = pd.DataFrame({"user_id": ["u1", "u2"]})
        self.policy = ItemKNNPolicy(k=3, n_neighbors=5, seed=0)

    def _fit(self):
        return self.policy.fit(self.train, items=self.items)

    def test_output_columns(self):
        recs = self._fit().recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        assert set(recs.columns) == {"user_id", "item_id", "policy", "rank"}

    def test_cf_signal_rank1(self):
        # u1 has only seen A; C co-occurs with A most strongly → C should be rank 1
        recs = self._fit().recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        u1_rank1 = recs[(recs["user_id"] == "u1") & (recs["rank"] == 1)]
        assert len(u1_rank1) == 1
        assert u1_rank1.iloc[0]["item_id"] == "C"

    def test_excludes_seen_items(self):
        recs = self._fit().recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        u1_items = set(recs[recs["user_id"] == "u1"]["item_id"])
        assert "A" not in u1_items
        u2_items = set(recs[recs["user_id"] == "u2"]["item_id"])
        assert "A" not in u2_items
        assert "C" not in u2_items

    def test_at_most_k_per_user(self):
        recs = self._fit().recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        for user_id in self.users["user_id"]:
            assert len(recs[recs["user_id"] == user_id]) <= self.policy.k

    def test_policy_name(self):
        recs = self._fit().recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        assert (recs["policy"] == "ItemKNNPolicy").all()

    def test_cold_start_user_gets_no_recs(self):
        unknown = pd.DataFrame({"user_id": ["unknown"]})
        recs = self._fit().recommend(
            unknown, train_interactions=self.train, items=self.items
        )
        assert recs.empty

    def test_rejects_non_positive_n_neighbors(self):
        with pytest.raises(ValueError, match="n_neighbors"):
            ItemKNNPolicy(k=5, n_neighbors=0)


# ---------------------------------------------------------------------------
# BPRPolicy
# ---------------------------------------------------------------------------

class TestBPRPolicy:
    def setup_method(self):
        # Same CF fixture as TestALSPolicy: u2..u5 co-interact with A and C,
        # u1 only with A. BPR should rank C highest for u1.
        self.items = _items("A", "B", "C", "D")
        self.train = _interactions(
            ("u1", "A"),
            ("u2", "A"), ("u2", "C"),
            ("u3", "A"), ("u3", "C"),
            ("u4", "A"), ("u4", "C"),
            ("u5", "A"), ("u5", "C"),
            ("u6", "B"), ("u6", "D"),
        )
        self.users = pd.DataFrame({"user_id": ["u1", "u2"]})
        # Higher learning_rate and more iterations for convergence on tiny data.
        self.policy = BPRPolicy(
            k=3, n_factors=4, learning_rate=0.05, iterations=300, seed=0
        )

    def _fit(self):
        return self.policy.fit(self.train, items=self.items)

    def test_output_columns(self):
        recs = self._fit().recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        assert set(recs.columns) == {"user_id", "item_id", "policy", "rank"}

    def test_cf_signal_rank1(self):
        recs = self._fit().recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        u1_rank1 = recs[(recs["user_id"] == "u1") & (recs["rank"] == 1)]
        assert len(u1_rank1) == 1
        assert u1_rank1.iloc[0]["item_id"] == "C"

    def test_excludes_seen_items(self):
        recs = self._fit().recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        u1_items = set(recs[recs["user_id"] == "u1"]["item_id"])
        assert "A" not in u1_items
        u2_items = set(recs[recs["user_id"] == "u2"]["item_id"])
        assert "A" not in u2_items
        assert "C" not in u2_items

    def test_at_most_k_per_user(self):
        recs = self._fit().recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        for user_id in self.users["user_id"]:
            assert len(recs[recs["user_id"] == user_id]) <= self.policy.k

    def test_policy_name(self):
        recs = self._fit().recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        assert (recs["policy"] == "BPRPolicy").all()

    def test_cold_start_user_gets_no_recs(self):
        unknown = pd.DataFrame({"user_id": ["unknown"]})
        recs = self._fit().recommend(
            unknown, train_interactions=self.train, items=self.items
        )
        assert recs.empty

    def test_deterministic(self):
        policy = self._fit()
        r1 = policy.recommend(self.users, train_interactions=self.train, items=self.items)
        r2 = policy.recommend(self.users, train_interactions=self.train, items=self.items)
        pd.testing.assert_frame_equal(r1, r2)

    def test_rejects_non_positive_n_factors(self):
        with pytest.raises(ValueError, match="n_factors"):
            BPRPolicy(k=5, n_factors=0)

    def test_rejects_non_positive_iterations(self):
        with pytest.raises(ValueError, match="iterations"):
            BPRPolicy(k=5, iterations=0)


# ---------------------------------------------------------------------------
# LightGCNPolicy
# ---------------------------------------------------------------------------

class TestLightGCNPolicy:
    def setup_method(self):
        # Same CF fixture as TestALSPolicy / TestBPRPolicy.
        # u2..u5 co-interact with A and C; u1 only with A; u6 with B and D.
        # LightGCN propagates neighborhood info so A and C share similar embeddings,
        # making C the top recommendation for u1.
        self.items = _items("A", "B", "C", "D")
        self.train = _interactions(
            ("u1", "A"),
            ("u2", "A"), ("u2", "C"),
            ("u3", "A"), ("u3", "C"),
            ("u4", "A"), ("u4", "C"),
            ("u5", "A"), ("u5", "C"),
            ("u6", "B"), ("u6", "D"),
        )
        self.users = pd.DataFrame({"user_id": ["u1", "u2"]})
        # Higher learning_rate and more iterations than defaults: GCN propagation
        # on a tiny 4-item graph needs more steps to produce a clear CF ranking.
        self.policy = LightGCNPolicy(
            k=3, n_factors=8, n_layers=2, learning_rate=0.05, iterations=500, seed=0
        )

    def _fit(self):
        return self.policy.fit(self.train, items=self.items)

    def test_output_columns(self):
        recs = self._fit().recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        assert set(recs.columns) == {"user_id", "item_id", "policy", "rank"}

    def test_cf_signal_rank1(self):
        recs = self._fit().recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        u1_rank1 = recs[(recs["user_id"] == "u1") & (recs["rank"] == 1)]
        assert len(u1_rank1) == 1
        assert u1_rank1.iloc[0]["item_id"] == "C"

    def test_excludes_seen_items(self):
        recs = self._fit().recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        u1_items = set(recs[recs["user_id"] == "u1"]["item_id"])
        assert "A" not in u1_items
        u2_items = set(recs[recs["user_id"] == "u2"]["item_id"])
        assert "A" not in u2_items
        assert "C" not in u2_items

    def test_at_most_k_per_user(self):
        recs = self._fit().recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        for user_id in self.users["user_id"]:
            assert len(recs[recs["user_id"] == user_id]) <= self.policy.k

    def test_policy_name(self):
        recs = self._fit().recommend(
            self.users, train_interactions=self.train, items=self.items
        )
        assert (recs["policy"] == "LightGCNPolicy").all()

    def test_cold_start_user_gets_no_recs(self):
        unknown = pd.DataFrame({"user_id": ["unknown"]})
        recs = self._fit().recommend(
            unknown, train_interactions=self.train, items=self.items
        )
        assert recs.empty

    def test_deterministic(self):
        policy = self._fit()
        r1 = policy.recommend(self.users, train_interactions=self.train, items=self.items)
        r2 = policy.recommend(self.users, train_interactions=self.train, items=self.items)
        pd.testing.assert_frame_equal(r1, r2)

    def test_rejects_non_positive_n_factors(self):
        with pytest.raises(ValueError, match="n_factors"):
            LightGCNPolicy(k=5, n_factors=0)

    def test_rejects_non_positive_n_layers(self):
        with pytest.raises(ValueError, match="n_layers"):
            LightGCNPolicy(k=5, n_layers=0)

    def test_rejects_non_positive_iterations(self):
        with pytest.raises(ValueError, match="iterations"):
            LightGCNPolicy(k=5, iterations=0)


# ---------------------------------------------------------------------------
# positive_feedback_column: training on a positive-preference signal
# ---------------------------------------------------------------------------

class TestPositiveFeedbackFilter:
    """Policies with `positive_feedback_column` train on positive rows only.

    The filtered fit frame of this fixture is exactly the shared CF fixture
    used by the per-policy test classes above (A/C co-occurrence), so the
    CF signal assertions mirror them. On top of that: B is consumed-but-
    disliked by u1 (positive for u6, so it is in the fitted item space and
    only the full-frame seen-mask can exclude it), E is never positive
    (no representation at all), and u7 has zero positive rows (cold start).
    """

    def setup_method(self):
        self.items = _items("A", "B", "C", "D", "E")
        self.train = _preference_interactions(
            ("u1", "A", 1),
            ("u2", "A", 1), ("u2", "C", 1),
            ("u3", "A", 1), ("u3", "C", 1),
            ("u4", "A", 1), ("u4", "C", 1),
            ("u5", "A", 1), ("u5", "C", 1),
            ("u6", "B", 1), ("u6", "D", 1),
            ("u1", "B", 0),
            ("u7", "E", 0),
        )
        self.users = pd.DataFrame({"user_id": ["u1", "u7"]})

    @staticmethod
    def _cf_policies():
        kwargs = {"positive_feedback_column": "feedback"}
        return [
            ALSPolicy(k=3, n_factors=2, iterations=30, seed=0, **kwargs),
            ItemKNNPolicy(k=3, n_neighbors=5, seed=0, **kwargs),
            BPRPolicy(
                k=3, n_factors=4, learning_rate=0.05, iterations=300, seed=0,
                **kwargs,
            ),
            LightGCNPolicy(
                k=3, n_factors=8, n_layers=2, learning_rate=0.05,
                iterations=500, seed=0, **kwargs,
            ),
        ]

    def _recommend(self, policy):
        return policy.fit(self.train, items=self.items).recommend(
            self.users, train_interactions=self.train, items=self.items
        )

    def test_cf_policies_recover_preference_signal(self):
        for policy in self._cf_policies():
            recs = self._recommend(policy)
            u1_rank1 = recs[(recs["user_id"] == "u1") & (recs["rank"] == 1)]
            assert len(u1_rank1) == 1, policy.name
            assert u1_rank1.iloc[0]["item_id"] == "C", policy.name

    def test_disliked_consumed_item_never_recommended(self):
        # B is disliked by u1 (filtered out of fit) but positive for u6, so it
        # has latent factors; only the full-frame seen-mask can exclude it.
        for policy in self._cf_policies():
            recs = self._recommend(policy)
            u1_items = set(recs[recs["user_id"] == "u1"]["item_id"])
            assert "B" not in u1_items, policy.name

    def test_negative_only_item_never_recommended(self):
        for policy in self._cf_policies():
            recs = self._recommend(policy)
            assert "E" not in set(recs["item_id"]), policy.name

    def test_zero_positive_user_is_cold_start(self):
        for policy in self._cf_policies():
            recs = self._recommend(policy)
            assert recs[recs["user_id"] == "u7"].empty, policy.name

    def test_fit_counts_recorded(self):
        for policy in self._cf_policies():
            policy.fit(self.train, items=self.items)
            assert policy.fit_row_count_ == 11, policy.name
            assert policy.fit_user_count_ == 6, policy.name

    def test_popularity_ranks_by_positive_counts(self):
        items = _items("i1", "i2", "i3")
        train = _preference_interactions(
            ("v1", "i1", 1),
            ("v2", "i1", 0),
            ("v3", "i1", 0),
            ("v4", "i2", 1),
            ("v5", "i2", 1),
        )
        users = pd.DataFrame({"user_id": ["v9"]})
        # Unfiltered: i1 (3 rows) beats i2 (2 rows).
        plain = PopularityPolicy(k=1, seed=0).fit(train, items=items).recommend(
            users, train_interactions=train, items=items
        )
        assert plain.iloc[0]["item_id"] == "i1"
        # Filtered: i2 (2 positives) beats i1 (1 positive).
        policy = PopularityPolicy(k=1, seed=0, positive_feedback_column="feedback")
        filtered = policy.fit(train, items=items).recommend(
            users, train_interactions=train, items=items
        )
        assert filtered.iloc[0]["item_id"] == "i2"
        assert policy.fit_row_count_ == 3
        assert policy.fit_user_count_ == 3

    def test_missing_feedback_column_raises(self):
        policy = PopularityPolicy(k=1, seed=0, positive_feedback_column="nope")
        with pytest.raises(ValueError, match="positive_feedback_column"):
            policy.fit(self.train, items=self.items)

    def test_non_binary_feedback_column_raises(self):
        bad = self.train.copy()
        bad.loc[0, "feedback"] = 2
        policy = PopularityPolicy(k=1, seed=0, positive_feedback_column="feedback")
        with pytest.raises(ValueError, match="binary"):
            policy.fit(bad, items=self.items)

    def test_no_filter_keeps_all_rows(self):
        policy = PopularityPolicy(k=1, seed=0).fit(self.train, items=self.items)
        assert policy.fit_row_count_ == len(self.train)
        assert policy.fit_user_count_ == 7
