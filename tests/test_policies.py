from __future__ import annotations

import pandas as pd
import pytest

from beyond_click_sim.tasks.policies import ALSPolicy, PopularityPolicy, RandomPolicy


def _items(*ids):
    return pd.DataFrame({"item_id": list(ids)})


def _interactions(*rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id"])


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
