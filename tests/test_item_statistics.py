from __future__ import annotations

import pandas as pd

from beyond_click_sim.tasks import TrainItemRatingStatistics


def test_train_item_rating_statistics_uses_train_only_values() -> None:
    items = pd.DataFrame({"item_id": ["i1", "i2", "i3"]})
    train_interactions = pd.DataFrame(
        {
            "item_id": ["i1", "i1", "i2", "i2"],
            "rating": [5, 3, pd.NA, pd.NA],
        }
    )

    enriched, manifest = TrainItemRatingStatistics().enrich_items(
        items=items,
        train_interactions=train_interactions,
        item_column="item_id",
    )

    by_item = enriched.set_index("item_id")
    assert by_item.loc["i1", "rating_mean"] == 4.0
    assert by_item.loc["i1", "rating_count"] == 2
    assert pd.isna(by_item.loc["i2", "rating_mean"])
    assert by_item.loc["i2", "rating_count"] == 0
    assert pd.isna(by_item.loc["i3", "rating_mean"])
    assert by_item.loc["i3", "rating_count"] == 0

    assert manifest["source"] == "train_split_only"
    assert manifest["value_column"] == "rating"
    assert manifest["mean_column"] == "rating_mean"
    assert manifest["count_column"] == "rating_count"
    assert manifest["missing_policy"] == {
        "rating_mean": "nan",
        "rating_count": 0,
    }
    assert manifest["items_with_statistics"] == 1
    assert manifest["items_total"] == 3
