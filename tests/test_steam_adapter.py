from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from beyond_click_sim.data.adapters.steam import SteamAdapter


def write_fixture(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True)
    user_rows = [
        {
            "user_id": "alice",
            "items_count": 2,
            "steam_id": "111",
            "user_url": "http://steamcommunity.com/id/alice",
            "items": [
                {
                    "item_id": "10",
                    "item_name": "Library Game A",
                    "playtime_forever": 1,
                    "playtime_2weeks": 0,
                },
                {
                    "item_id": "20",
                    "item_name": "Library Game B",
                    "playtime_forever": 0,
                    "playtime_2weeks": 0,
                },
            ],
        },
        {
            "user_id": "alice",
            "items_count": 2,
            "steam_id": "111",
            "user_url": "http://steamcommunity.com/id/alice",
            "items": [
                {
                    "item_id": "10",
                    "item_name": "Library Game A",
                    "playtime_forever": 10,
                    "playtime_2weeks": 3,
                },
                {
                    "item_id": "20",
                    "item_name": "Library Game B",
                    "playtime_forever": 0,
                    "playtime_2weeks": 0,
                },
            ],
        },
        {
            "user_id": "bob",
            "items_count": 1,
            "steam_id": "222",
            "user_url": "http://steamcommunity.com/id/bob",
            "items": [
                {
                    "item_id": "30",
                    "item_name": "Library Game C",
                    "playtime_forever": 130,
                    "playtime_2weeks": 1,
                }
            ],
        },
        {
            "user_id": "vanity",
            "items_count": 0,
            "steam_id": "333",
            "user_url": "http://steamcommunity.com/id/vanity",
            "items": [],
        },
        {
            "user_id": "333",
            "items_count": 0,
            "steam_id": "333",
            "user_url": "http://steamcommunity.com/profiles/333",
            "items": [],
        },
    ]
    (raw_dir / "australian_users_items.json").write_text(
        "".join(f"{row!r}\n" for row in user_rows),
        encoding="utf-8",
    )

    game_rows = [
        {
            "publisher": "Pub A",
            "genres": ["Action", "Indie"],
            "app_name": "Metadata Game A",
            "title": "Metadata Title A",
            "url": "http://store.steampowered.com/app/10/Game_A",
            "release_date": "2018-01-01",
            "tags": ["Action"],
            "reviews_url": "http://steamcommunity.com/app/10/reviews/",
            "specs": ["Single-player"],
            "price": 4.99,
            "early_access": False,
            "id": "10",
            "developer": "Dev A",
        },
        {
            "publisher": "Pub B",
            "genres": ["Adventure"],
            "app_name": "Metadata Game B",
            "title": "Metadata Title B",
            "url": "http://store.steampowered.com/app/20/Game_B",
            "release_date": "2019-01-01",
            "tags": ["Adventure"],
            "reviews_url": "http://steamcommunity.com/app/20/reviews/",
            "specs": ["Single-player"],
            "price": "Free To Play",
            "early_access": False,
            "id": "20",
            "developer": "Dev B",
        },
    ]
    with gzip.open(raw_dir / "steam_games.json.gz", "wt", encoding="utf-8") as file:
        for row in game_rows:
            file.write(f"{row!r}\n")


def test_steam_adapter_materializes_canonical_tables(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "steam"
    out_dir = tmp_path / "canonical" / "steam" / "v1"
    write_fixture(raw_dir)

    dataset = SteamAdapter().materialize(raw_dir, out_dir)

    assert dataset.users_path.exists()
    assert dataset.items_path.exists()
    assert dataset.interactions_path.exists()
    assert dataset.manifest_path.exists()

    users = pd.read_parquet(dataset.users_path)
    items = pd.read_parquet(dataset.items_path)
    interactions = pd.read_parquet(dataset.interactions_path)

    assert list(users.columns) == [
        "user_id",
        "steam_id",
        "raw_user_id",
        "user_url",
        "reported_items_count",
        "source_line",
        "raw_snapshot_count",
    ]
    assert list(interactions.columns) == [
        "interaction_id",
        "user_id",
        "item_id",
        "event_type",
        "playtime_forever",
        "playtime_2weeks",
        "source_user_line",
        "target_interact",
        "target_played_120",
        "target_playtime",
    ]

    assert len(users) == 3
    assert len(items) == 3
    assert len(interactions) == 3

    selected_alice = users.loc[users["steam_id"].eq("111")].iloc[0]
    assert selected_alice["user_id"] == "steam:111"
    assert selected_alice["source_line"] == 2
    assert selected_alice["raw_snapshot_count"] == 2

    assert interactions["interaction_id"].tolist() == [
        "steam:line:00000002:item:10",
        "steam:line:00000002:item:20",
        "steam:line:00000003:item:30",
    ]
    assert set(interactions["event_type"]) == {"owned"}
    assert set(interactions["playtime_forever"]) == {0, 10, 130}
    assert interactions["target_interact"].tolist() == [1, 1, 1]
    assert interactions["target_played_120"].tolist() == [0, 0, 1]
    assert interactions["target_playtime"].tolist() == [10, 0, 130]
    assert "part" not in interactions.columns
    assert "feedback_label" not in interactions.columns

    item_10 = items.loc[items["item_id"].eq("10")].iloc[0]
    item_30 = items.loc[items["item_id"].eq("30")].iloc[0]
    assert item_10["title"] == "Library Game A"
    assert item_10["metadata_available"]
    assert item_10["metadata_title"] == "Metadata Title A"
    assert not item_30["metadata_available"]
    assert item_30["title"] == "Library Game C"

    manifest = json.loads(dataset.manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"] == "steam"
    assert manifest["schema_version"] == "canonical-v1"
    assert manifest["tables"]["users"]["rows"] == 3
    assert manifest["tables"]["items"]["rows"] == 3
    assert manifest["tables"]["interactions"]["rows"] == 3
    assert manifest["standard_targets"] == {
        "target_interact": "1 for every observed owned-library row in interactions.parquet.",
        "target_played_120": "1 if playtime_forever >= 120 minutes else 0.",
        "target_playtime": "Raw playtime_forever in minutes.",
    }
    assert manifest["deduplication"]["duplicate_steam_id_groups"] == 2
    assert manifest["deduplication"]["duplicate_steam_id_surplus_rows"] == 2
    assert manifest["deduplication"]["duplicate_steam_id_group_size_counts"] == {"2": 2}
    assert manifest["signals"] == {
        "raw_user_rows": 5,
        "raw_owned_rows": 5,
        "owned_rows_after_user_snapshot_dedup": 3,
    }
    assert manifest["metadata"]["library_items_with_metadata"] == 2
    assert manifest["metadata"]["library_items_without_metadata"] == 1
