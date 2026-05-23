from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from beyond_click_sim.data.adapters.movielens import MovieLens1MAdapter


def write_fixture(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True)
    (raw_dir / "users.dat").write_text(
        "\n".join(
            [
                "1::F::1::10::48067",
                "2::M::56::16::70072",
            ]
        )
        + "\n",
        encoding="latin-1",
    )
    (raw_dir / "movies.dat").write_text(
        "\n".join(
            [
                "1::Toy Story (1995)::Animation|Children's|Comedy",
                "2::Jumanji (1995)::Adventure|Children's|Fantasy",
            ]
        )
        + "\n",
        encoding="latin-1",
    )
    (raw_dir / "ratings.dat").write_text(
        "\n".join(
            [
                "1::1::5::978300760",
                "1::2::2::978302109",
                "2::2::4::978301968",
            ]
        )
        + "\n",
        encoding="latin-1",
    )


def test_movielens_adapter_materializes_canonical_tables(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "ml-1m"
    out_dir = tmp_path / "canonical" / "ml-1m" / "v1"
    write_fixture(raw_dir)

    dataset = MovieLens1MAdapter().materialize(raw_dir, out_dir)

    assert dataset.users_path.exists()
    assert dataset.items_path.exists()
    assert dataset.interactions_path.exists()
    assert dataset.manifest_path.exists()

    users = pd.read_parquet(dataset.users_path)
    items = pd.read_parquet(dataset.items_path)
    interactions = pd.read_parquet(dataset.interactions_path)

    assert list(users.columns) == [
        "user_id",
        "raw_user_id",
        "gender",
        "age",
        "occupation",
        "zip_code",
    ]
    assert list(items.columns) == ["item_id", "raw_item_id", "title", "genres"]
    assert list(interactions.columns) == [
        "interaction_id",
        "user_id",
        "item_id",
        "event_type",
        "rating",
        "timestamp",
    ]
    assert len(users) == 2
    assert len(items) == 2
    assert len(interactions) == 3
    assert interactions["interaction_id"].tolist() == [
        "ml-1m:row:0000000001",
        "ml-1m:row:0000000002",
        "ml-1m:row:0000000003",
    ]
    assert set(interactions["rating"]) == {2, 4, 5}
    assert set(interactions["event_type"]) == {"rating"}
    assert "part" not in interactions.columns
    assert "feedback_label" not in interactions.columns

    manifest = json.loads(dataset.manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"] == "ml-1m"
    assert manifest["schema_version"] == "canonical-v1"
    assert manifest["tables"]["users"]["rows"] == 2
    assert manifest["tables"]["items"]["rows"] == 2
    assert manifest["tables"]["interactions"]["rows"] == 3
    assert len(manifest["raw_sources"]) == 3
