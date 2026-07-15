from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from beyond_click_sim.data.adapters.movielens import (
    AGENT4REC_MOVIE_SUMMARY_FILE_COMMIT,
    AGENT4REC_REPOSITORY,
    AGENT4REC_REPOSITORY_COMMIT,
    MovieLens1MAdapter,
)


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

    dataset = MovieLens1MAdapter().materialize(
        raw_dir,
        out_dir,
        movies_augmentation_path=None,
    )

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
        "target_interact",
        "target_like_ge4",
        "target_rating",
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
    assert interactions["target_interact"].tolist() == [1, 1, 1]
    assert interactions["target_like_ge4"].tolist() == [1, 0, 1]
    assert interactions["target_rating"].tolist() == [5, 2, 4]
    assert "part" not in interactions.columns
    assert "feedback_label" not in interactions.columns

    manifest = json.loads(dataset.manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"] == "ml-1m"
    assert manifest["schema_version"] == "canonical-v1"
    assert manifest["tables"]["users"]["rows"] == 2
    assert manifest["tables"]["items"]["rows"] == 2
    assert manifest["tables"]["interactions"]["rows"] == 3
    assert manifest["standard_targets"] == {
        "target_interact": "1 for every observed rating row in interactions.parquet.",
        "target_like_ge4": "1 if rating >= 4 else 0.",
        "target_rating": "Raw MovieLens rating on the 1-5 scale.",
    }
    assert len(manifest["raw_sources"]) == 3
    assert manifest["item_enrichment"] == {
        "movie_summaries": {
            "column": None,
            "enabled": False,
            "source": None,
        }
    }


def test_movielens_adapter_adds_strict_movie_summary_enrichment(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw" / "ml-1m"
    out_dir = tmp_path / "canonical" / "ml-1m" / "v1"
    summary_path = tmp_path / "movies_augmentation.csv"
    write_fixture(raw_dir)
    summary_path.write_text(
        "movie_id,title,genres,rating,summary\n"
        "1,Wrong title,Wrong genre,1.0,Toys plan a rescue.\n"
        "2,Also wrong,Wrong genre,2.0,A magical board game.\n",
        encoding="utf-8",
    )

    dataset = MovieLens1MAdapter().materialize(
        raw_dir,
        out_dir,
        movies_augmentation_path=summary_path,
    )

    items = pd.read_parquet(dataset.items_path)
    assert list(items.columns) == [
        "item_id",
        "raw_item_id",
        "title",
        "genres",
        "summary",
    ]
    assert items["title"].tolist() == ["Toy Story (1995)", "Jumanji (1995)"]
    assert items["summary"].tolist() == [
        "Toys plan a rescue.",
        "A magical board game.",
    ]

    manifest = dataset.load_manifest()
    enrichment = manifest["item_enrichment"]["movie_summaries"]
    assert enrichment["enabled"] is True
    assert enrichment["column"] == "summary"
    assert enrichment["join_key"] == "item_id == movie_id"
    assert enrichment["validation"] == "strict_exact_id_match"
    assert enrichment["coverage"] == {
        "canonical_items": 2,
        "summary_rows": 2,
        "matched_items": 2,
    }
    assert enrichment["source"]["repository"] == AGENT4REC_REPOSITORY
    assert enrichment["source"]["repository_commit"] == AGENT4REC_REPOSITORY_COMMIT
    assert (
        enrichment["source"]["source_file_commit"]
        == AGENT4REC_MOVIE_SUMMARY_FILE_COMMIT
    )
    assert enrichment["source"]["consumed_columns"] == ["movie_id", "summary"]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            "1,First summary\n1,Duplicate summary\n2,Second summary\n",
            "duplicate movie_id",
        ),
        ("1,First summary\n", "must exactly match"),
        ("1,First summary\n2,\n", "blank summaries"),
        (
            "1,First summary\n2,Second summary\n3,Extra summary\n",
            "must exactly match",
        ),
    ],
)
def test_movielens_adapter_rejects_invalid_movie_summary_coverage(
    tmp_path: Path,
    rows: str,
    message: str,
) -> None:
    raw_dir = tmp_path / "raw" / "ml-1m"
    out_dir = tmp_path / "canonical" / "ml-1m" / "v1"
    summary_path = tmp_path / "movies_augmentation.csv"
    write_fixture(raw_dir)
    summary_path.write_text("movie_id,summary\n" + rows, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        MovieLens1MAdapter().materialize(
            raw_dir,
            out_dir,
            movies_augmentation_path=summary_path,
        )
