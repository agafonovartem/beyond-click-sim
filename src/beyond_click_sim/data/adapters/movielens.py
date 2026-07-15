from __future__ import annotations

from pathlib import Path

import pandas as pd

from beyond_click_sim.data.canonical import (
    CanonicalDataset,
    CanonicalManifest,
    ManifestTable,
    file_record,
    write_manifest,
)


MOVIE_SUMMARY_COLUMN = "summary"
AGENT4REC_REPOSITORY = "https://github.com/LehengTHU/Agent4Rec"
AGENT4REC_REPOSITORY_COMMIT = "f4ee73e4fc686ef6177e9554d96ecf50add952b0"
AGENT4REC_MOVIE_SUMMARY_FILE_COMMIT = "9a08cc797032e0529f8799a134a6c560edc8db3e"


def default_movies_augmentation_path() -> Path:
    """Return the vendored Agent4Rec MovieLens augmentation file."""

    return (
        Path(__file__).resolve().parents[4]
        / "resources"
        / "agent4rec"
        / "ml-1m"
        / "movies_augmentation.csv"
    )


class MovieLens1MAdapter:
    """Canonicalize raw MovieLens-1M files.

    This adapter preserves all observed ratings and appends deterministic
    target columns. Filters, splits, and candidate sets belong to later task
    builders.
    """

    name = "movielens-1m"
    version = "v1"

    def materialize(
        self,
        raw_dir: Path,
        out_dir: Path,
        *,
        movies_augmentation_path: Path | None = default_movies_augmentation_path(),
    ) -> CanonicalDataset:
        raw_dir = raw_dir.expanduser().resolve()
        out_dir = out_dir.expanduser().resolve()
        movies_path = raw_dir / "movies.dat"
        ratings_path = raw_dir / "ratings.dat"
        users_path = raw_dir / "users.dat"
        for path in (movies_path, ratings_path, users_path):
            if not path.exists():
                raise FileNotFoundError(path)

        users = self._read_users(users_path)
        items = self._read_items(movies_path)
        items, movie_summary_manifest = self._add_movie_summaries(
            items,
            movies_augmentation_path,
        )
        interactions = self._read_interactions(ratings_path)

        out_dir.mkdir(parents=True, exist_ok=True)
        users_path_out = out_dir / "users.parquet"
        items_path_out = out_dir / "items.parquet"
        interactions_path_out = out_dir / "interactions.parquet"
        manifest_path = out_dir / "manifest.json"

        users.to_parquet(users_path_out, index=False)
        items.to_parquet(items_path_out, index=False)
        interactions.to_parquet(interactions_path_out, index=False)

        manifest = CanonicalManifest(
            dataset="ml-1m",
            adapter=self.name,
            version=self.version,
            raw_sources=[
                file_record(movies_path),
                file_record(ratings_path),
                file_record(users_path),
            ],
            tables={
                "users": ManifestTable(path=users_path_out.name, rows=len(users)),
                "items": ManifestTable(path=items_path_out.name, rows=len(items)),
                "interactions": ManifestTable(
                    path=interactions_path_out.name,
                    rows=len(interactions),
                ),
            },
            id_policy={
                "user_id": "MovieLens raw user id as string",
                "item_id": "MovieLens raw movie id as string",
                "interaction_id": "ml-1m:row:{1-based row number in ratings.dat}",
            },
            standard_targets={
                "target_interact": "1 for every observed rating row in interactions.parquet.",
                "target_like_ge4": "1 if rating >= 4 else 0.",
                "target_rating": "Raw MovieLens rating on the 1-5 scale.",
            },
            caveats=[
                "All observed ratings are preserved; no train/test split, candidate set, or row filter is applied.",
                "MovieLens-1M users may rate the same movie at most once in the canonical raw files.",
            ],
            extra_sections={
                "item_enrichment": {
                    "movie_summaries": movie_summary_manifest,
                }
            },
        )
        write_manifest(manifest_path, manifest)

        return CanonicalDataset(
            name="ml-1m",
            version=self.version,
            root=out_dir,
            users_path=users_path_out,
            items_path=items_path_out,
            interactions_path=interactions_path_out,
            manifest_path=manifest_path,
        )

    @staticmethod
    def _read_users(path: Path) -> pd.DataFrame:
        users = pd.read_csv(
            path,
            sep="::",
            engine="python",
            header=None,
            names=["raw_user_id", "gender", "age", "occupation", "zip_code"],
            encoding="latin-1",
            dtype={
                "raw_user_id": "string",
                "gender": "string",
                "age": "Int64",
                "occupation": "Int64",
                "zip_code": "string",
            },
        )
        users.insert(0, "user_id", users["raw_user_id"].astype("string"))
        return users[
            ["user_id", "raw_user_id", "gender", "age", "occupation", "zip_code"]
        ]

    @staticmethod
    def _read_items(path: Path) -> pd.DataFrame:
        items = pd.read_csv(
            path,
            sep="::",
            engine="python",
            header=None,
            names=["raw_item_id", "title", "genres"],
            encoding="latin-1",
            dtype={"raw_item_id": "string", "title": "string", "genres": "string"},
        )
        items.insert(0, "item_id", items["raw_item_id"].astype("string"))
        return items[["item_id", "raw_item_id", "title", "genres"]]

    @classmethod
    def _add_movie_summaries(
        cls,
        items: pd.DataFrame,
        source_path: Path | None,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        if source_path is None:
            return items, {
                "enabled": False,
                "column": None,
                "source": None,
            }

        source_path = source_path.expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        summaries = pd.read_csv(
            source_path,
            usecols=["movie_id", MOVIE_SUMMARY_COLUMN],
            dtype={"movie_id": "string", MOVIE_SUMMARY_COLUMN: "string"},
        )
        summaries["movie_id"] = summaries["movie_id"].str.strip()
        summaries[MOVIE_SUMMARY_COLUMN] = summaries[MOVIE_SUMMARY_COLUMN].str.strip()

        missing_id_rows = summaries["movie_id"].isna() | summaries["movie_id"].eq("")
        if missing_id_rows.any():
            raise ValueError(
                "Movie summary source contains rows with a missing movie_id: "
                f"{int(missing_id_rows.sum())}"
            )

        duplicate_ids = summaries.loc[
            summaries["movie_id"].duplicated(keep=False),
            "movie_id",
        ].drop_duplicates()
        if not duplicate_ids.empty:
            raise ValueError(
                "Movie summary source contains duplicate movie_id values: "
                f"{cls._id_sample(duplicate_ids)}"
            )

        blank_summaries = summaries[MOVIE_SUMMARY_COLUMN].isna() | summaries[
            MOVIE_SUMMARY_COLUMN
        ].eq("")
        if blank_summaries.any():
            blank_ids = summaries.loc[blank_summaries, "movie_id"]
            raise ValueError(
                "Movie summary source contains blank summaries for movie_id values: "
                f"{cls._id_sample(blank_ids)}"
            )

        canonical_ids = set(items["item_id"].astype("string"))
        summary_ids = set(summaries["movie_id"])
        missing_ids = canonical_ids - summary_ids
        extra_ids = summary_ids - canonical_ids
        if missing_ids or extra_ids:
            raise ValueError(
                "Movie summary IDs must exactly match canonical MovieLens item IDs. "
                f"Missing: {cls._id_sample(missing_ids)}; "
                f"extra: {cls._id_sample(extra_ids)}"
            )

        summary_by_id = summaries.set_index("movie_id")[MOVIE_SUMMARY_COLUMN]
        enriched = items.copy()
        enriched[MOVIE_SUMMARY_COLUMN] = enriched["item_id"].map(summary_by_id)
        if enriched[MOVIE_SUMMARY_COLUMN].isna().any():
            raise RuntimeError("Validated movie summaries produced missing joined values")

        return enriched, {
            "enabled": True,
            "column": MOVIE_SUMMARY_COLUMN,
            "join_key": "item_id == movie_id",
            "validation": "strict_exact_id_match",
            "coverage": {
                "canonical_items": int(len(items)),
                "summary_rows": int(len(summaries)),
                "matched_items": int(enriched[MOVIE_SUMMARY_COLUMN].notna().sum()),
            },
            "source": {
                **file_record(source_path),
                "repository": AGENT4REC_REPOSITORY,
                "repository_commit": AGENT4REC_REPOSITORY_COMMIT,
                "source_file_commit": AGENT4REC_MOVIE_SUMMARY_FILE_COMMIT,
                "license": "MIT",
                "consumed_columns": ["movie_id", MOVIE_SUMMARY_COLUMN],
            },
        }

    @staticmethod
    def _id_sample(values: object, limit: int = 5) -> list[str]:
        return sorted(str(value) for value in values)[:limit]

    @staticmethod
    def _read_interactions(path: Path) -> pd.DataFrame:
        interactions = pd.read_csv(
            path,
            sep="::",
            engine="python",
            header=None,
            names=["user_id", "item_id", "rating", "timestamp"],
            encoding="latin-1",
            dtype={
                "user_id": "string",
                "item_id": "string",
                "rating": "Int64",
                "timestamp": "Int64",
            },
        )
        interactions.insert(
            0,
            "interaction_id",
            [f"ml-1m:row:{row_number:010d}" for row_number in range(1, len(interactions) + 1)],
        )
        interactions["event_type"] = "rating"
        interactions["target_interact"] = pd.Series(
            1, index=interactions.index, dtype="int8"
        )
        interactions["target_like_ge4"] = interactions["rating"].ge(4).astype("int8")
        interactions["target_rating"] = interactions["rating"]
        return interactions[
            [
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
        ]
