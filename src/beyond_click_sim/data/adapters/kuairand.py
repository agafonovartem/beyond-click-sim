from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from beyond_click_sim.data.canonical import (
    CanonicalDataset,
    CanonicalManifest,
    ManifestTable,
    file_record,
    write_manifest,
)


KUAI_RAND_DATASET_PAGE = "https://kuairand.com/"
KUAI_RAND_DATA_RECORD = "https://zenodo.org/records/10439422"
KUAI_RAND_SEMANTIC_SUPPLEMENT_RECORD = "https://zenodo.org/records/18159199"

RAW_INTERACTION_COLUMNS = (
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "time_ms",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "duration_ms",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
    "is_rand",
    "tab",
)

BINARY_INTERACTION_COLUMNS = (
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "is_profile_enter",
    "is_rand",
)

CATEGORY_LEVELS = ("first", "second", "third", "fourth")
CATEGORY_COLUMNS = tuple(
    f"{level}_level_category_{suffix}"
    for level in CATEGORY_LEVELS
    for suffix in ("id", "name", "prob")
)

SOURCE_LOGS = (
    ("log_standard_4_08_to_4_21_pure.csv", "standard"),
    ("log_standard_4_22_to_5_08_pure.csv", "standard"),
    ("log_random_4_22_to_5_08_pure.csv", "random"),
)

_INTERACTIONS_SCHEMA = pa.schema(
    [
        ("interaction_id", pa.string()),
        ("user_id", pa.string()),
        ("item_id", pa.string()),
        ("event_type", pa.string()),
        ("exposure_policy", pa.string()),
        ("source_log", pa.string()),
        ("source_row", pa.int64()),
        ("date", pa.int64()),
        ("hourmin", pa.int64()),
        ("timestamp_ms", pa.int64()),
        ("is_click", pa.int8()),
        ("is_like", pa.int8()),
        ("is_follow", pa.int8()),
        ("is_comment", pa.int8()),
        ("is_forward", pa.int8()),
        ("is_hate", pa.int8()),
        ("long_view", pa.int8()),
        ("play_time_ms", pa.int64()),
        ("duration_ms", pa.int64()),
        ("profile_stay_time", pa.int64()),
        ("comment_stay_time", pa.int64()),
        ("is_profile_enter", pa.int8()),
        ("is_rand", pa.int8()),
        ("tab", pa.int64()),
        ("target_click", pa.int8()),
        ("target_long_view", pa.int8()),
    ]
)


class KuaiRandPureAdapter:
    """Canonicalize the public KuaiRand-Pure exposure logs and metadata.

    All released exposure rows are preserved. Date, scenario, policy, and task
    filtering belong to downstream task builders. The official semantic
    supplement is joined strictly by the Pure ``video_id``/``final_video_id``
    mapping and is never truncated during canonicalization.
    """

    name = "kuairand-pure"
    version = "v1"
    interaction_chunk_size = 250_000

    def materialize(self, raw_dir: Path, out_dir: Path) -> CanonicalDataset:
        raw_dir = raw_dir.expanduser().resolve()
        out_dir = out_dir.expanduser().resolve()
        data_dir = raw_dir / "data"
        supplement_dir = raw_dir / "supplement"

        users_source = data_dir / "user_features_pure.csv"
        items_source = data_dir / "video_features_basic_pure.csv"
        global_statistics_source = data_dir / "video_features_statistic_pure.csv"
        captions_source = supplement_dir / "kuairand_video_captions_pure.csv"
        categories_source = supplement_dir / "kuairand_video_categories_pure.csv"
        interaction_sources = [
            (data_dir / filename, policy) for filename, policy in SOURCE_LOGS
        ]
        required_paths = [
            users_source,
            items_source,
            captions_source,
            categories_source,
            *(path for path, _ in interaction_sources),
        ]
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(path)

        users = self._read_users(users_source)
        items, semantic_manifest = self._read_items(
            basic_path=items_source,
            captions_path=captions_source,
            categories_path=categories_source,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        users_path_out = out_dir / "users.parquet"
        items_path_out = out_dir / "items.parquet"
        interactions_path_out = out_dir / "interactions.parquet"
        manifest_path = out_dir / "manifest.json"

        users.to_parquet(users_path_out, index=False)
        items.to_parquet(items_path_out, index=False)
        interaction_manifest = self._write_interactions(
            sources=interaction_sources,
            path=interactions_path_out,
            known_user_ids=set(users["user_id"]),
            known_item_ids=set(items["item_id"]),
        )

        consumed_sources = [
            users_source,
            items_source,
            captions_source,
            categories_source,
            *(path for path, _ in interaction_sources),
        ]
        excluded_sources: list[dict[str, Any]] = []
        if global_statistics_source.exists():
            excluded_sources.append(
                {
                    **file_record(global_statistics_source),
                    "reason": (
                        "Released global behavioral aggregates are excluded from "
                        "canonical item features because their temporal support is "
                        "not suitable for leakage-controlled policy-shift tasks."
                    ),
                }
            )

        manifest = CanonicalManifest(
            dataset="kuairand-pure",
            adapter=self.name,
            version=self.version,
            raw_sources=[file_record(path) for path in consumed_sources],
            tables={
                "users": ManifestTable(path=users_path_out.name, rows=len(users)),
                "items": ManifestTable(path=items_path_out.name, rows=len(items)),
                "interactions": ManifestTable(
                    path=interactions_path_out.name,
                    rows=interaction_manifest["rows"],
                ),
            },
            id_policy={
                "user_id": "KuaiRand-Pure raw user_id as string.",
                "item_id": (
                    "KuaiRand-Pure raw video_id as string; the semantic supplement "
                    "uses the identical final_video_id domain."
                ),
                "interaction_id": (
                    "kuairand:{source CSV stem}:row:{1-based row number within file}"
                ),
            },
            standard_targets={
                "target_click": "Released binary is_click value for the exposure row.",
                "target_long_view": (
                    "Released binary long_view value for the exposure row."
                ),
            },
            caveats=[
                "All released exposure rows are preserved; no date, tab, policy, user, item, or task filter is applied.",
                "Random exposures were inserted into the ordinary feed; exposure_policy is not a before/after deployment label.",
                "KuaiRand-Pure standard logs omit ordinary-feed items outside the released intervention pool.",
                "The semantic supplement was released after the 2022 interaction logs and may overlap model pretraining data.",
                "Released global video behavioral statistics are not included as canonical item features.",
            ],
            extra_sections={
                "provenance": {
                    "official_dataset_page": KUAI_RAND_DATASET_PAGE,
                    "official_data_record": KUAI_RAND_DATA_RECORD,
                    "official_semantic_supplement_record": (
                        KUAI_RAND_SEMANTIC_SUPPLEMENT_RECORD
                    ),
                },
                "semantic_supplement": semantic_manifest,
                "exposure_logs": interaction_manifest["source_logs"],
                "signals": {
                    "exposure_rows": interaction_manifest["rows"],
                    "users_in_exposure_logs": interaction_manifest["users"],
                    "items_in_exposure_logs": interaction_manifest["items"],
                },
                "excluded_sources": excluded_sources,
            },
        )
        write_manifest(manifest_path, manifest)

        return CanonicalDataset(
            name="kuairand-pure",
            version=self.version,
            root=out_dir,
            users_path=users_path_out,
            items_path=items_path_out,
            interactions_path=interactions_path_out,
            manifest_path=manifest_path,
        )

    @staticmethod
    def _read_users(path: Path) -> pd.DataFrame:
        users = pd.read_csv(path, dtype={"user_id": "string"})
        if "user_id" not in users.columns:
            raise ValueError(f"Missing required user column in {path}: 'user_id'")
        KuaiRandPureAdapter._require_unique_nonempty_ids(
            users["user_id"],
            source_name=str(path),
            id_name="user_id",
        )
        users = users.rename(columns={"user_id": "raw_user_id"})
        users.insert(0, "user_id", users["raw_user_id"].astype("string"))
        return users

    @classmethod
    def _read_items(
        cls,
        *,
        basic_path: Path,
        captions_path: Path,
        categories_path: Path,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        basic = pd.read_csv(basic_path, dtype={"video_id": "string"})
        if "video_id" not in basic.columns:
            raise ValueError(
                f"Missing required item column in {basic_path}: 'video_id'"
            )
        cls._require_unique_nonempty_ids(
            basic["video_id"],
            source_name=str(basic_path),
            id_name="video_id",
        )
        basic = basic.rename(columns={"video_id": "raw_item_id"})
        basic.insert(0, "item_id", basic["raw_item_id"].astype("string"))

        captions = pd.read_csv(
            captions_path,
            dtype={
                "final_video_id": "string",
                "caption": "string",
                "show_cover_text": "string",
            },
        )
        required_caption_columns = {
            "final_video_id",
            "caption",
            "show_cover_text",
            "duration",
        }
        cls._require_columns(
            captions,
            required_caption_columns,
            source_name=str(captions_path),
        )
        cls._require_unique_nonempty_ids(
            captions["final_video_id"],
            source_name=str(captions_path),
            id_name="final_video_id",
        )
        captions = captions[
            ["final_video_id", "caption", "show_cover_text", "duration"]
        ].copy()
        captions["caption"] = cls._normalized_text(captions["caption"])
        captions["show_cover_text"] = cls._normalized_text(
            captions["show_cover_text"]
        )
        captions["description"] = captions["caption"].where(
            captions["caption"].ne(""),
            captions["show_cover_text"],
        )
        captions = captions.rename(
            columns={
                "final_video_id": "item_id",
                "duration": "caption_duration_ms",
            }
        )

        categories = pd.read_csv(
            categories_path,
            dtype={
                "final_video_id": "string",
                **{
                    f"{level}_level_category_name": "string"
                    for level in CATEGORY_LEVELS
                },
            },
        )
        cls._require_columns(
            categories,
            {"final_video_id", *CATEGORY_COLUMNS},
            source_name=str(categories_path),
        )
        cls._require_unique_nonempty_ids(
            categories["final_video_id"],
            source_name=str(categories_path),
            id_name="final_video_id",
        )
        categories = categories[
            ["final_video_id", *CATEGORY_COLUMNS]
        ].copy()
        for level in CATEGORY_LEVELS:
            name_column = f"{level}_level_category_name"
            categories[name_column] = cls._normalized_text(categories[name_column])
        categories["category_path"] = categories.apply(
            cls._category_path,
            axis=1,
        )
        categories = categories.rename(columns={"final_video_id": "item_id"})

        basic_ids = set(basic["item_id"])
        caption_ids = set(captions["item_id"])
        category_ids = set(categories["item_id"])
        cls._require_exact_id_match(
            basic_ids,
            caption_ids,
            left_name="video_features_basic_pure.csv",
            right_name="kuairand_video_captions_pure.csv",
        )
        cls._require_exact_id_match(
            basic_ids,
            category_ids,
            left_name="video_features_basic_pure.csv",
            right_name="kuairand_video_categories_pure.csv",
        )

        items = basic.merge(captions, on="item_id", how="left", validate="one_to_one")
        items = items.merge(
            categories,
            on="item_id",
            how="left",
            validate="one_to_one",
        )
        if len(items) != len(basic):
            raise RuntimeError("Strict KuaiRand item enrichment changed the item count")

        description_available = items["description"].ne("")
        category_available = items["category_path"].ne("")
        semantic_manifest = {
            "enabled": True,
            "join_key": "item_id == final_video_id",
            "validation": "strict_exact_id_match",
            "text_policy": (
                "Full normalized caption; fall back to full normalized "
                "show_cover_text; no truncation."
            ),
            "coverage": {
                "canonical_items": int(len(items)),
                "caption_nonblank": int(items["caption"].ne("").sum()),
                "cover_text_nonblank": int(
                    items["show_cover_text"].ne("").sum()
                ),
                "description_available": int(description_available.sum()),
                "description_missing": int((~description_available).sum()),
                "category_path_available": int(category_available.sum()),
                "category_path_missing": int((~category_available).sum()),
            },
            "sources": {
                "captions": file_record(captions_path),
                "categories": file_record(categories_path),
            },
            "official_record": KUAI_RAND_SEMANTIC_SUPPLEMENT_RECORD,
        }
        return items, semantic_manifest

    @classmethod
    def _write_interactions(
        cls,
        *,
        sources: list[tuple[Path, str]],
        path: Path,
        known_user_ids: set[str],
        known_item_ids: set[str],
    ) -> dict[str, Any]:
        writer = pq.ParquetWriter(path, _INTERACTIONS_SCHEMA)
        all_user_ids: set[str] = set()
        all_item_ids: set[str] = set()
        total_rows = 0
        source_manifests: list[dict[str, Any]] = []
        try:
            for source_path, policy in sources:
                source_summary = cls._write_interaction_source(
                    writer=writer,
                    source_path=source_path,
                    policy=policy,
                    all_user_ids=all_user_ids,
                    all_item_ids=all_item_ids,
                )
                total_rows += int(source_summary["rows"])
                source_manifests.append(source_summary)
        finally:
            writer.close()

        missing_users = all_user_ids - known_user_ids
        missing_items = all_item_ids - known_item_ids
        if missing_users or missing_items:
            raise ValueError(
                "Exposure logs contain IDs absent from canonical metadata. "
                f"Missing users: {cls._id_sample(missing_users)}; "
                f"missing items: {cls._id_sample(missing_items)}"
            )
        return {
            "rows": total_rows,
            "users": len(all_user_ids),
            "items": len(all_item_ids),
            "source_logs": source_manifests,
        }

    @classmethod
    def _write_interaction_source(
        cls,
        *,
        writer: pq.ParquetWriter,
        source_path: Path,
        policy: str,
        all_user_ids: set[str],
        all_item_ids: set[str],
    ) -> dict[str, Any]:
        header = pd.read_csv(source_path, nrows=0).columns.tolist()
        if header != list(RAW_INTERACTION_COLUMNS):
            raise ValueError(
                f"Unexpected interaction schema in {source_path}: {header}"
            )

        expected_is_rand = 1 if policy == "random" else 0
        source_log = source_path.stem
        source_rows = 0
        source_users: set[str] = set()
        source_items: set[str] = set()
        tab_counts: Counter[int] = Counter()
        date_min: int | None = None
        date_max: int | None = None

        for chunk in pd.read_csv(
            source_path,
            dtype={"user_id": "string", "video_id": "string"},
            chunksize=cls.interaction_chunk_size,
        ):
            cls._require_columns(
                chunk,
                set(RAW_INTERACTION_COLUMNS),
                source_name=str(source_path),
            )
            if chunk[list(RAW_INTERACTION_COLUMNS)].isna().any().any():
                raise ValueError(f"Missing interaction values in {source_path}")
            observed_is_rand = set(chunk["is_rand"].astype(int).unique())
            if observed_is_rand != {expected_is_rand}:
                raise ValueError(
                    f"Unexpected is_rand values in {source_path}: "
                    f"{sorted(observed_is_rand)}"
                )
            for column in BINARY_INTERACTION_COLUMNS:
                values = set(chunk[column].astype(int).unique())
                if not values <= {0, 1}:
                    raise ValueError(
                        f"Non-binary values in {source_path} column "
                        f"{column!r}: {sorted(values)}"
                    )

            n_rows = len(chunk)
            source_row_numbers = range(
                source_rows + 1,
                source_rows + n_rows + 1,
            )
            materialized = pd.DataFrame(
                {
                    "interaction_id": [
                        f"kuairand:{source_log}:row:{row_number:010d}"
                        for row_number in source_row_numbers
                    ],
                    "user_id": chunk["user_id"].astype("string"),
                    "item_id": chunk["video_id"].astype("string"),
                    "event_type": "exposure",
                    "exposure_policy": policy,
                    "source_log": source_log,
                    "source_row": list(source_row_numbers),
                    "date": chunk["date"].astype("int64"),
                    "hourmin": chunk["hourmin"].astype("int64"),
                    "timestamp_ms": chunk["time_ms"].astype("int64"),
                    "is_click": chunk["is_click"].astype("int8"),
                    "is_like": chunk["is_like"].astype("int8"),
                    "is_follow": chunk["is_follow"].astype("int8"),
                    "is_comment": chunk["is_comment"].astype("int8"),
                    "is_forward": chunk["is_forward"].astype("int8"),
                    "is_hate": chunk["is_hate"].astype("int8"),
                    "long_view": chunk["long_view"].astype("int8"),
                    "play_time_ms": chunk["play_time_ms"].astype("int64"),
                    "duration_ms": chunk["duration_ms"].astype("int64"),
                    "profile_stay_time": chunk["profile_stay_time"].astype(
                        "int64"
                    ),
                    "comment_stay_time": chunk["comment_stay_time"].astype(
                        "int64"
                    ),
                    "is_profile_enter": chunk["is_profile_enter"].astype("int8"),
                    "is_rand": chunk["is_rand"].astype("int8"),
                    "tab": chunk["tab"].astype("int64"),
                    "target_click": chunk["is_click"].astype("int8"),
                    "target_long_view": chunk["long_view"].astype("int8"),
                }
            )
            writer.write_table(
                pa.Table.from_pandas(
                    materialized,
                    schema=_INTERACTIONS_SCHEMA,
                    preserve_index=False,
                    safe=True,
                )
            )

            chunk_users = set(materialized["user_id"].astype(str))
            chunk_items = set(materialized["item_id"].astype(str))
            source_users.update(chunk_users)
            source_items.update(chunk_items)
            all_user_ids.update(chunk_users)
            all_item_ids.update(chunk_items)
            tab_counts.update(int(value) for value in materialized["tab"])
            chunk_date_min = int(materialized["date"].min())
            chunk_date_max = int(materialized["date"].max())
            date_min = (
                chunk_date_min if date_min is None else min(date_min, chunk_date_min)
            )
            date_max = (
                chunk_date_max if date_max is None else max(date_max, chunk_date_max)
            )
            source_rows += n_rows

        if source_rows == 0:
            raise ValueError(f"Empty KuaiRand exposure log: {source_path}")
        return {
            "path": str(source_path),
            "source_log": source_log,
            "exposure_policy": policy,
            "expected_is_rand": expected_is_rand,
            "rows": source_rows,
            "users": len(source_users),
            "items": len(source_items),
            "date_min": date_min,
            "date_max": date_max,
            "tab_counts": {
                str(key): int(value) for key, value in sorted(tab_counts.items())
            },
        }

    @staticmethod
    def _normalized_text(values: pd.Series) -> pd.Series:
        return values.fillna("").astype("string").str.replace(
            r"\s+",
            " ",
            regex=True,
        ).str.strip()

    @staticmethod
    def _category_path(row: pd.Series) -> str:
        values: list[str] = []
        for level in CATEGORY_LEVELS:
            value = str(row[f"{level}_level_category_name"]).strip()
            if not value or value.upper() in {"UNKNOWN", "NAN", "NONE"}:
                continue
            if value not in values:
                values.append(value)
        return " > ".join(values)

    @staticmethod
    def _require_columns(
        frame: pd.DataFrame,
        columns: set[str],
        *,
        source_name: str,
    ) -> None:
        missing = sorted(columns - set(frame.columns))
        if missing:
            raise ValueError(f"Missing columns in {source_name}: {missing}")

    @classmethod
    def _require_unique_nonempty_ids(
        cls,
        values: pd.Series,
        *,
        source_name: str,
        id_name: str,
    ) -> None:
        normalized = values.astype("string").str.strip()
        missing = normalized.isna() | normalized.eq("")
        if missing.any():
            raise ValueError(
                f"{source_name} contains {int(missing.sum())} missing {id_name} values"
            )
        duplicates = normalized[normalized.duplicated(keep=False)].drop_duplicates()
        if not duplicates.empty:
            raise ValueError(
                f"{source_name} contains duplicate {id_name} values: "
                f"{cls._id_sample(duplicates)}"
            )

    @classmethod
    def _require_exact_id_match(
        cls,
        left_ids: set[str],
        right_ids: set[str],
        *,
        left_name: str,
        right_name: str,
    ) -> None:
        missing = left_ids - right_ids
        extra = right_ids - left_ids
        if missing or extra:
            raise ValueError(
                f"{right_name} IDs must exactly match {left_name} IDs. "
                f"Missing: {cls._id_sample(missing)}; "
                f"extra: {cls._id_sample(extra)}"
            )

    @staticmethod
    def _id_sample(values: Any, limit: int = 5) -> list[str]:
        return sorted(str(value) for value in values)[:limit]
