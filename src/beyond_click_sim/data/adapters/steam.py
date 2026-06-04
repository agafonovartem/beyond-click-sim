from __future__ import annotations

import ast
import gzip
import json
from collections import Counter, defaultdict
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


_INTERACTIONS_SCHEMA = pa.schema(
    [
        ("interaction_id", pa.string()),  # Canonical row id: raw user line + raw item_id.
        ("user_id", pa.string()),  # Canonical user id: steam:{steam_id}.
        ("item_id", pa.string()),  # Raw Steam item_id from the nested items list.
        ("event_type", pa.string()),  # Constant "owned"; Steam raw is a library snapshot.
        ("playtime_forever", pa.int64()),  # Raw total playtime in minutes.
        ("playtime_2weeks", pa.int64()),  # Raw recent playtime in minutes.
        ("source_user_line", pa.int64()),  # 1-based line in australian_users_items.json.
        ("target_interact", pa.int8()),  # 1 for every observed ownership row.
        ("target_played_120", pa.int8()),  # 1 if total playtime is at least 120 minutes.
        ("target_playtime", pa.int64()),  # Raw total playtime target in minutes.
    ]
)


class SteamAdapter:
    """Canonicalize the Steam user-library snapshot dataset."""

    name = "steam"
    version = "v1"

    def materialize(self, raw_dir: Path, out_dir: Path) -> CanonicalDataset:
        raw_dir = raw_dir.expanduser().resolve()
        out_dir = out_dir.expanduser().resolve()
        users_items_path = raw_dir / "australian_users_items.json"
        games_path = raw_dir / "steam_games.json.gz"
        for path in (users_items_path, games_path):
            if not path.exists():
                raise FileNotFoundError(path)

        first_pass = self._summarize_user_snapshots(users_items_path)
        metadata, metadata_stats = self._read_game_metadata(games_path)

        out_dir.mkdir(parents=True, exist_ok=True)
        users_path_out = out_dir / "users.parquet"
        items_path_out = out_dir / "items.parquet"
        interactions_path_out = out_dir / "interactions.parquet"
        manifest_path = out_dir / "manifest.json"

        materialized = self._write_selected_snapshots(
            users_items_path=users_items_path,
            selected_by_steam_id=first_pass["selected_by_steam_id"],
            snapshot_counts=first_pass["snapshot_counts"],
            metadata=metadata,
            users_path=users_path_out,
            items_path=items_path_out,
            interactions_path=interactions_path_out,
        )

        manifest = CanonicalManifest(
            dataset="steam",
            adapter=self.name,
            version=self.version,
            raw_sources=[
                file_record(users_items_path),
                file_record(games_path),
            ],
            tables={
                "users": ManifestTable(
                    path=users_path_out.name,
                    rows=materialized["users_rows"],
                ),
                "items": ManifestTable(
                    path=items_path_out.name,
                    rows=materialized["items_rows"],
                ),
                "interactions": ManifestTable(
                    path=interactions_path_out.name,
                    rows=materialized["interactions_rows"],
                ),
            },
            id_policy={
                "user_id": "steam:{steam_id}; one selected library snapshot per Steam account",
                "item_id": "Steam raw item_id as string",
                "interaction_id": "steam:line:{source user line}:item:{item_id}",
            },
            standard_targets={
                "target_interact": "1 for every observed owned-library row in interactions.parquet.",
                "target_played_120": "1 if playtime_forever >= 120 minutes else 0.",
                "target_playtime": "Raw playtime_forever in minutes.",
            },
            caveats=[
                "Steam raw data is a user-library snapshot dataset, not a chronological event log.",
                "Canonical interactions represent observed ownership/library entries.",
                "Zero-playtime owned items are preserved.",
                "No train/test split, candidate set, or row filter is applied.",
                "Some library items have no matching row in steam_games.json.gz metadata.",
            ],
            extra_sections={
                "deduplication": {
                    "user_key": "steam_id",
                    "policy": (
                        "Choose the snapshot with maximum total_playtime_forever; "
                        "break ties by earliest source line."
                    ),
                    **first_pass["dedup_stats"],
                },
                "metadata": {
                    **metadata_stats,
                    "library_items_with_metadata": materialized[
                        "library_items_with_metadata"
                    ],
                    "library_items_without_metadata": materialized[
                        "library_items_without_metadata"
                    ],
                },
                "signals": {
                    "raw_user_rows": first_pass["raw_user_rows"],
                    "raw_owned_rows": first_pass["raw_owned_rows"],
                    "owned_rows_after_user_snapshot_dedup": materialized[
                        "interactions_rows"
                    ],
                },
            },
        )
        write_manifest(manifest_path, manifest)

        return CanonicalDataset(
            name="steam",
            version=self.version,
            root=out_dir,
            users_path=users_path_out,
            items_path=items_path_out,
            interactions_path=interactions_path_out,
            manifest_path=manifest_path,
        )

    @classmethod
    def _summarize_user_snapshots(cls, path: Path) -> dict[str, Any]:
        selected_by_steam_id: dict[str, dict[str, Any]] = {}
        snapshots_by_steam_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        raw_user_ids: Counter[str] = Counter()
        raw_user_rows = 0
        raw_owned_rows = 0

        for line_no, row in cls._iter_user_rows(path):
            summary = cls._snapshot_summary(line_no, row)
            steam_id = summary["steam_id"]
            raw_user_rows += 1
            raw_owned_rows += summary["observed_items_count"]
            raw_user_ids[summary["raw_user_id"]] += 1

            snapshots_by_steam_id[steam_id].append(summary)
            previous = selected_by_steam_id.get(steam_id)
            if previous is None or cls._is_better_snapshot(summary, previous):
                selected_by_steam_id[steam_id] = summary

        # Number of raw snapshots per Steam account; used in users.parquet for traceability.
        snapshot_counts = {
            steam_id: len(snapshots)
            for steam_id, snapshots in snapshots_by_steam_id.items()
        }
        # Duplicate groups under the canonical user key. These are the rows we deduplicate.
        duplicate_groups = {
            steam_id: snapshots
            for steam_id, snapshots in snapshots_by_steam_id.items()
            if len(snapshots) > 1
        }
        # Diagnostic only: raw user_id can be a vanity name, so it is not the dedup key.
        duplicate_raw_user_groups = {
            raw_user_id: count for raw_user_id, count in raw_user_ids.items() if count > 1
        }
        return {
            # Deduplicated: steam_id -> snapshot selected by _is_better_snapshot.
            "selected_by_steam_id": selected_by_steam_id,
            # Before dedup: steam_id -> number of raw snapshots.
            "snapshot_counts": snapshot_counts,
            "raw_user_rows": raw_user_rows,
            "raw_owned_rows": raw_owned_rows,
            "dedup_stats": {
                "unique_steam_id": len(snapshots_by_steam_id),
                "unique_raw_user_id": len(raw_user_ids),
                "duplicate_steam_id_groups": len(duplicate_groups),
                "duplicate_steam_id_surplus_rows": raw_user_rows
                - len(snapshots_by_steam_id),
                "duplicate_steam_id_group_size_counts": dict(
                    sorted(
                        Counter(
                            len(snapshots) for snapshots in duplicate_groups.values()
                        ).items()
                    )
                ),
                "duplicate_raw_user_id_groups": len(duplicate_raw_user_groups),
                "duplicate_raw_user_id_surplus_rows": raw_user_rows
                - len(raw_user_ids),
            },
        }

    @classmethod
    def _write_selected_snapshots(
        cls,
        *,
        users_items_path: Path,
        selected_by_steam_id: dict[str, dict[str, Any]],
        snapshot_counts: dict[str, int],
        metadata: dict[str, dict[str, Any]],
        users_path: Path,
        items_path: Path,
        interactions_path: Path,
    ) -> dict[str, int]:
        selected_source_lines = {
            snapshot["source_line"]: steam_id
            for steam_id, snapshot in selected_by_steam_id.items()
        }
        users: list[dict[str, Any]] = []
        item_names: dict[str, str] = {}
        interaction_buffer: list[dict[str, Any]] = []
        interactions_rows = 0

        writer = pq.ParquetWriter(interactions_path, _INTERACTIONS_SCHEMA)
        try:
            for line_no, row in cls._iter_user_rows(users_items_path):
                steam_id = selected_source_lines.get(line_no)
                if steam_id is None:
                    continue

                snapshot = selected_by_steam_id[steam_id]
                user_id = cls._canonical_user_id(steam_id)
                users.append(
                    {
                        "user_id": user_id,
                        "steam_id": steam_id,
                        "raw_user_id": snapshot["raw_user_id"],
                        "user_url": snapshot["user_url"],
                        "reported_items_count": snapshot["reported_items_count"],
                        "source_line": snapshot["source_line"],
                        "raw_snapshot_count": snapshot_counts[steam_id],
                    }
                )

                for item in row.get("items") or []:
                    item_id = cls._none_to_empty_str(item.get("item_id"))
                    playtime_forever = cls._as_int(item.get("playtime_forever"))
                    playtime_2weeks = cls._as_int(item.get("playtime_2weeks"))
                    item_name = cls._none_to_empty_str(item.get("item_name"))
                    if item_id and item_id not in item_names:
                        item_names[item_id] = item_name

                    interaction_buffer.append(
                        {
                            "interaction_id": f"steam:line:{line_no:08d}:item:{item_id}",
                            "user_id": user_id,
                            "item_id": item_id,
                            "event_type": "owned",
                            "playtime_forever": playtime_forever,
                            "playtime_2weeks": playtime_2weeks,
                            "source_user_line": line_no,
                            "target_interact": 1,
                            "target_played_120": int(playtime_forever >= 120),
                            "target_playtime": playtime_forever,
                        }
                    )
                    interactions_rows += 1

                    if len(interaction_buffer) >= 100_000:
                        cls._write_interaction_batch(writer, interaction_buffer)
                        interaction_buffer.clear()

            if interaction_buffer:
                cls._write_interaction_batch(writer, interaction_buffer)
        finally:
            writer.close()

        users_df = pd.DataFrame(users).sort_values("source_line").reset_index(drop=True)
        users_df.to_parquet(users_path, index=False)

        items_df = cls._build_items_table(item_names, metadata)
        items_df.to_parquet(items_path, index=False)

        library_items_with_metadata = int(items_df["metadata_available"].sum())
        return {
            "users_rows": int(len(users_df)),
            "items_rows": int(len(items_df)),
            "interactions_rows": interactions_rows,
            "library_items_with_metadata": library_items_with_metadata,
            "library_items_without_metadata": int(
                len(items_df) - library_items_with_metadata
            ),
        }

    @staticmethod
    def _write_interaction_batch(
        writer: pq.ParquetWriter, rows: list[dict[str, Any]]
    ) -> None:
        writer.write_table(pa.Table.from_pylist(rows, schema=_INTERACTIONS_SCHEMA))

    @classmethod
    def _build_items_table(
        cls, item_names: dict[str, str], metadata: dict[str, dict[str, Any]]
    ) -> pd.DataFrame:
        rows = []
        for item_id, library_item_name in sorted(item_names.items(), key=lambda pair: pair[0]):
            meta = metadata.get(item_id)
            metadata_title = cls._none_to_empty_str(
                None if meta is None else meta.get("title") or meta.get("app_name")
            )
            title = library_item_name or metadata_title
            rows.append(
                {
                    "item_id": item_id,
                    "raw_item_id": item_id,
                    "title": title,
                    "library_item_name": library_item_name,
                    "metadata_available": meta is not None,
                    "metadata_title": metadata_title,
                    "app_name": cls._none_to_empty_str(None if meta is None else meta.get("app_name")),
                    "release_date": cls._none_to_empty_str(
                        None if meta is None else meta.get("release_date")
                    ),
                    "genres_json": cls._json_value(None if meta is None else meta.get("genres")),
                    "tags_json": cls._json_value(None if meta is None else meta.get("tags")),
                    "specs_json": cls._json_value(None if meta is None else meta.get("specs")),
                    "price_raw": cls._none_to_empty_str(None if meta is None else meta.get("price")),
                    "developer": cls._none_to_empty_str(None if meta is None else meta.get("developer")),
                    "publisher": cls._none_to_empty_str(None if meta is None else meta.get("publisher")),
                }
            )
        return pd.DataFrame(rows)

    @classmethod
    def _read_game_metadata(cls, path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
        metadata: dict[str, dict[str, Any]] = {}
        id_counts: Counter[str] = Counter()
        rows = 0
        rows_with_id = 0

        with gzip.open(path, "rt", encoding="utf-8") as file:
            for line in file:
                row = ast.literal_eval(line)
                rows += 1
                item_id = cls._none_to_empty_str(row.get("id"))
                if not item_id:
                    continue
                rows_with_id += 1
                id_counts[item_id] += 1
                metadata.setdefault(item_id, row)

        duplicate_ids = {item_id: count for item_id, count in id_counts.items() if count > 1}
        stats = {
            "metadata_rows": rows,
            "metadata_rows_with_id": rows_with_id,
            "metadata_unique_items": len(metadata),
            "metadata_duplicate_item_id_groups": len(duplicate_ids),
            "metadata_duplicate_item_id_surplus_rows": rows_with_id - len(metadata),
        }
        return metadata, stats

    @staticmethod
    def _iter_user_rows(path: Path):
        with path.open(encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                yield line_no, ast.literal_eval(line)

    @classmethod
    def _snapshot_summary(cls, line_no: int, row: dict[str, Any]) -> dict[str, Any]:
        items = row.get("items") or []
        playtimes_forever = [cls._as_int(item.get("playtime_forever")) for item in items]
        playtimes_2weeks = [cls._as_int(item.get("playtime_2weeks")) for item in items]
        # Current raw Steam data has steam_id on every row; fallback keeps malformed rows traceable.
        steam_id = cls._none_to_empty_str(row.get("steam_id")) or f"missing-steam-id-line-{line_no}"
        return {
            "source_line": line_no,
            "steam_id": steam_id,
            "raw_user_id": cls._none_to_empty_str(row.get("user_id")),
            "user_url": cls._none_to_empty_str(row.get("user_url")),
            "reported_items_count": cls._as_int(row.get("items_count")),
            "observed_items_count": len(items),
            "total_playtime_forever": sum(playtimes_forever),
            "total_playtime_2weeks": sum(playtimes_2weeks),
        }

    @staticmethod
    def _is_better_snapshot(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
        return (
            candidate["total_playtime_forever"] > current["total_playtime_forever"]
            or (
                candidate["total_playtime_forever"] == current["total_playtime_forever"]
                and candidate["source_line"] < current["source_line"]
            )
        )

    @staticmethod
    def _canonical_user_id(steam_id: str) -> str:
        return f"steam:{steam_id}"

    @staticmethod
    def _none_to_empty_str(value: object) -> str:
        return "" if value is None else str(value)

    @staticmethod
    def _as_int(value: object) -> int:
        return 0 if value in (None, "") else int(value)

    @staticmethod
    def _json_value(value: object) -> str:
        if value is None:
            return ""
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
