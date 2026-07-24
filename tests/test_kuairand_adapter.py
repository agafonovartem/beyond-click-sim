from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from beyond_click_sim.data.adapters.kuairand import (
    CATEGORY_LEVELS,
    RAW_INTERACTION_COLUMNS,
    KuaiRandPureAdapter,
)


def _interaction_row(
    *,
    user_id: int,
    video_id: int,
    date: int,
    is_click: int,
    long_view: int,
    is_rand: int,
    tab: int = 1,
) -> dict[str, int]:
    return {
        "user_id": user_id,
        "video_id": video_id,
        "date": date,
        "hourmin": 1234,
        "time_ms": 1_650_000_000_000 + video_id,
        "is_click": is_click,
        "is_like": 0,
        "is_follow": 0,
        "is_comment": 0,
        "is_forward": 0,
        "is_hate": 0,
        "long_view": long_view,
        "play_time_ms": 12_000,
        "duration_ms": 30_000,
        "profile_stay_time": 0,
        "comment_stay_time": 0,
        "is_profile_enter": 0,
        "is_rand": is_rand,
        "tab": tab,
    }


def _category_row(
    video_id: int,
    names: tuple[str, str, str, str],
) -> dict[str, int | float | str]:
    row: dict[str, int | float | str] = {"final_video_id": video_id}
    for index, (level, name) in enumerate(zip(CATEGORY_LEVELS, names, strict=True)):
        row[f"{level}_level_category_id"] = index
        row[f"{level}_level_category_name"] = name
        row[f"{level}_level_category_prob"] = 1.0 if name else 0.0
    return row


def write_fixture(raw_dir: Path) -> None:
    data_dir = raw_dir / "data"
    supplement_dir = raw_dir / "supplement"
    data_dir.mkdir(parents=True)
    supplement_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {"user_id": 0, "user_active_degree": "high_active"},
            {"user_id": 1, "user_active_degree": "middle_active"},
        ]
    ).to_csv(data_dir / "user_features_pure.csv", index=False)

    pd.DataFrame(
        [
            {
                "video_id": 0,
                "author_id": 10,
                "video_type": "NORMAL",
                "video_duration": 30_000,
            },
            {
                "video_id": 1,
                "author_id": 11,
                "video_type": "NORMAL",
                "video_duration": 40_000,
            },
            {
                "video_id": 2,
                "author_id": 12,
                "video_type": "NORMAL",
                "video_duration": 50_000,
            },
        ]
    ).to_csv(data_dir / "video_features_basic_pure.csv", index=False)
    pd.DataFrame(
        [
            {"video_id": 0, "show_cnt": 99_999},
            {"video_id": 1, "show_cnt": 88_888},
            {"video_id": 2, "show_cnt": 77_777},
        ]
    ).to_csv(data_dir / "video_features_statistic_pure.csv", index=False)

    full_caption = (
        "A deliberately long caption retained in full for semantic prompting "
        "and never cut to a short prefix."
    )
    pd.DataFrame(
        [
            {
                "final_video_id": 0,
                "caption": full_caption,
                "show_cover_text": "Unused cover text",
                "duration": 30_000,
            },
            {
                "final_video_id": 1,
                "caption": "",
                "show_cover_text": "  Cover text used as the fallback.  ",
                "duration": 40_000,
            },
            {
                "final_video_id": 2,
                "caption": "",
                "show_cover_text": "",
                "duration": 50_000,
            },
        ]
    ).to_csv(
        supplement_dir / "kuairand_video_captions_pure.csv",
        index=False,
    )
    pd.DataFrame(
        [
            _category_row(0, ("Games", "Console", "Action", "UNKNOWN")),
            _category_row(1, ("Life", "Weddings", "", "")),
            _category_row(2, ("UNKNOWN", "", "", "")),
        ]
    ).to_csv(
        supplement_dir / "kuairand_video_categories_pure.csv",
        index=False,
    )

    interaction_rows = {
        "log_standard_4_08_to_4_21_pure.csv": [
            _interaction_row(
                user_id=0,
                video_id=0,
                date=20220408,
                is_click=1,
                long_view=1,
                is_rand=0,
            ),
            _interaction_row(
                user_id=1,
                video_id=1,
                date=20220409,
                is_click=0,
                long_view=0,
                is_rand=0,
            ),
        ],
        "log_standard_4_22_to_5_08_pure.csv": [
            _interaction_row(
                user_id=0,
                video_id=2,
                date=20220422,
                is_click=0,
                long_view=1,
                is_rand=0,
            )
        ],
        "log_random_4_22_to_5_08_pure.csv": [
            _interaction_row(
                user_id=0,
                video_id=0,
                date=20220422,
                is_click=0,
                long_view=0,
                is_rand=1,
            )
        ],
    }
    for filename, rows in interaction_rows.items():
        pd.DataFrame(rows, columns=RAW_INTERACTION_COLUMNS).to_csv(
            data_dir / filename,
            index=False,
        )


def test_kuairand_adapter_preserves_exposures_and_full_semantics(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw" / "KuaiRand-Pure"
    out_dir = tmp_path / "canonical" / "kuairand-pure" / "v1"
    write_fixture(raw_dir)

    dataset = KuaiRandPureAdapter().materialize(raw_dir, out_dir)

    assert dataset.users_path.exists()
    assert dataset.items_path.exists()
    assert dataset.interactions_path.exists()
    assert dataset.manifest_path.exists()

    users = pd.read_parquet(dataset.users_path)
    items = pd.read_parquet(dataset.items_path)
    interactions = pd.read_parquet(dataset.interactions_path)

    assert users["user_id"].tolist() == ["0", "1"]
    assert users["raw_user_id"].tolist() == ["0", "1"]
    assert "show_cnt" not in items.columns

    item_0 = items.loc[items["item_id"].eq("0")].iloc[0]
    item_1 = items.loc[items["item_id"].eq("1")].iloc[0]
    item_2 = items.loc[items["item_id"].eq("2")].iloc[0]
    assert item_0["description"] == item_0["caption"]
    assert item_0["description"].endswith("never cut to a short prefix.")
    assert item_0["category_path"] == "Games > Console > Action"
    assert item_1["caption"] == ""
    assert item_1["description"] == "Cover text used as the fallback."
    assert item_1["category_path"] == "Life > Weddings"
    assert item_2["description"] == ""
    assert item_2["category_path"] == ""

    assert list(interactions.columns) == [
        "interaction_id",
        "user_id",
        "item_id",
        "event_type",
        "exposure_policy",
        "source_log",
        "source_row",
        "date",
        "hourmin",
        "timestamp_ms",
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
        "target_click",
        "target_long_view",
    ]
    assert len(interactions) == 4
    assert interactions["interaction_id"].tolist() == [
        "kuairand:log_standard_4_08_to_4_21_pure:row:0000000001",
        "kuairand:log_standard_4_08_to_4_21_pure:row:0000000002",
        "kuairand:log_standard_4_22_to_5_08_pure:row:0000000001",
        "kuairand:log_random_4_22_to_5_08_pure:row:0000000001",
    ]
    assert interactions["event_type"].eq("exposure").all()
    assert interactions["exposure_policy"].tolist() == [
        "standard",
        "standard",
        "standard",
        "random",
    ]
    assert interactions["target_click"].tolist() == [1, 0, 0, 0]
    assert interactions["target_long_view"].tolist() == [1, 0, 1, 0]
    assert len(interactions.loc[interactions["user_id"].eq("0")]) == 3
    assert (
        len(
            interactions.loc[
                interactions["user_id"].eq("0")
                & interactions["item_id"].eq("0")
            ]
        )
        == 2
    )

    manifest = json.loads(dataset.manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"] == "kuairand-pure"
    assert manifest["schema_version"] == "canonical-v1"
    assert manifest["tables"]["users"]["rows"] == 2
    assert manifest["tables"]["items"]["rows"] == 3
    assert manifest["tables"]["interactions"]["rows"] == 4
    assert manifest["signals"] == {
        "exposure_rows": 4,
        "users_in_exposure_logs": 2,
        "items_in_exposure_logs": 3,
    }
    assert manifest["semantic_supplement"]["validation"] == "strict_exact_id_match"
    assert manifest["semantic_supplement"]["coverage"] == {
        "canonical_items": 3,
        "caption_nonblank": 1,
        "cover_text_nonblank": 2,
        "description_available": 2,
        "description_missing": 1,
        "category_path_available": 2,
        "category_path_missing": 1,
    }
    assert "no truncation" in manifest["semantic_supplement"]["text_policy"]
    assert len(manifest["exposure_logs"]) == 3
    assert manifest["exposure_logs"][0]["date_min"] == 20220408
    assert manifest["exposure_logs"][0]["date_max"] == 20220409
    assert manifest["exposure_logs"][2]["exposure_policy"] == "random"
    assert manifest["exposure_logs"][2]["expected_is_rand"] == 1
    assert len(manifest["excluded_sources"]) == 1
    assert "global behavioral aggregates" in manifest["excluded_sources"][0]["reason"]
    assert "video_features_statistic_pure.csv" not in {
        Path(source["path"]).name for source in manifest["raw_sources"]
    }


@pytest.mark.parametrize("semantic_source", ["captions", "categories"])
def test_kuairand_adapter_requires_exact_semantic_item_coverage(
    tmp_path: Path,
    semantic_source: str,
) -> None:
    raw_dir = tmp_path / "raw" / "KuaiRand-Pure"
    out_dir = tmp_path / "canonical" / "kuairand-pure" / "v1"
    write_fixture(raw_dir)
    filename = (
        "kuairand_video_captions_pure.csv"
        if semantic_source == "captions"
        else "kuairand_video_categories_pure.csv"
    )
    path = raw_dir / "supplement" / filename
    frame = pd.read_csv(path)
    frame.iloc[:-1].to_csv(path, index=False)

    with pytest.raises(ValueError, match="IDs must exactly match"):
        KuaiRandPureAdapter().materialize(raw_dir, out_dir)


def test_kuairand_adapter_rejects_duplicate_semantic_item_ids(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw" / "KuaiRand-Pure"
    out_dir = tmp_path / "canonical" / "kuairand-pure" / "v1"
    write_fixture(raw_dir)
    captions_path = (
        raw_dir / "supplement" / "kuairand_video_captions_pure.csv"
    )
    captions = pd.read_csv(captions_path)
    pd.concat([captions, captions.iloc[[0]]], ignore_index=True).to_csv(
        captions_path,
        index=False,
    )

    with pytest.raises(ValueError, match="duplicate final_video_id"):
        KuaiRandPureAdapter().materialize(raw_dir, out_dir)


def test_kuairand_adapter_rejects_policy_flag_mismatch(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "KuaiRand-Pure"
    out_dir = tmp_path / "canonical" / "kuairand-pure" / "v1"
    write_fixture(raw_dir)
    path = raw_dir / "data" / "log_standard_4_08_to_4_21_pure.csv"
    log = pd.read_csv(path)
    log["is_rand"] = 1
    log.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Unexpected is_rand values"):
        KuaiRandPureAdapter().materialize(raw_dir, out_dir)
