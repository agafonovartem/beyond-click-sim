"""Render yes/no prompts offline, with no LLM call and no network.

Two jobs:

1. **Regression baseline.** Render ml-1m and steam prompts and hash them, before
   and after wiring a new dataset. The hashes must match byte-for-byte; if they
   move, a supposedly additive change was not additive.
2. **Coat inspection.** Render real Coat prompts from the canonical tables so the
   synthesised item text can be eyeballed before any spend.

The scorer is constructed directly with ``client=None`` and only its private
``_build_messages`` is called, following the pattern already used by
``notebooks/inspect_agent4rec_final_prompts.ipynb`` and
``runners/in_distribution/regression_prediction/snapshot_ollama_smoke10_prompts.py``.

Usage::

    python scripts/render_coat_prompts.py --dataset ml-1m --hash-only
    python scripts/render_coat_prompts.py --dataset coat --users 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from beyond_click_sim.scorers.history.yes_no import LLMPreferenceYesNoScorer  # noqa: E402
from runners.in_distribution.policy_ranking_agreement_preference.methods import (  # noqa: E402
    llm_yes_no as _preference,
)
from runners.in_distribution.policy_ranking_agreement_rerank.methods import (  # noqa: E402
    llm_yes_no as _rerank,
)

# Each package owns its own datasets' prompt config; there is no shared copy to
# drift out of sync. Coat lives in the rerank fork, ml-1m and steam in the
# preference package.
_CONFIG_OWNER = {"coat": _rerank, "ml-1m": _preference, "steam": _preference}


def _config_for(dataset: str):
    if dataset not in _CONFIG_OWNER:
        raise SystemExit(
            f"{dataset!r} has no known prompt-config owner; known: "
            f"{sorted(_CONFIG_OWNER)}"
        )
    owner = _CONFIG_OWNER[dataset]
    return (
        owner.DATASET_PROMPT_COLUMNS[dataset],
        owner.DATASET_JSON_LIST_COLUMNS[dataset],
        owner.TARGET_DESCRIPTIONS[dataset],
    )

MAX_HISTORY_ITEMS = 24  # DATASET_MAX_HISTORY_ITEMS['coat'] in the rerank fork


# ---------------------------------------------------------------------------
# Fixtures for the regression baseline
#
# Synthetic on purpose: the only ways this repo's prompts can change are the
# shared templates and the per-dataset column dicts, and a fixed frame exercises
# both without depending on canonical data being materialized.
# ---------------------------------------------------------------------------

_BASELINE_FIXTURES: dict[str, dict[str, list[dict[str, object]]]] = {
    "ml-1m": {
        "history": [
            {"user_id": "1", "item_title": "Toy Story (1995)", "item_genres": "Animation|Children's|Comedy", "rating": 5},
            {"user_id": "1", "item_title": "Fargo (1996)", "item_genres": "Crime|Drama|Thriller", "rating": 2},
        ],
        "candidates": [
            {"user_id": "1", "item_title": "Shrek (2001)", "item_genres": "Animation|Comedy"},
            {"user_id": "1", "item_title": "Se7en (1995)", "item_genres": "Crime|Thriller"},
        ],
    },
    "steam": {
        "history": [
            {"user_id": "steam:1", "item_title": "Portal 2", "item_genres_json": '["Action", "Puzzle"]', "item_tags_json": '["Co-op"]', "playtime_forever": 640},
            {"user_id": "steam:1", "item_title": "Euro Truck Simulator 2", "item_genres_json": '["Simulation"]', "item_tags_json": '["Driving"]', "playtime_forever": 0},
        ],
        "candidates": [
            {"user_id": "steam:1", "item_title": "Half-Life 2", "item_genres_json": '["Action"]', "item_tags_json": '["FPS"]'},
            {"user_id": "steam:1", "item_title": "Stardew Valley", "item_genres_json": '["Simulation", "RPG"]', "item_tags_json": '["Farming"]'},
        ],
    },
}


def _baseline_frames(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixture = _BASELINE_FIXTURES[dataset]
    history = pd.DataFrame(fixture["history"])
    candidates = pd.DataFrame(fixture["candidates"])
    candidates["candidate_group"] = "g0"
    return history, candidates


# ---------------------------------------------------------------------------
# Coat frames, built from the canonical tables
# ---------------------------------------------------------------------------


def _coat_frames(
    n_users: int,
    canonical_root: Path,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build history and candidate frames the way the real task builder does.

    Uses the production `SourcePartitionSplitter`, so history is every
    self-selected row minus that user's overlap cells (20-24 rows) exactly as in
    a real run. The one stand-in left is the candidate set: policies are not
    fitted here, so candidates are the user's whole 16-item pool rather than one
    policy's top-k slate. That exercises every part of the prompt except which
    k items a policy would pick.
    """

    from beyond_click_sim.tasks.splitters import SourcePartitionSplitter

    items = pd.read_parquet(canonical_root / "items.parquet")
    interactions = pd.read_parquet(canonical_root / "interactions.parquet")

    # Mirror tasks/base.py:_prefixed_features, which prefixes item columns at
    # task-build time rather than in the adapter.
    items = items.rename(
        columns={c: f"item_{c}" for c in items.columns if c != "item_id"}
    )
    split = SourcePartitionSplitter().split(interactions)
    users = sorted(interactions["user_id"].unique(), key=int)[:n_users]

    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame[frame["user_id"].isin(users)].merge(items, on="item_id", how="left")
        return frame.sort_values(["user_id", "item_id"], key=lambda c: c.map(int)).reset_index(drop=True)

    history = prepare(split.train)
    candidates = prepare(split.test)
    candidates = candidates.assign(candidate_group=candidates["user_id"])
    return history, candidates


# ---------------------------------------------------------------------------


def render(
    dataset: str,
    n_users: int,
    canonical_root: Path | None,
    seed: int = 0,
) -> list[str]:
    columns, json_list_columns, target_description = _config_for(dataset)

    if dataset == "coat":
        root = canonical_root or REPO_ROOT / "data" / "canonical" / "coat" / "v1"
        if not root.exists():
            raise SystemExit(f"Canonical Coat tables not found at {root}")
        history, candidates = _coat_frames(n_users, root, seed)
    else:
        history, candidates = _baseline_frames(dataset)

    scorer = LLMPreferenceYesNoScorer(
        client=None,
        model="offline-render",
        target_description=target_description,
        history_description_columns=columns["history_description_columns"],
        candidate_description_columns=columns["candidate_description_columns"],
        json_list_columns=json_list_columns,
        max_history_items=MAX_HISTORY_ITEMS,
    )
    scorer.fit(history, pd.Series([0] * len(history)))

    rendered: list[str] = []
    for _, group in candidates.groupby("candidate_group", sort=True):
        labels = [f"C{i}" for i in range(1, len(group) + 1)]
        if len(set(labels)) != len(labels):
            raise AssertionError("candidate labels are not unique")
        messages = scorer._build_messages(
            user_id=group["user_id"].iloc[0],
            candidates=group,
            labels=labels,
        )
        rendered.append(
            "\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
        )

    if dataset == "coat":
        _check_coat_prompts(rendered, history, scorer)
    return rendered


def _check_coat_prompts(
    rendered: list[str],
    history: pd.DataFrame,
    scorer: LLMPreferenceYesNoScorer,
) -> None:
    """Assertions that would otherwise only fail after money has been spent."""

    per_user = history.groupby("user_id").size()
    if per_user.max() > MAX_HISTORY_ITEMS:
        raise AssertionError(
            f"history of {per_user.max()} rows exceeds MAX_HISTORY_ITEMS="
            f"{MAX_HISTORY_ITEMS}; the prompt would be silently truncated"
        )
    for user_id, rows in scorer.history_by_user_.items():
        if len(rows) != per_user[user_id]:
            raise AssertionError(
                f"user {user_id}: {per_user[user_id]} history rows but "
                f"{len(rows)} rendered lines — truncation happened"
            )

    # `on_front_page` is an exposure variable and must never reach a prompt.
    banned = ("on_front_page", "front page", "promoted")
    for prompt in rendered:
        lowered = prompt.lower()
        for token in banned:
            if token in lowered:
                raise AssertionError(f"promotion signal {token!r} leaked into a prompt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--users", type=int, default=2, help="Coat users to render.")
    parser.add_argument("--canonical-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Coat train-split seed.")
    parser.add_argument("--hash-only", action="store_true")
    parser.add_argument("--out", type=Path, default=None, help="Write JSON snapshot.")
    args = parser.parse_args()

    prompts = render(args.dataset, args.users, args.canonical_root, args.seed)
    joined = "\n\n===PROMPT-BOUNDARY===\n\n".join(prompts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()

    if args.hash_only:
        print(f"{args.dataset}\t{len(prompts)} prompt(s)\tsha256={digest}")
    else:
        for i, prompt in enumerate(prompts, start=1):
            print(f"{'=' * 70}\nPROMPT {i}/{len(prompts)}\n{'=' * 70}")
            print(prompt)
        print(f"\nsha256={digest}")

    if args.out:
        args.out.write_text(
            json.dumps(
                {"dataset": args.dataset, "sha256": digest, "prompts": prompts},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
