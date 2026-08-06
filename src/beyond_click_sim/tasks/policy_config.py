"""Config-driven policy rosters for the policy-ranking experiments.

Policy rosters are defined by committed JSON configs under
`configs/policies/{setup}.json`, reviewed and edited by hand after offline
tuning (`src/beyond_click_sim/tasks/policy_tuning.py` produces the evidence
artifacts; `tuned_from` links connect config entries back to them).

Config schema (schema_version 1):

```json
{
  "schema_version": 1,
  "setup": "preference",
  "datasets": {
    "ml-1m": {
      "default": [
        {"name": "random", "class": "RandomPolicy"},
        {"name": "als", "class": "ALSPolicy",
         "params": {"n_factors": 64, "regularization": 0.1},
         "tuned_from": "artifacts/ml-1m_target_like_ge4_seed0.json",
         "comment": "why this entry exists"}
      ],
      "seed3": [ {"name": "...", "class": "...", "params": {...}} ]
    }
  }
}
```

Resolution: `datasets[dataset]["seed{N}"]` when present, else `["default"]`.
`k`, `seed`, `positive_feedback_column`, and `name` are injected by the loader
from task-level constants — they must not appear in `params`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from beyond_click_sim.tasks.policies import Policy
from beyond_click_sim.tasks.policy_tuning import POLICY_CLASSES

CONFIG_SCHEMA_VERSION = 1
_ENTRY_KEYS = frozenset({"name", "class", "params", "tuned_from", "comment"})
_RESERVED_PARAMS = frozenset({"k", "seed", "name", "positive_feedback_column"})
_SECTION_KEY_PATTERN = re.compile(r"seed\d+")


class PolicyConfigError(ValueError):
    """Invalid policy config (bad schema, unknown class, bad params, ...)."""


def default_policies_config_dir() -> Path:
    """Default policy-config location: `<repo root>/configs/policies/`."""

    here = Path(__file__).resolve()
    for path in [here, *here.parents]:
        pyproject = path / "pyproject.toml"
        if pyproject.exists() and 'name = "beyond-click-sim"' in pyproject.read_text(
            encoding="utf-8"
        ):
            return path / "configs" / "policies"
    raise RuntimeError("Could not find beyond-click-sim repo root")


def _fail(config_path: Path, message: str) -> None:
    raise PolicyConfigError(f"Invalid policy config {config_path}: {message}")


def _validate_entry(
    entry: Any,
    *,
    config_path: Path,
    section: str,
) -> tuple[str, type[Policy], dict[str, Any]]:
    if not isinstance(entry, dict):
        _fail(config_path, f"entry in {section} is not an object: {entry!r}")
    unknown_keys = sorted(set(entry) - _ENTRY_KEYS)
    if unknown_keys:
        _fail(
            config_path,
            f"entry {entry.get('name')!r} in {section} has unknown keys "
            f"{unknown_keys}; allowed: {sorted(_ENTRY_KEYS)}",
        )
    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        _fail(config_path, f"entry in {section} has a missing/empty 'name'")
    class_name = entry.get("class")
    if class_name not in POLICY_CLASSES:
        _fail(
            config_path,
            f"entry {name!r} in {section} has unknown class {class_name!r}; "
            f"available: {sorted(POLICY_CLASSES)}",
        )
    params = entry.get("params", {})
    if not isinstance(params, dict):
        _fail(config_path, f"entry {name!r} in {section}: 'params' must be an object")
    reserved = sorted(set(params) & _RESERVED_PARAMS)
    if reserved:
        _fail(
            config_path,
            f"entry {name!r} in {section}: params {reserved} are injected from "
            "task constants and must not appear in the config",
        )
    return name, POLICY_CLASSES[class_name], params


def load_policy_roster(
    *,
    setup: str,
    dataset_name: str,
    split_seed: int,
    policy_k: int,
    policy_seed: int = 0,
    positive_feedback_column: str | None = None,
    config_dir: Path | None = None,
    repo_root: Path | None = None,
) -> list[Policy]:
    """Build the policy roster for one (setup, dataset, seed) from a config."""

    config_dir = default_policies_config_dir() if config_dir is None else config_dir
    config_path = config_dir / f"{setup}.json"
    if not config_path.exists():
        raise RuntimeError(
            f"Policy config not found: {config_path}. Create it (optionally "
            "from tuning artifacts via `runners/in_distribution/policy_tuning/run.py "
            "--emit-config`) and review it before running experiments."
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        _fail(
            config_path,
            f"unsupported schema_version {payload.get('schema_version')!r} "
            f"(expected {CONFIG_SCHEMA_VERSION})",
        )
    if payload.get("setup") != setup:
        _fail(
            config_path,
            f"setup mismatch: file declares {payload.get('setup')!r}, "
            f"requested {setup!r}",
        )
    datasets = payload.get("datasets", {})
    if dataset_name not in datasets:
        _fail(
            config_path,
            f"no dataset section for {dataset_name!r}; "
            f"present: {sorted(datasets)}",
        )
    dataset_section = datasets[dataset_name]
    unknown_sections = sorted(
        key
        for key in dataset_section
        if key != "default"
        and not _SECTION_KEY_PATTERN.fullmatch(key)
        and not key.startswith("_")  # informational keys, e.g. _review_notes
    )
    if unknown_sections:
        _fail(
            config_path,
            f"dataset {dataset_name!r} has unknown section keys "
            f"{unknown_sections}; allowed: 'default', 'seed<N>', '_'-prefixed "
            "informational keys",
        )
    seed_key = f"seed{split_seed}"
    if seed_key in dataset_section:
        section_key = seed_key
    elif "default" in dataset_section:
        section_key = "default"
    else:
        _fail(
            config_path,
            f"dataset {dataset_name!r} has no 'seed{split_seed}' section and "
            "no 'default' fallback",
        )
    section = f"datasets.{dataset_name}.{section_key}"
    entries = dataset_section[section_key]
    if not isinstance(entries, list) or not entries:
        _fail(config_path, f"section {section} must be a non-empty list")

    roster: list[Policy] = []
    seen_names: set[str] = set()
    for entry in entries:
        name, cls, params = _validate_entry(
            entry, config_path=config_path, section=section
        )
        if name in seen_names:
            _fail(
                config_path,
                f"duplicate policy name {name!r} in section {section}",
            )
        seen_names.add(name)
        try:
            policy = cls(
                k=policy_k,
                seed=policy_seed,
                positive_feedback_column=positive_feedback_column,
                name=name,
                **params,
            )
        except (TypeError, ValueError) as error:
            _fail(
                config_path,
                f"entry {name!r} in {section}: bad params for "
                f"{cls.__name__}: {error}",
            )
        if repo_root is not None:
            config_ref = str(config_path.relative_to(repo_root))
        else:
            config_ref = str(config_path)
        policy.hyperparameter_selection = {
            "kind": "manual_config",
            "config": config_ref,
            "section": section,
            "tuned_from": entry.get("tuned_from"),
            "comment": entry.get("comment"),
        }
        roster.append(policy)
    return roster


def build_config_skeleton(
    *,
    setup: str,
    dataset_sections: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    """Draft a config payload from tuning artifacts for manual review.

    `dataset_sections` maps dataset_name -> {split_seed: artifact_payload}.
    The `default` roster comes from the lowest seed's artifact (`best_params`
    plus a `tuned_from` link); a `_review_notes` block summarizes cross-seed
    hyperparameter divergence so the reviewer can decide whether per-seed
    `seed<N>` overrides are warranted.
    """

    datasets: dict[str, Any] = {}
    for dataset_name, seed_to_artifact in sorted(dataset_sections.items()):
        if not seed_to_artifact:
            continue
        base_seed = min(seed_to_artifact)
        base_artifact = seed_to_artifact[base_seed]
        base_target = base_artifact["identity"]["target_source_column"]

        roster: list[dict[str, Any]] = []
        for class_name, entry in base_artifact["policies"].items():
            item: dict[str, Any] = {
                "name": class_name,
                "class": class_name,
                "params": dict(entry["selected_params"]),
                "tuned_from": (
                    f"artifacts/{dataset_name}_{base_target}_seed{base_seed}.json"
                ),
            }
            if not entry["tuned"]:
                item["comment"] = "baseline (no hyperparameters to tune)"
            roster.append(item)

        review_notes: list[str] = []
        for class_name in base_artifact["policies"]:
            per_seed = {
                seed: artifact["policies"][class_name]["selected_params"]
                for seed, artifact in sorted(seed_to_artifact.items())
            }
            distinct = {
                json.dumps(params, sort_keys=True) for params in per_seed.values()
            }
            if len(distinct) > 1:
                detail = ", ".join(
                    f"seed{seed}={params}" for seed, params in per_seed.items()
                )
                review_notes.append(
                    f"{class_name}: selected params differ across seeds "
                    f"({detail}). Decide: keep one default, or add seed<N> "
                    "override sections."
                )
            else:
                review_notes.append(
                    f"{class_name}: same params selected on all "
                    f"{len(per_seed)} seeds."
                )

        datasets[dataset_name] = {
            "default": roster,
            "_review_notes": review_notes,
        }
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "setup": setup,
        "datasets": datasets,
    }
