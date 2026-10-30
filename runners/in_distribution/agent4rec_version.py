"""Version checks scoped to Agent4Rec; historical outputs remain untouched."""

import json
from pathlib import Path

from beyond_click_sim.scorers.agent4rec.profiles import AGENT4REC_IMPLEMENTATION_VERSION


def compatible_result(path: Path, method_name: str) -> bool:
    if not method_name.startswith("agent4rec_"):
        return True
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        profile = manifest.get("scorer", {}).get("profile_generator", {})
        return profile.get("implementation_version") == AGENT4REC_IMPLEMENTATION_VERSION
    except (OSError, ValueError, TypeError):
        return False


def protect_output_directory(path: Path) -> None:
    if any(
        (path / name).exists()
        for name in (
            "manifest.json",
            "predictions.parquet",
            "metrics.json",
            "metrics_ranking.json",
        )
    ):
        raise ValueError(
            "Agent4Rec requires a fresh output directory; preserve historical results and choose a new run directory"
        )
