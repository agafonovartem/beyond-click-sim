from __future__ import annotations

from pathlib import Path
from typing import Protocol

from beyond_click_sim.data.canonical import CanonicalDataset


class DatasetAdapter(Protocol):
    """Convert one raw/source dataset into canonical tables."""

    name: str

    def materialize(self, raw_dir: Path, out_dir: Path) -> CanonicalDataset:
        """Write canonical artifacts and return a descriptor."""
        ...
