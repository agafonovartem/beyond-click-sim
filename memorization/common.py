"""Shared paths and data loaders for the memorization experiment scripts.

Deliberately small: these scripts are meant to be a simple, readable set of
standalone runners, not another task-builder framework. They reuse the canonical
MovieLens adapter and the classical policies from ``beyond_click_sim`` but do not
touch the runners/task machinery.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATASET = "ml-1m"
CANONICAL_VERSION = "v1"
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def repo_root() -> Path:
    for path in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (path / "pyproject.toml").exists() and (path / "src").exists():
            return path
    raise RuntimeError("Could not find repo root (pyproject.toml + src/)")


def canonical_dir() -> Path:
    return repo_root() / "data" / "canonical" / DATASET / CANONICAL_VERSION


def raw_dir() -> Path:
    return repo_root() / "data" / "raw" / DATASET


def mem_dir() -> Path:
    return repo_root() / "memorization"


def mem_data_dir() -> Path:
    d = mem_dir() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def mem_out_dir() -> Path:
    d = mem_dir() / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_canonical() -> Path:
    """Materialize canonical parquet (downloading raw ML-1M if needed)."""
    out = canonical_dir()
    if (out / "interactions.parquet").exists():
        return out

    raw = raw_dir()
    if not (raw / "ratings.dat").exists():
        raw.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {ML1M_URL} ...")
        with urllib.request.urlopen(ML1M_URL) as resp:  # noqa: S310 (trusted host)
            blob = resp.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            for member in zf.namelist():
                name = Path(member).name
                if name in {"ratings.dat", "movies.dat", "users.dat"}:
                    (raw / name).write_bytes(zf.read(member))
        print(f"Raw ML-1M written to {raw}")

    from beyond_click_sim.data.adapters.movielens import MovieLens1MAdapter

    MovieLens1MAdapter().materialize(raw, out)
    print(f"Canonical ML-1M materialized to {out}")
    return out


def load_interactions() -> pd.DataFrame:
    """Interactions in ratings-file order (row i == ml-1m:row:{i+1})."""
    path = ensure_canonical() / "interactions.parquet"
    cols = ["interaction_id", "user_id", "item_id", "rating", "timestamp"]
    return pd.read_parquet(path, columns=cols)


def load_items() -> pd.DataFrame:
    path = ensure_canonical() / "items.parquet"
    return pd.read_parquet(path, columns=["item_id", "title", "genres"])


def title_map() -> dict[str, str]:
    items = load_items()
    return dict(zip(items["item_id"], items["title"], strict=True))


def catalog_titles(normalizer) -> list[str]:
    """All ML-1M titles, normalized — used by matching modes that filter to the catalog."""
    return [normalizer(t) for t in load_items()["title"].tolist()]


def load_split(split_name: str) -> pd.DataFrame:
    """Load a split CSV written by prepare_data.py ('file_order' or 'random')."""
    path = mem_data_dir() / f"split_{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run prepare_data.py first.")
    return pd.read_csv(path, dtype={"user_id": str, "item_id": str})


def load_eval_users() -> list[str]:
    path = mem_data_dir() / "eval_users.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run prepare_data.py first.")
    return pd.read_csv(path, dtype={"user_id": str})["user_id"].tolist()
