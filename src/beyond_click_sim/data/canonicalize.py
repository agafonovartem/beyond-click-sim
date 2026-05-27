from __future__ import annotations

import argparse
from pathlib import Path

from beyond_click_sim.data.adapters import MovieLens1MAdapter, SteamAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize canonical dataset tables.")
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    movielens = subparsers.add_parser("movielens", help="Canonicalize MovieLens-1M.")
    movielens.add_argument("--raw-dir", type=Path, required=True)
    movielens.add_argument("--out-dir", type=Path, required=True)

    steam = subparsers.add_parser("steam", help="Canonicalize Steam user-library snapshots.")
    steam.add_argument("--raw-dir", type=Path, required=True)
    steam.add_argument("--out-dir", type=Path, required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "movielens":
        dataset = MovieLens1MAdapter().materialize(args.raw_dir, args.out_dir)
    elif args.dataset == "steam":
        dataset = SteamAdapter().materialize(args.raw_dir, args.out_dir)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    manifest = dataset.load_manifest()
    print(f"Wrote canonical {manifest['dataset']} {manifest['version']} to {dataset.root}")
    for table, info in manifest["tables"].items():
        print(f"- {table}: {info['rows']} rows -> {info['path']}")


if __name__ == "__main__":
    main()
