from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from beyond_click_sim.data.canonical import (
    CanonicalDataset,
    CanonicalManifest,
    ManifestTable,
    file_record,
    write_manifest,
)


RAW_FILES = (
    "train.ascii",
    "test.ascii",
    "propensities.ascii",
    "user_item_features/item_features.ascii",
    "user_item_features/item_features_map.txt",
    "user_item_features/user_features.ascii",
    "user_item_features/user_features_map.txt",
)

N_USERS = 290
N_ITEMS = 300
N_SELF_SELECTED_PER_USER = 24
N_RANDOM_PER_USER = 16

ITEM_FEATURE_GROUPS = ("gender", "jackettype", "color", "onfrontpage")
USER_FEATURE_GROUPS = ("gender", "age", "location", "fashioninterest")

# Coat items have no name of any kind, so `title` is synthesised from the
# one-hot covariates. Both maps below are cosmetic: they change wording, never
# information, and are recorded in the manifest so the text stays auditable.
STYLE_SINGULAR = {"parkas": "parka", "shells": "shell", "vests": "vest"}
COLOUR_ALIASES = {"multi": "multicolour"}
GENDER_POSSESSIVE = {"men": "men's", "women": "women's"}

ITEM_TITLE_TEMPLATE = "{gender} {style} coat in {colour}"

SOURCE_SELF_SELECTED = "self_selected"
SOURCE_RANDOM = "random"


class CoatAdapter:
    """Canonicalize the Coat MNAR dataset (Schnabel et al., ICML 2016).

    Coat ships as whitespace-delimited ASCII matrices, not tables: a 290x300
    matrix of self-selected ratings, a second of uniformly-random-assigned
    ratings, estimated propensities, and one-hot user/item covariates. This
    adapter stacks both rating halves into one long frame with a `source` column
    recording which selection mechanism produced each row.

    Two Coat-specific facts drive the design:

    * **Items have no text.** `title`, `style`, and `colour` are synthesised
      from the one-hot covariates; 300 coats produce only 165 distinct titles.
    * **The two halves are not disjoint.** 366 cells appear in both matrices,
      with identical ratings, so those (user, item) pairs occur twice in
      `interactions.parquet` under different `source` values. Removing them from
      the training frame is a split decision and belongs to the task builder,
      not here.

    As with the other adapters, all observed ratings are preserved and no split,
    filter, or candidate set is applied.
    """

    name = "coat"
    version = "v1"

    def materialize(self, raw_dir: Path, out_dir: Path) -> CanonicalDataset:
        raw_dir = raw_dir.expanduser().resolve()
        out_dir = out_dir.expanduser().resolve()
        paths = {name: raw_dir / name for name in RAW_FILES}
        for path in paths.values():
            if not path.exists():
                raise FileNotFoundError(path)

        train = np.loadtxt(paths["train.ascii"])
        test = np.loadtxt(paths["test.ascii"])
        propensities = np.loadtxt(paths["propensities.ascii"])
        item_features = np.loadtxt(paths["user_item_features/item_features.ascii"])
        user_features = np.loadtxt(paths["user_item_features/user_features.ascii"])
        item_names = self._read_names(paths["user_item_features/item_features_map.txt"])
        user_names = self._read_names(paths["user_item_features/user_features_map.txt"])

        self._check_layout(
            train=train,
            test=test,
            propensities=propensities,
            item_features=item_features,
            user_features=user_features,
            item_names=item_names,
            user_names=user_names,
        )

        item_groups = self._group_index(item_names, ITEM_FEATURE_GROUPS)
        user_groups = self._group_index(user_names, USER_FEATURE_GROUPS)
        self._check_one_hot(item_features, item_groups, "item")
        self._check_one_hot(user_features, user_groups, "user")

        items = self._build_items(item_features, item_names, item_groups)
        users = self._build_users(user_features, user_names, user_groups)
        interactions = self._build_interactions(train, test, propensities)

        overlap = int(((train > 0) & (test > 0)).sum())
        overlap_disagreements = int(
            (train[(train > 0) & (test > 0)] != test[(train > 0) & (test > 0)]).sum()
        )
        distinct_titles = int(items["title"].nunique())

        out_dir.mkdir(parents=True, exist_ok=True)
        users_path_out = out_dir / "users.parquet"
        items_path_out = out_dir / "items.parquet"
        interactions_path_out = out_dir / "interactions.parquet"
        manifest_path = out_dir / "manifest.json"

        users.to_parquet(users_path_out, index=False)
        items.to_parquet(items_path_out, index=False)
        interactions.to_parquet(interactions_path_out, index=False)

        manifest = CanonicalManifest(
            dataset="coat",
            adapter=self.name,
            version=self.version,
            raw_sources=[file_record(paths[name]) for name in RAW_FILES],
            tables={
                "users": ManifestTable(path=users_path_out.name, rows=len(users)),
                "items": ManifestTable(path=items_path_out.name, rows=len(items)),
                "interactions": ManifestTable(
                    path=interactions_path_out.name,
                    rows=len(interactions),
                ),
            },
            id_policy={
                "user_id": "0-based row index of the Coat rating matrices, as string",
                "item_id": "0-based column index of the Coat rating matrices, as string",
                "interaction_id": "coat:{source}:u{user_index:03d}:i{item_index:03d}",
            },
            standard_targets={
                "target_interact": "1 for every observed rating row in interactions.parquet.",
                "target_like_ge4": "1 if rating >= 4 else 0.",
                "target_rating": "Raw Coat rating on the 1-5 scale.",
            },
            caveats=[
                "All observed ratings are preserved; no train/test split, candidate set, or row filter is applied.",
                (
                    f"The self-selected and random halves are NOT disjoint: {overlap} (user, item) cells appear "
                    "in both matrices and therefore twice in interactions.parquet, under different `source` "
                    "values. Ratings agree in every such cell. Excluding them from the training frame is a "
                    "split decision left to the task builder."
                ),
                (
                    "Coat has no timestamps, so no temporal split is possible; the split structure is given by "
                    "the collection design (`source`)."
                ),
                (
                    "`propensity` is populated only for source == 'self_selected'. It is Schnabel et al.'s "
                    "*estimated* observation probability from a covariate logistic regression, not a logged "
                    "quantity, so any inverse-propensity estimator inherits that model's error."
                ),
                (
                    "`title`, `style`, and `colour` are synthesised from one-hot covariates: Coat ships no item "
                    f"text. Only {distinct_titles} of {N_ITEMS} titles are distinct, so a text-only model cannot "
                    "separate every pair of items."
                ),
                (
                    "`on_front_page` records shop placement, not a garment property. It is primarily an exposure "
                    "variable (promoted coats drew ~3x the self-selected ratings) and only weakly a quality one, "
                    "so it is kept out of the default prompt columns."
                ),
                (
                    f"Every user has exactly {N_SELF_SELECTED_PER_USER} self-selected and "
                    f"{N_RANDOM_PER_USER} random ratings, so any MinUserInteractions filter below "
                    f"{N_SELF_SELECTED_PER_USER} is a no-op."
                ),
            ],
            extra_sections={
                "selection_design": {
                    "self_selected_rows": int((train > 0).sum()),
                    "random_rows": int((test > 0).sum()),
                    "overlapping_cells": overlap,
                    "overlapping_cells_with_disagreeing_ratings": overlap_disagreements,
                    "users_affected_by_overlap": int(
                        (((train > 0) & (test > 0)).sum(axis=1) > 0).sum()
                    ),
                    "source_values": [SOURCE_SELF_SELECTED, SOURCE_RANDOM],
                },
                "text_synthesis": {
                    "item_title_template": ITEM_TITLE_TEMPLATE,
                    "style_singularized": STYLE_SINGULAR,
                    "colour_aliases": COLOUR_ALIASES,
                    "distinct_titles": distinct_titles,
                    "items": N_ITEMS,
                    "note": (
                        "Wording is cosmetic; the title is a pure function of the "
                        "(gender, jackettype, color) tuple, so no phrasing can raise distinct_titles."
                    ),
                },
                "provenance": {
                    "citation": (
                        "Schnabel, Swaminathan, Singh, Chandak, Joachims. Recommendations as Treatments: "
                        "Debiasing Learning and Evaluation. ICML 2016."
                    ),
                    "canonical_source": "https://www.cs.cornell.edu/~schnabts/mnar/ (coat.zip)",
                    "license": "CC BY-NC (research use only)",
                    "note": (
                        "See data/raw/coat/PROVENANCE.json: this copy came from a mirror and was "
                        "cross-verified value-for-value against a second independent mirror. Byte-level "
                        "equality with the official coat.zip is not established."
                    ),
                },
            },
        )
        write_manifest(manifest_path, manifest)

        return CanonicalDataset(
            name="coat",
            version=self.version,
            root=out_dir,
            users_path=users_path_out,
            items_path=items_path_out,
            interactions_path=interactions_path_out,
            manifest_path=manifest_path,
        )

    @staticmethod
    def _read_names(path: Path) -> list[str]:
        """Read a feature map line-wise.

        Whitespace splitting would be wrong: two user feature names contain
        spaces (`age:over 60`, `fashioninterest:not at all`), which yields 18
        names for a 14-column matrix and silently misaligns every covariate.
        """

        return path.read_text(encoding="utf-8").splitlines()

    @staticmethod
    def _check_layout(
        *,
        train: np.ndarray,
        test: np.ndarray,
        propensities: np.ndarray,
        item_features: np.ndarray,
        user_features: np.ndarray,
        item_names: list[str],
        user_names: list[str],
    ) -> None:
        """Fail loudly if the archive is not the documented one.

        Deliberately absent: the disjointness assertion from the Coat setup doc.
        The shipped archive overlaps on 366 cells, so asserting it would reject
        the real data; the count is recorded in the manifest instead.
        """

        expected_shape = (N_USERS, N_ITEMS)
        for name, matrix in (
            ("train.ascii", train),
            ("test.ascii", test),
            ("propensities.ascii", propensities),
        ):
            if matrix.shape != expected_shape:
                raise ValueError(f"{name} has shape {matrix.shape}, expected {expected_shape}")

        train_counts = np.unique((train > 0).sum(axis=1)).tolist()
        if train_counts != [N_SELF_SELECTED_PER_USER]:
            raise ValueError(
                f"Expected exactly {N_SELF_SELECTED_PER_USER} self-selected ratings per user, "
                f"got counts {train_counts}"
            )
        test_counts = np.unique((test > 0).sum(axis=1)).tolist()
        if test_counts != [N_RANDOM_PER_USER]:
            raise ValueError(
                f"Expected exactly {N_RANDOM_PER_USER} random ratings per user, "
                f"got counts {test_counts}"
            )

        for name, matrix in (("train.ascii", train), ("test.ascii", test)):
            observed = set(np.unique(matrix[matrix > 0]).astype(int).tolist())
            if not observed <= {1, 2, 3, 4, 5}:
                raise ValueError(f"{name} contains ratings outside 1-5: {sorted(observed)}")

        if item_features.shape != (N_ITEMS, 33):
            raise ValueError(f"item_features has shape {item_features.shape}, expected {(N_ITEMS, 33)}")
        if user_features.shape != (N_USERS, 14):
            raise ValueError(f"user_features has shape {user_features.shape}, expected {(N_USERS, 14)}")
        if len(item_names) != item_features.shape[1]:
            raise ValueError(
                f"item_features_map has {len(item_names)} names for "
                f"{item_features.shape[1]} columns"
            )
        if len(user_names) != user_features.shape[1]:
            raise ValueError(
                f"user_features_map has {len(user_names)} names for "
                f"{user_features.shape[1]} columns"
            )

    @staticmethod
    def _group_index(names: list[str], expected_groups: tuple[str, ...]) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for index, name in enumerate(names):
            groups.setdefault(name.split(":", 1)[0], []).append(index)
        missing = set(expected_groups) - set(groups)
        unexpected = set(groups) - set(expected_groups)
        if missing or unexpected:
            raise ValueError(
                f"Feature map groups do not match expectation. Missing: {sorted(missing)}; "
                f"unexpected: {sorted(unexpected)}"
            )
        return groups

    @staticmethod
    def _check_one_hot(
        features: np.ndarray,
        groups: dict[str, list[int]],
        label: str,
    ) -> None:
        """Every covariate group must be an exact one-hot.

        The value lookup below is an argmax, which would silently pick a winner
        from an all-zero or multi-hot row.
        """

        for group, columns in groups.items():
            sums = np.unique(features[:, columns].sum(axis=1)).tolist()
            if sums != [1.0]:
                raise ValueError(
                    f"{label} covariate group {group!r} is not one-hot; row sums {sums}"
                )

    @classmethod
    def _attribute(
        cls,
        features: np.ndarray,
        names: list[str],
        groups: dict[str, list[int]],
        row: int,
        group: str,
    ) -> str:
        columns = groups[group]
        winner = columns[int(np.argmax(features[row, columns]))]
        return names[winner].split(":", 1)[1]

    @classmethod
    def _build_items(
        cls,
        features: np.ndarray,
        names: list[str],
        groups: dict[str, list[int]],
    ) -> pd.DataFrame:
        records = []
        for item_index in range(N_ITEMS):
            raw_gender = cls._attribute(features, names, groups, item_index, "gender")
            raw_style = cls._attribute(features, names, groups, item_index, "jackettype")
            raw_colour = cls._attribute(features, names, groups, item_index, "color")
            on_front_page = cls._attribute(features, names, groups, item_index, "onfrontpage")

            style = STYLE_SINGULAR.get(raw_style, raw_style)
            colour = COLOUR_ALIASES.get(raw_colour, raw_colour)
            records.append(
                {
                    "item_id": str(item_index),
                    "raw_item_id": str(item_index),
                    # `title` is the prose form, kept for readability and for the
                    # Agent4Rec renderer, which leaves position 0 unlabeled. The
                    # three attribute columns carry the same facts individually.
                    "title": ITEM_TITLE_TEMPLATE.format(
                        gender=GENDER_POSSESSIVE[raw_gender],
                        style=style,
                        colour=colour,
                    ),
                    "gender": raw_gender,
                    "style": style,
                    "colour": colour,
                    "genres": f"{raw_gender}|{style}|{colour}",
                    "on_front_page": on_front_page,
                }
            )
        items = pd.DataFrame.from_records(records)
        return items.astype(
            {
                "item_id": "string",
                "raw_item_id": "string",
                "title": "string",
                "gender": "string",
                "style": "string",
                "colour": "string",
                "genres": "string",
                "on_front_page": "string",
            }
        )

    @classmethod
    def _build_users(
        cls,
        features: np.ndarray,
        names: list[str],
        groups: dict[str, list[int]],
    ) -> pd.DataFrame:
        records = [
            {
                "user_id": str(user_index),
                "raw_user_id": str(user_index),
                "gender": cls._attribute(features, names, groups, user_index, "gender"),
                "age_group": cls._attribute(features, names, groups, user_index, "age"),
                "location": cls._attribute(features, names, groups, user_index, "location"),
                "fashion_interest": cls._attribute(
                    features, names, groups, user_index, "fashioninterest"
                ),
            }
            for user_index in range(N_USERS)
        ]
        users = pd.DataFrame.from_records(records)
        return users.astype({column: "string" for column in users.columns})

    @staticmethod
    def _build_interactions(
        train: np.ndarray,
        test: np.ndarray,
        propensities: np.ndarray,
    ) -> pd.DataFrame:
        frames = []
        for source, matrix in ((SOURCE_SELF_SELECTED, train), (SOURCE_RANDOM, test)):
            rows, columns = np.where(matrix > 0)
            frame = pd.DataFrame(
                {
                    "interaction_id": [
                        f"coat:{source}:u{user:03d}:i{item:03d}"
                        for user, item in zip(rows, columns, strict=True)
                    ],
                    "user_id": [str(user) for user in rows],
                    "item_id": [str(item) for item in columns],
                    "event_type": "rating",
                    "rating": matrix[rows, columns].astype(int),
                    "source": source,
                    # The shipped propensities model observation in the
                    # self-selected half only; pool assignment was uniform, so
                    # the value carries no meaning for random rows.
                    "propensity": (
                        propensities[rows, columns]
                        if source == SOURCE_SELF_SELECTED
                        else np.nan
                    ),
                }
            )
            frames.append(frame)

        interactions = pd.concat(frames, ignore_index=True)
        interactions["target_interact"] = pd.Series(
            1, index=interactions.index, dtype="int8"
        )
        interactions["target_like_ge4"] = interactions["rating"].ge(4).astype("int8")
        interactions["target_rating"] = interactions["rating"]
        interactions = interactions.astype(
            {
                "interaction_id": "string",
                "user_id": "string",
                "item_id": "string",
                "event_type": "string",
                "rating": "Int64",
                "source": "string",
                "propensity": "float64",
                "target_rating": "Int64",
            }
        )
        return interactions[
            [
                "interaction_id",
                "user_id",
                "item_id",
                "event_type",
                "rating",
                "source",
                "propensity",
                "target_interact",
                "target_like_ge4",
                "target_rating",
            ]
        ]
