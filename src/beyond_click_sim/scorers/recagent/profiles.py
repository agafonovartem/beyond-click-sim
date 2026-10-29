"""Training-only MovieLens profile and event serialization."""

from __future__ import annotations

import math
from numbers import Real

import pandas as pd

from beyond_click_sim.scorers.history.common import _format_prompt_value


AGES = {
    1: "under 18",
    18: "18–24",
    25: "25–34",
    35: "35–44",
    45: "45–49",
    50: "50–55",
    56: "56 or older",
}
OCCUPATIONS = dict(
    enumerate(
        (
            "other or not specified",
            "academic/educator",
            "artist",
            "clerical/admin",
            "college/graduate student",
            "customer service",
            "doctor/health care",
            "executive/managerial",
            "farmer",
            "homemaker",
            "K-12 student",
            "lawyer",
            "programmer",
            "retired",
            "sales/marketing",
            "scientist",
            "self-employed",
            "technician/engineer",
            "tradesman/craftsman",
            "unemployed",
            "writer",
        )
    )
)
HISTORY_COLUMNS = ("item_title", "item_genres", "rating")
CANDIDATE_COLUMNS = ("item_title", "item_genres")
ALLOWED_CANDIDATE_COLUMNS = frozenset(
    (*CANDIDATE_COLUMNS, "item_summary", "item_rating_mean", "item_rating_count")
)
ALLOWED_HISTORY_COLUMNS = frozenset(
    (*HISTORY_COLUMNS, "item_rating_mean", "item_rating_count")
)


def unix_time(value) -> float:
    if pd.isna(value):
        raise ValueError("Null historical timestamp")
    if isinstance(value, Real):
        result = float(value)
    else:
        result = pd.to_datetime(value, utc=True).timestamp()
    if not math.isfinite(result):
        raise ValueError("Non-finite historical timestamp")
    return result


def utc_time(value: float) -> str:
    return pd.Timestamp(value, unit="s", tz="UTC").isoformat()


def demographics(rows: pd.DataFrame, *, enabled: bool) -> dict[str, str]:
    result = {"name": "Movie user"}
    if not enabled:
        return result
    values = {}
    for column in ("user_gender", "user_age", "user_occupation"):
        if (
            column not in rows
            or rows[column].isna().any()
            or rows[column].nunique() != 1
        ):
            raise ValueError(f"Missing or inconsistent recorded demographics: {column}")
        values[column] = rows[column].iloc[0]
    try:
        if any(
            float(values[column]) != int(values[column])
            for column in ("user_age", "user_occupation")
        ):
            raise ValueError("Demographic category codes must be integers")
        result.update(
            gender={"F": "female", "M": "male"}[values["user_gender"]],
            age_category=AGES[int(values["user_age"])],
            occupation=OCCUPATIONS[int(values["user_occupation"])],
        )
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError("Unrecognized MovieLens demographic category") from error
    return result


def describe(row: pd.Series, columns: tuple[str, ...]) -> str:
    return "; ".join(
        f"{column}: {_format_prompt_value(row[column], parse_json_list=column == 'item_genres')}"
        for column in columns
    )
