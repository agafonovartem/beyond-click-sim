from __future__ import annotations

from collections.abc import Mapping, Sequence

# it is movielens-only for now.
_AGENT4REC_SOCIAL_TRAITS_PREFIX = (
    "You excel at role-playing. Picture yourself as a user exploring a movie recommendation system. You have the following social traits:"
)
_AGENT4REC_ACTIVITY_EXPLANATION = (
    "The activity characteristic pertains to the frequency of your movie-watching habits."
)
_AGENT4REC_CONFORMITY_EXPLANATION = (
    "The conformity characteristic measures the degree to which your ratings are influenced by historical ratings."
)
_AGENT4REC_DIVERSITY_EXPLANATION = (
    "The diversity characteristic gauges your likelihood of watching movies that may not align with your usual taste."
)
_AGENT4REC_TASTE_USER_INSTRUCTION = (
    "You only watch movies which align with your taste."
)
_AGENT4REC_PROFILE_USER_INSTRUCTION = (
    "Judge each movie using your available profile and the candidate information."
)
_AGENT4REC_TASTE_RATING_INSTRUCTION = (
    "Rate the movie according to your taste."
)
_AGENT4REC_PROFILE_RATING_INSTRUCTION = (
    "Rate the movie using your available profile and the candidate information."
)
# prefixes are different same to original code
AGENT4REC_FORCED_ITEMS_SYSTEM_PROMPT_TEMPLATE = (
    "Assume you are a user browsing movie recommendation system who has the following characteristics: "
    "\nYour movie tastes are: {taste}. "
)


AGENT4REC_SOCIAL_TRAITS_SYSTEM_PROMPT_TEMPLATE = (
    _AGENT4REC_SOCIAL_TRAITS_PREFIX
    + "\nYour activity trait is described as: {activity}"
    + "\nYour conformity trait is described as: {conformity}"
    + "\nYour diversity trait is described as: {diversity}"
    + "\nBeyond that, your movie tastes are: {taste}. "
    + "\n"
    + _AGENT4REC_ACTIVITY_EXPLANATION
    + " "
    + _AGENT4REC_CONFORMITY_EXPLANATION
    + " "
    + _AGENT4REC_DIVERSITY_EXPLANATION
)


def agent4rec_system_prompt(
    *,
    taste: str | None,
    activity: str | None,
    conformity: str | None,
    diversity: str | None,
) -> str:
    """Build an Agent4Rec-style system prompt from available profile parts."""

    has_activity = bool(activity)
    has_conformity = bool(conformity)
    has_diversity = bool(diversity)
    has_traits = has_activity or has_conformity or has_diversity
    if not has_traits:
        return AGENT4REC_FORCED_ITEMS_SYSTEM_PROMPT_TEMPLATE.format(taste=taste or "")

    parts = [_AGENT4REC_SOCIAL_TRAITS_PREFIX]
    explanations: list[str] = []
    if activity:
        parts.append(f"\nYour activity trait is described as: {activity}")
        explanations.append(_AGENT4REC_ACTIVITY_EXPLANATION)
    if conformity:
        parts.append(f"\nYour conformity trait is described as: {conformity}")
        explanations.append(_AGENT4REC_CONFORMITY_EXPLANATION)
    if diversity:
        parts.append(f"\nYour diversity trait is described as: {diversity}")
        explanations.append(_AGENT4REC_DIVERSITY_EXPLANATION)
    if taste:
        parts.append(f"\nBeyond that, your movie tastes are: {taste}. ")
    if explanations:
        parts.append("\n" + " ".join(explanations))
    return "".join(parts)


AGENT4REC_FORCED_ITEMS_USER_PROMPT_TEMPLATE = """##recommended list##
{candidates}
Please judge all movies in the ##recommended list## and explain why.
{profile_instruction}
Use this format: ID: [candidate id]; MOVIE: [movie name]; WATCH: [yes or no]; REASON: [brief reason]
You must judge all the movies. If you don't want to watch a movie, use WATCH: no; REASON: [brief reason]
Each response should be on one line. Do not include any additional information or explanations and stay grounded in reality."""


def agent4rec_user_prompt(*, candidates: str, taste: str | None) -> str:
    """Build the Agent4Rec forced-items user prompt for available profile parts."""

    profile_instruction = (
        _AGENT4REC_TASTE_USER_INSTRUCTION
        if taste
        else _AGENT4REC_PROFILE_USER_INSTRUCTION
    )
    return AGENT4REC_FORCED_ITEMS_USER_PROMPT_TEMPLATE.format(
        candidates=candidates,
        profile_instruction=profile_instruction,
    )


AGENT4REC_RATING_USER_PROMPT_TEMPLATE = """##movie##
{candidate}
{target_description}
{profile_instruction}
Use this format: RATING: [integer from 1 to 5]
Do not include any additional information or explanations and stay grounded in reality."""


def agent4rec_rating_user_prompt(
    *,
    candidate: str,
    taste: str | None,
    target_description: str | None = None,
) -> str:
    """Build the Agent4Rec rating-prediction user prompt."""

    profile_instruction = (
        _AGENT4REC_TASTE_RATING_INSTRUCTION
        if taste
        else _AGENT4REC_PROFILE_RATING_INSTRUCTION
    )
    return AGENT4REC_RATING_USER_PROMPT_TEMPLATE.format(
        candidate=candidate,
        target_description=(
            target_description
            or "Please predict the rating you would give to this movie from 1 to 5."
        ),
        profile_instruction=profile_instruction,
    )


AGENT4REC_TASTE_PROMPT_VERSION = "agent4rec_modify_v1"

AGENT4REC_TASTE_SYSTEM_PROMPT = """
I want you to act as an agent. You will act as a movie taste analyst roleplaying the user using the first person pronoun "I".
"""

AGENT4REC_TASTE_MODIFY_USER_PROMPT_TEMPLATE = """
Given a user's rating history:

user gives a rating of 1 for following movies: {rating_1_movies}
user gives a rating of 2 for following movies: {rating_2_movies}
user gives a rating of 3 for following movies: {rating_3_movies}
user gives a rating of 4 for following movies: {rating_4_movies}
user gives a rating of 5 for following movies: {rating_5_movies}

My first request is "I need help creating movie taste for a user given the movie-rating history. (in no particular order)"  Generate as many TASTE-REASON pairs as possible, taste should focus on the movies' genres.
Strictly follow the output format below:

TASTE: <-descriptive taste->
REASON: <-brief reason->

TASTE: <-descriptive taste->
REASON: <-brief reason->
.....

Secondly, analyze user tend to give what kinds of movies high ratings, and tend to give what kinds of movies low ratings.
Strictly follow the output format below:
HIGH RATINGS: <-conclusion of movies of high ratings(above 3)->
LOW RATINGS: <-conclusion of movies of low ratings(below 2)->
Answer should not be a combination of above two parts and not contain other words and should not contain movie names.


"""


def agent4rec_taste_modify_user_prompt(
    *,
    rating_movies: Mapping[int, Sequence[str]],
) -> str:
    """Build the Agent4Rec `prompt_modify` taste-generation prompt."""

    return AGENT4REC_TASTE_MODIFY_USER_PROMPT_TEMPLATE.format(
        rating_1_movies=_format_rating_movies(rating_movies.get(1, ())),
        rating_2_movies=_format_rating_movies(rating_movies.get(2, ())),
        rating_3_movies=_format_rating_movies(rating_movies.get(3, ())),
        rating_4_movies=_format_rating_movies(rating_movies.get(4, ())),
        rating_5_movies=_format_rating_movies(rating_movies.get(5, ())),
    )


def _format_rating_movies(titles: Sequence[str]) -> str:
    clean_titles = [str(title).strip() for title in titles if str(title).strip()]
    if not clean_titles:
        return "None"
    return "; ".join(clean_titles)
