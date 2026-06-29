from __future__ import annotations

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
You only watch movies which align with your taste.
Use this format: ID: [candidate id]; MOVIE: [movie name]; WATCH: [yes or no]; REASON: [brief reason]
You must judge all the movies. If you don't want to watch a movie, use WATCH: no; REASON: [brief reason]
Each response should be on one line. Do not include any additional information or explanations and stay grounded in reality."""
