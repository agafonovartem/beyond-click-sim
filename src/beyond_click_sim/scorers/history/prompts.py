from __future__ import annotations

from typing import Literal


HistoryPromptFamily = Literal["simulator", "openp5_style"]

SIMULATOR_PROMPT_VERSION = "simulator_v1"
OPENP5_STYLE_PROMPT_VERSION = "openp5_style_iid_v1"
OPENP5_STYLE_PROMPT_SOURCE = (
    "https://github.com/JiangDeccc/EvaLLMasRecommender/"
    "blob/main/eva/templates/template.txt"
)


def validate_history_prompt_family(prompt_family: str) -> HistoryPromptFamily:
    """Validate and narrow the configured history-conditioned prompt family."""

    if prompt_family not in ("simulator", "openp5_style"):
        raise ValueError(
            "prompt_family must be 'simulator' or 'openp5_style', "
            f"got {prompt_family!r}"
        )
    return prompt_family


def history_prompt_version(prompt_family: HistoryPromptFamily) -> str:
    """Return the reproducible prompt version recorded in run manifests."""

    if prompt_family == "simulator":
        return SIMULATOR_PROMPT_VERSION
    return OPENP5_STYLE_PROMPT_VERSION


def history_prompt_metadata(
    prompt_family: HistoryPromptFamily,
) -> dict[str, object]:
    """Describe the prompt family and its information-bearing adaptations."""

    metadata: dict[str, object] = {
        "family": prompt_family,
        "version": history_prompt_version(prompt_family),
        "system_role": prompt_family == "simulator",
        "temporal_language": False,
    }
    if prompt_family == "openp5_style":
        metadata["source"] = OPENP5_STYLE_PROMPT_SOURCE
        metadata["adaptations"] = [
            "remove chronological and next-item language for the IID split",
            "replace free-form item generation with the task-provided candidates",
            "adapt parser-safe outputs to yes/no, listwise, and regression scorers",
        ]
    return metadata


def history_prompt_messages(
    *,
    prompt_family: HistoryPromptFamily,
    simulator_system_prompt: str,
    simulator_user_prompt: str,
    openp5_style_user_prompt: str,
) -> list[dict[str, str]]:
    """Build chat messages without adding a role to the neutral prompt family."""

    if prompt_family == "simulator":
        return [
            {"role": "system", "content": simulator_system_prompt},
            {"role": "user", "content": simulator_user_prompt},
        ]
    return [{"role": "user", "content": openp5_style_user_prompt}]


INTERACTION_YES_NO_SYSTEM_PROMPT = """You are simulating a user's response in a recommender system.
Given the user's interaction history and candidate items, decide whether the user would interact with each candidate.
Use only the provided history and candidate information.
Return only one line per candidate in the required format."""


INTERACTION_YES_NO_USER_PROMPT_TEMPLATE = """User interaction history:
{history}

Candidate items:
{candidates}

For each candidate, answer whether the user would interact with it.

Required output format:
Return exactly one line for each candidate label.
The text before each colon must match these labels:
{output_labels}
Fill each line after the colon with either yes or no."""


OPENP5_STYLE_INTERACTION_YES_NO_USER_PROMPT_TEMPLATE = """Considering that the user has interacted with the following items:
{history}

Candidate items:
{candidates}

For each candidate item, predict whether this user would interact with it.

Required output format:
Return exactly one line for each candidate label.
The text before each colon must match these labels:
{output_labels}
Fill each line after the colon with either yes or no.
Do not output any additional text."""


PREFERENCE_YES_NO_SYSTEM_PROMPT = """You are simulating a user's positive preference response in a recommender system.
Given the user's observed feedback history, a positive-preference target, and candidate items, decide whether the user's response to each candidate would meet the target.
Use only the provided history, target definition, and candidate information.
Return only one line per candidate in the required format."""


PREFERENCE_YES_NO_USER_PROMPT_TEMPLATE = """User feedback history:
{history}

Positive-preference target:
{target_description}

Candidate items:
{candidates}

For each candidate, answer whether the user's response would meet the positive-preference target.

Required output format:
Return exactly one line for each candidate label.
The text before each colon must match these labels:
{output_labels}
Fill each line after the colon with either yes or no."""


OPENP5_STYLE_PREFERENCE_YES_NO_USER_PROMPT_TEMPLATE = """Considering the user's observed feedback history:
{history}

Positive-preference target:
{target_description}

Candidate items:
{candidates}

For each candidate, predict whether this user's response would meet the positive-preference target.

Required output format:
Return exactly one line for each candidate label.
The text before each colon must match these labels:
{output_labels}
Fill each line after the colon with either yes or no.
Do not output any additional text."""


INTERACTION_LISTWISE_RANKING_SYSTEM_PROMPT = """You are simulating a user's relative preferences in a recommender system.
Given the user's interaction history and candidate items, rank all candidates from most likely to least likely to be interacted with by the user.
Use only the provided history and candidate information.
The history is context only: never include history items in the ranking.
Do not explain or show your reasoning.
Return only the candidate labels in ranked order, one label per line."""


INTERACTION_LISTWISE_RANKING_USER_PROMPT_TEMPLATE = """User interaction history:
{history}

Candidate items:
{candidates}

Rank every candidate from most likely to least likely to be interacted with by this user.

Required output format:
The complete set of allowed candidate labels is:
{output_labels}
Return exactly one ranked line for each candidate label.
There are exactly {candidate_count} candidates, so your response must contain exactly {candidate_count} non-empty lines.
Use each candidate label exactly once.
Before returning, silently verify that every allowed label appears exactly once. If a label is missing or duplicated, correct the list.
Do not output item names or history items.
Do not include rank numbers, punctuation, explanations, or analysis.
The first character of your response must be C."""


OPENP5_STYLE_INTERACTION_LISTWISE_RANKING_USER_PROMPT_TEMPLATE = """Considering that the user has interacted with the following items:
{history}

Candidate items:
{candidates}

Rank all candidate items from most likely to least likely to be interacted with by this user.

Required output format:
The complete set of allowed candidate labels is:
{output_labels}
Return exactly one ranked line for each candidate label.
There are exactly {candidate_count} candidates, so your response must contain exactly {candidate_count} non-empty lines.
Use each candidate label exactly once.
Before returning, silently verify that every allowed label appears exactly once. If a label is missing or duplicated, correct the list.
Do not output item names or history items.
Do not include rank numbers, punctuation, explanations, or analysis.
The first character of your response must be C."""


PREFERENCE_LISTWISE_RANKING_SYSTEM_PROMPT = """You are simulating a user's relative positive preferences in a recommender system.
Given the user's observed feedback history, a positive-preference target, and candidate items, rank all candidates from most likely to least likely to meet the target.
Use only the provided history, target definition, and candidate information.
The history is context only: never include history items in the ranking.
Do not explain or show your reasoning.
Return only the candidate labels in ranked order, one label per line."""


PREFERENCE_LISTWISE_RANKING_USER_PROMPT_TEMPLATE = """User feedback history:
{history}

Positive-preference target:
{target_description}

Candidate items:
{candidates}

Rank every candidate from most likely to least likely to meet the positive-preference target for this user.

Required output format:
The complete set of allowed candidate labels is:
{output_labels}
Return exactly one ranked line for each candidate label.
There are exactly {candidate_count} candidates, so your response must contain exactly {candidate_count} non-empty lines.
Use each candidate label exactly once.
Before returning, silently verify that every allowed label appears exactly once. If a label is missing or duplicated, correct the list.
Do not output item names or history items.
Do not include rank numbers, punctuation, explanations, or analysis.
The first character of your response must be C."""


OPENP5_STYLE_PREFERENCE_LISTWISE_RANKING_USER_PROMPT_TEMPLATE = """Considering the user's observed feedback history:
{history}

Positive-preference target:
{target_description}

Candidate items:
{candidates}

Rank all candidate items from most likely to least likely to receive a response meeting the positive-preference target from this user.

Required output format:
The complete set of allowed candidate labels is:
{output_labels}
Return exactly one ranked line for each candidate label.
There are exactly {candidate_count} candidates, so your response must contain exactly {candidate_count} non-empty lines.
Use each candidate label exactly once.
Before returning, silently verify that every allowed label appears exactly once. If a label is missing or duplicated, correct the list.
Do not output item names or history items.
Do not include rank numbers, punctuation, explanations, or analysis.
The first character of your response must be C."""


POLICY_RANKING_ITEMWISE_SYSTEM_PROMPT = """You are simulating a user's response to a recommendation.
Given the user's interaction history and one recommended item, predict whether the user would interact with it.
Use only the provided history and item information.
Answer with exactly one word: yes or no."""


POLICY_RANKING_ITEMWISE_USER_PROMPT_TEMPLATE = """User interaction history:
{history}

Recommended item:
{candidate}

Would the user interact with this recommendation?

Answer: yes or no."""


OPENP5_STYLE_POLICY_RANKING_ITEMWISE_USER_PROMPT_TEMPLATE = """Considering that the user has interacted with the following items:
{history}

Recommended item:
{candidate}

Predict whether this user would interact with the recommended item.
Answer with exactly one word: yes or no."""


REGRESSION_SYSTEM_PROMPT = """You are simulating a user's numeric response in a recommender system.
Given the user's interaction history and one candidate item, predict the user's numeric response.
Use only the provided history and candidate information.
Return only the required numeric value."""


REGRESSION_USER_PROMPT_TEMPLATE = """User interaction history:
{history}

Candidate item:
{candidate}

Task:
{target_description}

Required output format:
{output_instructions}"""


OPENP5_STYLE_REGRESSION_USER_PROMPT_TEMPLATE = """Considering the user's observed response history:
{history}

Candidate item:
{candidate}

Prediction target:
{target_description}

Required output format:
{output_instructions}"""
