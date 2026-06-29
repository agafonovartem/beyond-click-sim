from __future__ import annotations


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
