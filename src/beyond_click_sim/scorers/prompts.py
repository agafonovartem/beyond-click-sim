"""Backward-compatible imports for generic history prompt templates."""

from beyond_click_sim.scorers.history.prompts import (
    INTERACTION_YES_NO_SYSTEM_PROMPT,
    INTERACTION_YES_NO_USER_PROMPT_TEMPLATE,
    REGRESSION_SYSTEM_PROMPT,
    REGRESSION_USER_PROMPT_TEMPLATE,
)

__all__ = [
    "INTERACTION_YES_NO_SYSTEM_PROMPT",
    "INTERACTION_YES_NO_USER_PROMPT_TEMPLATE",
    "REGRESSION_SYSTEM_PROMPT",
    "REGRESSION_USER_PROMPT_TEMPLATE",
]
