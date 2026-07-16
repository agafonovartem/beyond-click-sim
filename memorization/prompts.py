"""Candidate-free recommendation prompt — verbatim from Di Palma et al. (2025), Fig. 3.

Transcribed from the paper source (arXiv-2505.10212v1/Figures/fig_rs_prompt.tex),
not paraphrased. Their generation config (Methodology.tex): temperature=0, top_p=1,
frequency_penalty=0, presence_penalty=0, seed=42.

``PROMPT_VERSION`` is recorded in every run manifest so a wording change is traceable.
"""

from __future__ import annotations

import re

PROMPT_VERSION = "dipalma2025_fig3_verbatim"

SYSTEM_PROMPT = (
    "You are a movie recommendation system for the MovieLens-1M dataset. "
    "Based on the user's past interactions, generate a ranked list of exactly 50 "
    "new movie recommendations. Your output must contain only the list in the "
    "following format: one line per recommendation in the exact format "
    "`Rank. Title' (e.g., `1. Harry Potter'). Do not include any additional text, "
    "commentary, or explanation."
)

USER_PROMPT_TEMPLATE = (
    "User {user_id} has interacted with the following movies: {training_history_str}. "
    "Based solely on these interactions, please generate a ranked list of exactly 50 "
    "movie recommendations. Output only the list with no additional commentary or "
    "explanation. Each recommendation must be on a new line in the exact format: "
    "`Rank. Title' (for example: `1. Harry Potter')."
)

# The paper's prompt hard-codes "exactly 50"; k is kept as a parameter only so the
# templates stay honest if a different cutoff is ever used deliberately.
DEFAULT_K = 50

_LIST_LINE = re.compile(r"^\s*\d+\s*[\.\)\-:]\s*(.+?)\s*$")


def build_messages(user_id: str, history_titles: list[str], *, k: int = DEFAULT_K) -> list[dict]:
    """Build the Fig. 3 chat messages for one user."""
    history_str = ", ".join(history_titles) if history_titles else "(none)"
    system = SYSTEM_PROMPT
    user = USER_PROMPT_TEMPLATE.format(user_id=user_id, training_history_str=history_str)
    if k != DEFAULT_K:
        system = system.replace("exactly 50", f"exactly {k}")
        user = user.replace("exactly 50", f"exactly {k}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_numbered_list(text: str, *, max_items: int = DEFAULT_K) -> list[str]:
    """Extract titles from the `Rank. Title' list; fall back to non-empty lines.

    Preserves rank order, strips the leading enumerator, and keeps at most
    ``max_items`` items.
    """
    if not text:
        return []
    items: list[str] = []
    for line in text.splitlines():
        match = _LIST_LINE.match(line)
        if match:
            title = match.group(1).strip().strip("*").strip()
            if title:
                items.append(title)
    if not items:
        # Model ignored the numbered-list instruction — take non-empty lines.
        items = [ln.strip("-* \t") for ln in text.splitlines() if ln.strip()]
    return items[:max_items]
