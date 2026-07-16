from __future__ import annotations

import json
from numbers import Integral, Real
from typing import Any


def _chat_completion_text(response: Any) -> str:
    choice = response.choices[0]
    content = choice.message.content
    if content is None:
        raise ValueError("Chat completion response has no text content")
    return str(content)


def _format_prompt_value(value: Any, *, parse_json_list: bool = False) -> str:
    """Return compact text for scalar values shown in LLM prompts."""

    if parse_json_list and isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(parsed, list):
            return ", ".join(str(item) for item in parsed) if parsed else "none"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.2f}"
    return str(value)
