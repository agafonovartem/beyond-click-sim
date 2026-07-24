from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import unicodedata


Message = dict[str, str]
TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class CostEstimate:
    n_prompts: int
    input_tokens: int
    expected_output_tokens: int
    maximum_output_tokens: int
    input_price_per_million: float
    output_price_per_million: float
    expected_cost_usd: float
    maximum_cost_usd: float
    safety_factor: float
    guarded_cost_usd: float
    token_counter: str
    prompt_plan_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def estimate_prompt_plan_cost(
    prompts: Sequence[Sequence[Message]],
    *,
    max_completion_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float,
    expected_output_tokens: int = 8,
    safety_factor: float = 1.10,
    token_counter: TokenCounter | None = None,
    token_counter_name: str | None = None,
) -> CostEstimate:
    """Price every materialized prompt before any client is constructed.

    Provider usage remains authoritative. The default counter is deliberately
    conservative for native-Chinese text and requires no network download.
    Tests and production wrappers may inject a provider tokenizer.
    """

    if max_completion_tokens < 1 or expected_output_tokens < 0:
        raise ValueError("Invalid completion-token budget")
    if safety_factor < 1:
        raise ValueError("safety_factor must be at least 1")
    if input_price_per_million < 0 or output_price_per_million < 0:
        raise ValueError("Token prices must be non-negative")
    counter = token_counter or conservative_unicode_token_count
    counter_name = token_counter_name or (
        "conservative_unicode_v1"
        if token_counter is None
        else getattr(counter, "__name__", "injected")
    )
    input_tokens = sum(_chat_tokens(messages, counter) for messages in prompts)
    n_prompts = len(prompts)
    expected_output = n_prompts * expected_output_tokens
    maximum_output = n_prompts * max_completion_tokens
    expected_cost = token_cost(
        input_tokens,
        expected_output,
        input_price_per_million,
        output_price_per_million,
    )
    maximum_cost = token_cost(
        input_tokens,
        maximum_output,
        input_price_per_million,
        output_price_per_million,
    )
    return CostEstimate(
        n_prompts=n_prompts,
        input_tokens=input_tokens,
        expected_output_tokens=expected_output,
        maximum_output_tokens=maximum_output,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        expected_cost_usd=expected_cost,
        maximum_cost_usd=maximum_cost,
        safety_factor=safety_factor,
        guarded_cost_usd=maximum_cost * safety_factor,
        token_counter=counter_name,
        prompt_plan_sha256=_prompt_plan_sha256(prompts),
    )


def enforce_hard_budget(
    estimate: CostEstimate,
    *,
    hard_budget_usd: float,
) -> None:
    if hard_budget_usd <= 0:
        raise ValueError("hard_budget_usd must be positive")
    if estimate.guarded_cost_usd > hard_budget_usd:
        raise RuntimeError(
            "LLM preflight exceeds hard budget before any API call: "
            f"${estimate.guarded_cost_usd:.4f} > ${hard_budget_usd:.4f}; "
            "reduce users, cells, seeds, prompt length, or completion tokens"
        )


def conservative_unicode_token_count(text: str) -> int:
    """Offline upper-biased approximation for mixed English/Chinese prompts."""

    ascii_count = sum(ord(char) < 128 for char in text)
    non_ascii_count = len(text) - ascii_count
    combining = sum(bool(unicodedata.combining(char)) for char in text)
    # English prose is usually close to four characters/token; using three is
    # conservative. Native Chinese is often near one token/character in modern
    # OpenAI tokenizers; 1.2 leaves margin for uncommon characters.
    return max(
        1,
        math.ceil(ascii_count / 3)
        + math.ceil(non_ascii_count * 1.2)
        + combining,
    )


def token_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float,
) -> float:
    return (
        input_tokens * input_price_per_million
        + output_tokens * output_price_per_million
    ) / 1_000_000


def _chat_tokens(
    messages: Sequence[Message],
    counter: TokenCounter,
) -> int:
    # Explicitly record the framing convention in the counter implementation.
    return 2 + sum(6 + counter(message["content"]) for message in messages)


def _prompt_plan_sha256(prompts: Sequence[Sequence[Message]]) -> str:
    payload = json.dumps(
        prompts,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
