from __future__ import annotations


DEFAULT_MAX_ERROR_RATE = 0.10
DEFAULT_MIN_GROUPS_BEFORE_CHECK = 20


class LLMErrorRateExceededError(RuntimeError):
    """Raised when LLM scoring errors exceed the allowed rate for a run.

    Carries the partial error log so the caller can persist it (e.g. to
    llm_errors.jsonl) for post-mortem before the run aborts.
    """

    def __init__(
        self,
        message: str,
        *,
        errors: list[dict[str, object]],
        attempted: int,
        total: int,
    ) -> None:
        super().__init__(message)
        self.errors = errors
        self.attempted = attempted
        self.total = total


def check_error_budget(
    *,
    errors: list[dict[str, object]],
    attempted: int,
    total: int,
    method_name: str,
    task_name: str,
    max_error_rate: float = DEFAULT_MAX_ERROR_RATE,
    min_groups_before_check: int = DEFAULT_MIN_GROUPS_BEFORE_CHECK,
    force: bool = False,
    max_examples: int = 5,
) -> None:
    """Raise `LLMErrorRateExceededError` if the error rate exceeds `max_error_rate`.

    Mid-run (`force=False`): a no-op until `attempted >= min_groups_before_check`,
    so a handful of early failures cannot false-trigger an abort.

    End-of-run (`force=True`): always checked regardless of `attempted`, so a
    small/smoke run with a high failure rate still gets validated even if it
    never reached the minimum sample.
    """

    if not errors or attempted == 0:
        return
    if not force and attempted < min_groups_before_check:
        return

    error_rate = len(errors) / attempted
    if error_rate <= max_error_rate:
        return

    examples = errors[:max_examples]
    examples_text = "\n".join(
        f"  [{index}] candidate_group={example.get('candidate_group')!r} "
        f"attempts={example.get('attempts')}\n"
        + "\n".join(f"      {line}" for line in example.get("errors", []))
        for index, example in enumerate(examples, start=1)
    )
    raise LLMErrorRateExceededError(
        f"{method_name} on task {task_name!r}: LLM scoring error rate "
        f"{error_rate:.1%} ({len(errors)}/{attempted} groups attempted"
        f"{'' if force else f' out of {total} total'}) exceeds "
        f"max_error_rate={max_error_rate:.0%}. Aborting to avoid wasting further "
        f"API calls on a run that is already invalid.\n"
        f"First {len(examples)} failing group(s) (see llm_errors.jsonl for all):\n"
        f"{examples_text}",
        errors=errors,
        attempted=attempted,
        total=total,
    )
