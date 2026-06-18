# Planned LLM Model Set

Date: 2026-06-18

This file records the planned LLM set for the multi-task LLM user simulator evaluation. The ranking is a priority order for inclusion in the paper, not a general-purpose model leaderboard.

## Main Set

| Rank | Model | Status | Role in the study | Notes |
| --- | --- | --- | --- | --- |
| 1 | `meta-llama/Llama-3.3-70B-Instruct` | core | Strong open-weight anchor | Already used in the current vLLM runs; keep as the main open-weight reference point. |
| 2 | `Qwen/Qwen3.6-27B` | core | Modern dense mid-size Qwen | Use with reasoning/thinking disabled for comparability with non-reasoning simulators. |
| 3 | `Qwen/Qwen3.6-35B-A3B` | core | Modern Qwen MoE | Lets us compare dense Qwen against a sparse/MoE Qwen at a similar practical scale. Use with reasoning/thinking disabled. |
| 4 | `Qwen/Qwen3-8B` | core | Small open-weight model | Small-model point in the scaling grid; preferred over Llama 3.1 8B because it is newer and keeps the small-model comparison in the Qwen family. |
| 5 | `gpt-5.4-mini` | core | Proprietary mini model | Practical API baseline for a cheap/fast hosted simulator. Verify the exact API alias before final runs. |
| 6 | `gpt-5.5` | core | Proprietary large model | Hosted upper-bound model. If cost is prohibitive, run on a reduced matrix or replace with the strongest affordable OpenAI model. |
| 7 | `claude-sonnet-4-6` | optional | Anthropic competitor | Valuable because Claude is a visible model family for user-preference simulation; include if budget allows. Verify the exact API alias before final runs. |
| 8 | `gemini-2.5-flash` | optional | Google competitor | Useful third-provider point; lower priority than Claude for the main narrative. Verify the exact API alias before final runs. |

## Experimental Rationale

The intended model grid is:

```text
small open-weight -> mid dense open-weight -> mid MoE open-weight -> large open-weight -> proprietary mini -> proprietary large -> optional external providers
Qwen 8B           -> Qwen 27B              -> Qwen 35B-A3B         -> Llama 70B         -> OpenAI mini       -> OpenAI large       -> Claude/Gemini
```

This avoids over-sampling near-duplicate models from the same family while still testing the axes that matter for the paper:

- scale among open-weight models;
- dense vs MoE Qwen at a comparable practical tier;
- open-weight vs hosted proprietary models;
- provider sensitivity through optional Anthropic and Google models.

## Qwen Reasoning Configuration

For Qwen models with thinking/reasoning support, final evaluation should disable reasoning unless a separate reasoning ablation is explicitly planned. The goal is to compare them as direct user simulators under a consistent prompt-and-response protocol.

For vLLM/SGLang OpenAI-compatible serving, use the chat template setting when supported:

```python
extra_body = {
    "chat_template_kwargs": {
        "enable_thinking": False,
    },
}
```

Provider-specific endpoints may use a different parameter name, so each final run manifest must record the exact serving backend, model id, and reasoning/thinking setting.

## Source Links

- Llama 3.3 70B: <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>
- Qwen3.6 27B: <https://huggingface.co/Qwen/Qwen3.6-27B>
- Qwen3.6 35B-A3B: <https://huggingface.co/Qwen/Qwen3.6-35B-A3B>
- Qwen3 8B: <https://huggingface.co/Qwen/Qwen3-8B>
- OpenAI models: <https://platform.openai.com/docs/models>
- Anthropic models: <https://docs.anthropic.com/en/docs/about-claude/models/overview>
- Gemini API pricing/models: <https://ai.google.dev/gemini-api/docs/pricing>
