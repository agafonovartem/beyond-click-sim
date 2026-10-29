"""Paper templates A/T3, with candidate labels and a Steam domain adaptation."""

from __future__ import annotations

import json


PROMPT_VERSION = "llmsrec_syn_paper_labels_v1"
AUTHOR_REVISION = "1272c62785d920c406d7233c772a704e960c224a"


def example(history: list[str], candidates: list[str], *, dataset: str) -> str:
    if dataset == "ml-1m":
        profile, history_name, noun = "Movie", "Watched Movies", "movies"
        basis = "watched movies"
    elif dataset == "steam":
        profile, history_name, noun = "Game", "Reviewed Games", "games"
        basis = "reviewed games"
    else:
        raise ValueError("LLMSRec-Syn supports ml-1m and steam prompts")
    history_text = json.dumps(
        [f"{i}. {title}" for i, title in enumerate(history)],
        ensure_ascii=False,
    )
    candidate_text = "\n".join(
        f"C{i}. {json.dumps(title, ensure_ascii=False)}"
        for i, title in enumerate(candidates, start=1)
    )
    return (
        f"The User's {profile} Profile:\n- {history_name}: {history_text}\n\n"
        f"The User's Potential Matches:\n- Candidate {profile}s:\n{candidate_text}\n\n"
        f"Based on the user's {basis}, please rank the candidate {noun} that align "
        "closely with the user's preferences.\n"
        f"- You ONLY rank the given Candidate {profile}s.\n"
        "- Do not add history items or items outside the candidate list.\n\n"
        f"Return exactly {len(candidates)} candidate labels, ranked from most to least "
        "recommended, one label per line.\n"
        f"Use every label C1 through C{len(candidates)} exactly once.\n"
        "Do not output item names, rank numbers, explanations or additional labels.\n"
        "Answer:\n"
    )


def messages(
    *,
    demo_history: list[str],
    demo_candidates: list[str],
    demo_ranking: list[str],
    target_history: list[str],
    target_candidates: list[str],
    dataset: str,
) -> list[dict[str, str]]:
    prompt = (
        "Aggregated Demonstration Example:\n\n"
        + example(demo_history, demo_candidates, dataset=dataset)
        + "\n".join(demo_ranking)
        + "\n\nTest User:\n\n"
        "Learn from the above demonstration example to solve the following test example.\n\n"
        + example(target_history, target_candidates, dataset=dataset)
    )
    return [{"role": "user", "content": prompt}]
