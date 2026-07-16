"""Candidate-free ranking metrics and title matching.

A candidate-free recommender emits, per user, an ordered list of items (classical
policies) or of movie titles (LLMs). We score that list against the user's held-out
test items with HR@K / nDCG@K, macro-averaged over users.

Classical policies are scored by exact item id (`exact_hit_flags`). LLMs emit free
text, so their titles must be matched against held-out titles (`hit_flags`), and
the matching rule is itself a research variable — Di Palma et al.'s rule has three
defects, so we implement it exactly plus a corrected variant:

  paper       Their released rule (src/evaluate_recommendations.py):
              normalize = drop parenthesised year + punctuation + lowercase;
              hit = any(fuzz.ratio(rec, held_out) >= threshold).
              Two consequences, reproduced deliberately:
                * NO dedup — one held-out title can be credited by many
                  recommendations, inflating DCG;
                * the trailing article is NOT moved, so MovieLens' "Matrix, The
                  (1999)" -> "matrix the" never matches an LLM's "The Matrix" ->
                  "the matrix" (fuzz.ratio = 60). 20.8% of the ML-1M catalog has a
                  trailing article, skewed toward popular films.
  article     paper + move the trailing article before matching.
  dedup       paper + greedy matching that consumes each held-out title once, so
              hits cannot exceed the number of held-out items.
  in_catalog  paper + drop generated titles that match no ML-1M item (removes
              hallucinations and post-2000 films; no title is special-cased).
  fair        all three corrections together.

The corrections push in opposite directions: `article` and `in_catalog` raise the
score, `dedup` lowers it (it only affects nDCG — HR is a "any hit" indicator).

Threshold: the paper never documents one. Their code is self-contradictory (dead
default 80, function default 0.85, __main__ 1.0 = exact). 85 reproduces their
Table 3 (MAE 0.0056) and is the default here — see README.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np
from rapidfuzz import fuzz, process

DEFAULT_KS: tuple[int, ...] = (1, 5, 10)
DEFAULT_THRESHOLD = 85.0

# fuzz.ratio (Levenshtein) is what they use, and it is the right choice: WRatio's
# partial/token-set components produce false matches on short shared tokens (e.g.
# "The Third Man" ~= "Ghost Dog: The Way of the Samurai" scores 86), which inflates
# Recall several-fold. Kept selectable for sensitivity analysis only.
SCORERS = {
    "ratio": fuzz.ratio,
    "token_sort": fuzz.token_sort_ratio,
    "token_set": fuzz.token_set_ratio,
    "QRatio": fuzz.QRatio,
    "WRatio": fuzz.WRatio,
}
DEFAULT_SCORER = "ratio"

MATCHING_MODES: dict[str, dict[str, bool]] = {
    "paper": {"article": False, "dedup": False, "in_catalog": False},
    "article": {"article": True, "dedup": False, "in_catalog": False},
    "dedup": {"article": False, "dedup": True, "in_catalog": False},
    "in_catalog": {"article": False, "dedup": False, "in_catalog": True},
    "fair": {"article": True, "dedup": True, "in_catalog": True},
}

_PAREN_RE = re.compile(r"\s*\(.*?\)")
_PUNCT_RE = re.compile(r"[^\w\s]")
_ARTICLE_RE = re.compile(r"^(.*),\s+(the|a|an)$", re.IGNORECASE)


def normalize_title_paper(title: str | None) -> str:
    """Di Palma et al.'s normalize_title: drop year + punctuation, lowercase.

    "Matrix, The (1999)" -> "matrix the". Note this also collapses remakes
    ("King Kong (1933|1976|2005)" -> "king kong") — a property of their protocol.
    """
    if title is None:
        return ""
    cleaned = _PAREN_RE.sub("", str(title)).strip().lower()
    return _PUNCT_RE.sub("", cleaned)


def normalize_title_fixed(title: str | None) -> str:
    """Their normalization plus the article fix: "Matrix, The" -> "the matrix"."""
    if title is None:
        return ""
    s = _PAREN_RE.sub("", str(title)).strip()
    if (m := _ARTICLE_RE.match(s)):
        s = f"{m.group(2)} {m.group(1)}"
    return _PUNCT_RE.sub("", s.lower().strip())


def normalizer_for(mode: str):
    """Return the title normalizer used by a matching mode."""
    return normalize_title_fixed if MATCHING_MODES[mode]["article"] else normalize_title_paper


def hit_flags(
    ranked_titles: Sequence[str],
    test_titles: Iterable[str],
    *,
    mode: str = "paper",
    threshold: float = DEFAULT_THRESHOLD,
    catalog: Sequence[str] | None = None,
    scorer: str = DEFAULT_SCORER,
    catalog_cache: dict[str, bool] | None = None,
) -> list[bool]:
    """Per-rank hit vector for an LLM's generated titles under a matching mode.

    ``catalog`` is required by modes with ``in_catalog`` and must already be
    normalized with ``normalizer_for(mode)``. Pass a shared ``catalog_cache`` dict
    across users to memoize the catalog lookup: LLMs recommend the same popular
    titles to most users, so this turns ~300k lookups per run into a few thousand.
    """
    if mode not in MATCHING_MODES:
        raise ValueError(f"unknown matching mode: {mode!r} (have {list(MATCHING_MODES)})")
    opts = MATCHING_MODES[mode]
    if opts["in_catalog"] and catalog is None:
        raise ValueError(f"mode {mode!r} requires a normalized `catalog`")

    scorer_fn = SCORERS[scorer]
    norm = normalizer_for(mode)
    test_norm = [norm(t) for t in test_titles]
    queries = [norm(t) for t in ranked_titles]

    if opts["in_catalog"]:
        cache = catalog_cache if catalog_cache is not None else {}
        kept: list[str] = []
        for q in queries:
            if not q:
                continue
            known = cache.get(q)
            if known is None:
                known = process.extractOne(
                    q, catalog, scorer=scorer_fn, score_cutoff=threshold
                ) is not None
                cache[q] = known
            if known:
                kept.append(q)
        queries = kept

    flags: list[bool] = []
    if opts["dedup"]:
        remaining = list(test_norm)
        for q in queries:
            if not q or not remaining:
                flags.append(False)
                continue
            match = process.extractOne(q, remaining, scorer=scorer_fn, score_cutoff=threshold)
            if match is None:
                flags.append(False)
            else:
                remaining.pop(match[2])
                flags.append(True)
    else:  # their rule: a held-out title is never consumed
        for q in queries:
            flags.append(bool(q) and any(scorer_fn(q, t) >= threshold for t in test_norm))
    return flags


def exact_hit_flags(
    ranked_item_ids: Sequence[object],
    test_item_ids: Iterable[object],
) -> list[bool]:
    """Hit vector for a classical policy: item id in the held-out set."""
    test = set(test_item_ids)
    return [item in test for item in ranked_item_ids]


def ranking_metrics_from_hits(
    hits: Sequence[bool] | np.ndarray,
    n_relevant: int,
    *,
    ks: Iterable[int] = DEFAULT_KS,
) -> dict[str, float]:
    """HR@K, nDCG@K and Recall@K from a per-rank hit vector.

    ``hits[i]`` is True iff the item at rank i+1 is relevant; ``n_relevant`` is the
    number of held-out items for the user. HR@K is the "at least one hit" indicator
    and nDCG@K uses IDCG over min(n_relevant, K) ideal positions — matching the
    definitions in their evaluate_recommendations.py.
    """
    flags = np.asarray(hits, dtype=bool)
    ks = tuple(dict.fromkeys(int(k) for k in ks))
    out: dict[str, float] = {}
    if n_relevant <= 0:
        for k in ks:
            out[f"hit_rate@{k}"] = out[f"ndcg@{k}"] = out[f"recall@{k}"] = float("nan")
        return out

    discounts = 1.0 / np.log2(np.arange(2, len(flags) + 2)) if len(flags) else np.array([])
    for k in ks:
        top = flags[:k]
        n_hit = int(top.sum())
        out[f"hit_rate@{k}"] = 1.0 if n_hit > 0 else 0.0
        out[f"recall@{k}"] = n_hit / n_relevant
        dcg = float((top * discounts[: len(top)]).sum()) if len(top) else 0.0
        ideal = min(n_relevant, k)
        idcg = float((1.0 / np.log2(np.arange(2, ideal + 2))).sum()) if ideal > 0 else 0.0
        out[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0
    return out


def aggregate_user_metrics(
    per_user: Sequence[dict[str, float]],
    *,
    ks: Iterable[int] = DEFAULT_KS,
) -> dict[str, float | int]:
    """Macro-average per-user metric dicts, ignoring users with no held-out items."""
    ks = tuple(dict.fromkeys(int(k) for k in ks))
    out: dict[str, float | int] = {}
    for key in (f"{name}@{k}" for k in ks for name in ("hit_rate", "ndcg", "recall")):
        vals = np.array([u[key] for u in per_user if key in u], dtype=float)
        vals = vals[~np.isnan(vals)]
        out[f"mean_{key}"] = float(vals.mean()) if len(vals) else float("nan")
    out["n_users"] = int(len(per_user))
    out["n_users_scored"] = int(
        sum(1 for u in per_user if not np.isnan(u.get(f"hit_rate@{ks[0]}", float("nan"))))
    )
    return out
