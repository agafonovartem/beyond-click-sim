from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from beyond_click_sim.scorers.agent4rec.prompts import (
    agent4rec_listwise_user_prompt,
    agent4rec_preference_listwise_user_prompt,
)
from beyond_click_sim.scorers.agent4rec.yes_no import Agent4RecYesNoScorer
from beyond_click_sim.scorers.history.listwise import parse_ranked_labels_response


class Agent4RecListwiseRankingScorer(Agent4RecYesNoScorer):
    """Agent4Rec profile-conditioned direct-ranking scorer."""

    name = "agent4rec_listwise_ranking"

    def _build_user_prompt(
        self,
        *,
        candidates: str,
        taste: str | None,
        candidate_labels: Sequence[str],
    ) -> str:
        return agent4rec_listwise_user_prompt(
            candidates=candidates,
            taste=taste,
            candidate_labels=candidate_labels,
            entity_name=self.entity_name,
            entity_plural=self.entity_plural,
        )

    def _parse_response(
        self,
        text: str,
        *,
        labels: Sequence[str],
    ) -> dict[str, float]:
        ranked_labels = parse_ranked_labels_response(text, labels=labels)
        return {
            label: float(len(ranked_labels) - rank)
            for rank, label in enumerate(ranked_labels, start=1)
        }


class Agent4RecPreferenceListwiseRankingScorer(Agent4RecListwiseRankingScorer):
    """Agent4Rec direct-ranking scorer for a positive-preference target."""

    name = "agent4rec_preference_listwise_ranking"

    def __init__(self, *, target_description: str, **kwargs: Any) -> None:
        if not target_description.strip():
            raise ValueError("target_description must be non-empty")
        super().__init__(**kwargs)
        self.target_description = target_description

    def _build_user_prompt(
        self,
        *,
        candidates: str,
        taste: str | None,
        candidate_labels: Sequence[str],
    ) -> str:
        del taste
        return agent4rec_preference_listwise_user_prompt(
            candidates=candidates,
            target_description=self.target_description,
            candidate_labels=candidate_labels,
            entity_plural=self.entity_plural,
        )
