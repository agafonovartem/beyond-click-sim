"""Scoring contracts and scorer implementations."""

from beyond_click_sim.scorers.base import Scorer
from beyond_click_sim.scorers.popularity import PopularityScorer

__all__ = ["PopularityScorer", "Scorer"]
