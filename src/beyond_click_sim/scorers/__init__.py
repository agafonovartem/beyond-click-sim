"""Scoring contracts and scorer implementations."""

from beyond_click_sim.scorers.base import Scorer
from beyond_click_sim.scorers.constant import (
    ItemMeanRegressor,
    ItemModeRegressor,
    MeanRegressor,
    ModeRegressor,
    UserMeanRegressor,
    UserModeRegressor,
)
from beyond_click_sim.scorers.agent4rec import (
    Agent4RecProfileGenerator,
    Agent4RecRegressor,
    Agent4RecTasteProfile,
    Agent4RecUserProfile,
    Agent4RecYesNoScorer,
)
from beyond_click_sim.scorers.history import (
    LLMInteractionYesNoScorer,
    LLMPreferenceYesNoScorer,
    LLMRegressor,
)
from beyond_click_sim.scorers.item_knn import ColdItemKNNScorer
from beyond_click_sim.scorers.popularity import PopularityScorer

__all__ = [
    "ColdItemKNNScorer",
    "ItemMeanRegressor",
    "ItemModeRegressor",
    "Agent4RecProfileGenerator",
    "Agent4RecRegressor",
    "Agent4RecTasteProfile",
    "Agent4RecUserProfile",
    "Agent4RecYesNoScorer",
    "LLMInteractionYesNoScorer",
    "LLMPreferenceYesNoScorer",
    "LLMRegressor",
    "MeanRegressor",
    "ModeRegressor",
    "PopularityScorer",
    "Scorer",
    "UserMeanRegressor",
    "UserModeRegressor",
]
