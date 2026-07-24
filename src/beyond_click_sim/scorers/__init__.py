"""Scoring contracts and scorer implementations."""

from beyond_click_sim.scorers.base import Scorer
from beyond_click_sim.scorers.bpr import BPRScorer
from beyond_click_sim.scorers.constant import (
    ItemMeanRegressor,
    ItemModeRegressor,
    MeanRegressor,
    ModeRegressor,
    UserMeanRegressor,
    UserModeRegressor,
)
from beyond_click_sim.scorers.exposure_logistic import (
    ExposureLogisticBiasScorer,
)
from beyond_click_sim.scorers.agent4rec import (
    Agent4RecListwiseRankingScorer,
    Agent4RecProfileGenerator,
    Agent4RecPreferenceListwiseRankingScorer,
    Agent4RecPreferenceYesNoScorer,
    Agent4RecRegressor,
    Agent4RecTasteProfile,
    Agent4RecUserProfile,
    Agent4RecYesNoScorer,
)
from beyond_click_sim.scorers.history import (
    LLMInteractionListwiseRankingScorer,
    LLMInteractionYesNoScorer,
    LLMPreferenceListwiseRankingScorer,
    LLMPreferenceYesNoScorer,
    LLMRegressor,
    LLMTargetResponseListwiseRankingScorer,
    LLMTargetResponseProbabilityScorer,
)
from beyond_click_sim.scorers.item_knn import ColdItemKNNScorer, ItemKNNScorer
from beyond_click_sim.scorers.popularity import PopularityScorer
from beyond_click_sim.scorers.text_affinity import HistoryTextAffinityScorer
from beyond_click_sim.scorers.user_ctr import SmoothedUserCTRScorer

__all__ = [
    "BPRScorer",
    "ColdItemKNNScorer",
    "ExposureLogisticBiasScorer",
    "ItemKNNScorer",
    "HistoryTextAffinityScorer",
    "SmoothedUserCTRScorer",
    "ItemMeanRegressor",
    "ItemModeRegressor",
    "Agent4RecListwiseRankingScorer",
    "Agent4RecProfileGenerator",
    "Agent4RecPreferenceListwiseRankingScorer",
    "Agent4RecPreferenceYesNoScorer",
    "Agent4RecRegressor",
    "Agent4RecTasteProfile",
    "Agent4RecUserProfile",
    "Agent4RecYesNoScorer",
    "LLMInteractionListwiseRankingScorer",
    "LLMInteractionYesNoScorer",
    "LLMPreferenceListwiseRankingScorer",
    "LLMPreferenceYesNoScorer",
    "LLMRegressor",
    "LLMTargetResponseListwiseRankingScorer",
    "LLMTargetResponseProbabilityScorer",
    "MeanRegressor",
    "ModeRegressor",
    "PopularityScorer",
    "Scorer",
    "UserMeanRegressor",
    "UserModeRegressor",
]
