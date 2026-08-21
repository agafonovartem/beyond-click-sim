"""Raw dataset adapters."""

from beyond_click_sim.data.adapters.coat import CoatAdapter
from beyond_click_sim.data.adapters.movielens import MovieLens1MAdapter
from beyond_click_sim.data.adapters.steam import SteamAdapter

__all__ = ["CoatAdapter", "MovieLens1MAdapter", "SteamAdapter"]
