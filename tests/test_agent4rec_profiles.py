from __future__ import annotations

import pandas as pd

from beyond_click_sim.scorers.agent4rec import Agent4RecProfileGenerator


def test_diversity_counts_agent4rec_top_genre_mass_per_user() -> None:
    X = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u2", "u3", "u3"],
            "item_id": ["i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8"],
            "rating": [5, 4, 3, 2, 5, 4, 5, 4],
            "item_genres": [
                "Action|Drama",
                "Action",
                "Action|Comedy",
                "Horror",
                "Horror",
                "Thriller",
                "Sci-Fi",
                "Sci-Fi",
            ],
        }
    )

    diversity_num = Agent4RecProfileGenerator()._agent4rec_diversity_count_by_user(X)

    assert diversity_num.to_dict() == {"u1": 2, "u2": 1, "u3": 1}
