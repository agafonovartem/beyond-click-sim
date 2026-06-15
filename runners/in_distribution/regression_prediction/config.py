from __future__ import annotations


MAX_HISTORY_ITEMS = 20

DATASET_TARGET_REGRESSION_CONFIG = {
    "ml-1m": {
        "target_rating": {
            "history_description_columns": ("item_title", "item_genres", "rating"),
            "candidate_description_columns": ("item_title", "item_genres"),
            "history_value_column": "rating",
            "target_name": "rating",
            "target_description": (
                "Predict the integer rating this user would give to "
                "this movie on a 1 to 5 scale."
            ),
            "output_instructions": (
                "Return exactly one integer: 1, 2, 3, 4, or 5. "
                "Return no other text."
            ),
            "valid_values": (1, 2, 3, 4, 5),
            "output_format": "bare_integer",
        },
    },
}
