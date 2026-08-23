"""Policy ranking agreement runner with a positive-preference target.

Same protocol as `policy_ranking_agreement`, but the real held-out outcome is a
dataset-specific positive-preference label (`target_like_ge4` / `target_played_120`)
instead of the raw interaction indicator (`target_interact`), and the simulators
are asked the matching preference question rather than an interaction question.
"""
