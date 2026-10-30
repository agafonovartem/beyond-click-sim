"""Versioned operational reconstruction of text-only SimUSER(zero)."""

from dataclasses import asdict, dataclass

VERSION = "simuser_zero_text_only_v1"
GRAPH_VERSION = "simuser_graph_v1"
PROMPT_VERSION = "simuser_interaction_v1"
EMBEDDING_TEXT_VERSION = "simuser_embedding_text_v1"
AGES = (16, 18, 25, 35, 45, 50, 56, 65)
OCCUPATIONS = (
    "other/unspecified",
    "academic/educator",
    "artist",
    "clerical/admin",
    "college/grad student",
    "customer service",
    "doctor/health care",
    "executive/managerial",
    "farmer",
    "homemaker",
    "K-12 student",
    "lawyer",
    "programmer",
    "retired",
    "sales/marketing",
    "scientist",
    "self-employed",
    "technician/engineer",
    "tradesman/craftsman",
    "unemployed",
    "writer",
)
TRAITS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)
TOKENS = {
    "taste": 1024,
    "personas": 2048,
    "ratings": 1024,
    "self_ask": 512,
    "watch": 2048,
    "revise": 2048,
    "check": 2048,
}
ADAPTATIONS = (
    "text-only architectural ablation; no thumbnails or visual perception",
    "history-initialized zero; no simulated training sessions",
    "independent interaction-occurrence groups; no ratings/navigation/reflection/state updates",
    "shared per-user temporal split, sampled unobserved negatives, multi-positive supplied groups",
    "binary watch/skip also evaluated as tied ranking scores; no threshold",
    "reconstructed persona menus, j/rho, itemwise contrast ratings and stable ties",
    "simuser_graph_v1: genre/interaction paths, history projection, normalized cosine fusion; no actor paths",
    "training-only graph/statistics; no global historical cutoff",
    "frozen external summaries and canonical genres; training-derived quality/popularity",
    "rating 4 liked in summary/graph but neutral in initial episodic text",
    "two self-ask follow-ups, per-query merge and bounded refinement, strict JSON and caches",
    "declared backbone for all cognitive stages; Qwen substitutes for source GPT-4o-mini",
    "temperature zero and versioned token/field limits are operational choices",
    "method-specific history, metadata and call budgets; not an architecture-only comparison",
)


@dataclass(frozen=True)
class SimUSERConfig:
    seed: int = 2023
    history_sample: int = 50
    contrast_subsets: int = 3
    contrast_size: int = 10
    episodic_k: int = 5
    graph_k: int = 3
    delta_k: int = 2
    max_refinements: int = 2
    initialization_attempts: int = 5
    temperature: float = 0.0
    provider_seed: bool = True
    context_tokens: int | None = None
    title_chars: int = 200
    summary_chars: int = 600
    reason_chars: int = 400

    def __post_init__(self):
        for name in (
            "history_sample",
            "contrast_subsets",
            "contrast_size",
            "episodic_k",
            "graph_k",
            "delta_k",
            "initialization_attempts",
            "title_chars",
            "summary_chars",
            "reason_chars",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.seed) is not int
            or type(self.max_refinements) is not int
            or not 0 <= self.max_refinements <= 2
        ):
            raise ValueError("Integer seed and zero to two refinements required")
        if (
            self.initialization_attempts > 5
            or self.history_sample > 50
            or self.contrast_size > 10
        ):
            raise ValueError("Configuration exceeds declared initialization budgets")
        if self.context_tokens is not None and (
            type(self.context_tokens) is not int or self.context_tokens < 1
        ):
            raise ValueError("context_tokens must be positive or None")
        if self.temperature != 0:
            raise ValueError("This variant uses temperature zero")

    def to_dict(self):
        return {
            **asdict(self),
            "version": VERSION,
            "graph_version": GRAPH_VERSION,
            "prompt_version": PROMPT_VERSION,
            "embedding_text_version": EMBEDDING_TEXT_VERSION,
            "graph_reconstruction": {
                "channels": ["item-user-item", "item-genre-item"],
                "channel_weights": [0.5, 0.5],
                "item_weight": 0.8,
                "user_projection_weight": 0.2,
                "structural_weight": 0.75,
                "semantic_weight": 0.25,
                "semantic_normalization": "(1 + cosine) / 2",
                "user_projection": "mean item similarity over unique observed training items",
            },
            "historical_preference": {
                "summary_graph_liked": ">=4",
                "episodic_liked": ">4",
                "disliked": "<=2",
            },
            "tokens": TOKENS.copy(),
            "persona_count": 5,
            "follow_up_queries": 2,
            "ages": list(AGES),
            "occupations": list(OCCUPATIONS),
        }
