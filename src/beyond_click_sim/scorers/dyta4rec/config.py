"""Declared reconstruction settings; these are not recovered author defaults."""

from dataclasses import asdict, dataclass
import math


VERSION = "dyta4rec-offline-v1"
PROMPT_VERSION = "dyta4rec-prompts-v1"
AUTHOR_REVISION = "c4a0cc61c4a539b75cb426380261d6520859e84f"
UPSTREAM_REVISION = "fff935570079502d5a17509b0baf590becd1eaab"
SOURCE_ARTIFACT_SHA256 = {
    "paper_arxiv_2509.16895v1.pdf": "91ecafa990a6984b7e10d3ea47bb85e866b6c1bd259d966672d979e9e04865b8",
    "author-agent.md": "2f267f99f6743dce3a192d9d6097d065959a0e63fdfff26df707aeaa5a90d1d1",
    "upstream-recagent-memory.txt": "319960b7dfead348ea10dd02da863aa5dc035637b1d9219cac20a9f19252e41b",
}


@dataclass(frozen=True)
class DyTA4RecConfig:
    profile_version: str = "dyta_profile_v1"
    memory_mode: str = "yulan_retrieval_two_level_v1"
    history_items: int = 10
    icl_examples: int = 3
    block_size: int = 50
    merge_fan_in: int = 10
    retrieval_k: int = 5
    relevance_k: int = 100
    importance_weight: float = 0.9
    decay_rate: float = 0.01
    fusion: str = "rrf"
    rrf_k: int = 60
    bc_length_policy: str = "candidate_count"
    max_initialization_attempts: int = 5
    summary_max_tokens: int = 1024
    importance_max_tokens: int = 16
    recent_max_tokens: int = 256
    analysis_max_tokens: int = 1024
    ranking_max_tokens: int = 2048
    context_tokens: int = 32768
    metadata_max_chars: int = 512
    text_max_chars: int = 3000
    profile_field_max_chars: int = 600
    seed: int = 0
    temperature: float = 0.0
    top_p: float = 1.0

    def __post_init__(self):
        for key in (
            "history_items",
            "block_size",
            "merge_fan_in",
            "retrieval_k",
            "relevance_k",
            "rrf_k",
            "max_initialization_attempts",
            "summary_max_tokens",
            "importance_max_tokens",
            "recent_max_tokens",
            "analysis_max_tokens",
            "ranking_max_tokens",
            "context_tokens",
            "metadata_max_chars",
            "text_max_chars",
            "profile_field_max_chars",
        ):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.merge_fan_in < 2 or self.max_initialization_attempts > 5:
            raise ValueError("Merge fan-in must be >=2; attempts must be <=5")
        if type(self.icl_examples) is not int or self.icl_examples < 0:
            raise ValueError("icl_examples must be nonnegative")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if self.fusion not in ("rrf", "bc"):
            raise ValueError("fusion must be rrf or bc")
        if (self.profile_version, self.memory_mode, self.bc_length_policy) != (
            "dyta_profile_v1",
            "yulan_retrieval_two_level_v1",
            "candidate_count",
        ):
            raise ValueError("Unsupported reconstruction version or BC policy")
        for key in ("importance_weight", "decay_rate", "temperature", "top_p"):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"Nonfinite {key}")
        if not (
            0 <= self.importance_weight <= 1
            and 0 < self.decay_rate < 1
            and 0 <= self.temperature <= 2
            and 0 < self.top_p <= 1
        ):
            raise ValueError("Invalid retrieval or generation settings")

    def to_dict(self):
        return asdict(self)

    def tokens(self, stage):
        if stage == "importance":
            return self.importance_max_tokens
        if stage.startswith("recent_"):
            return self.recent_max_tokens
        if stage.endswith("analysis"):
            return self.analysis_max_tokens
        if stage.endswith("ranking"):
            return self.ranking_max_tokens
        return self.summary_max_tokens
