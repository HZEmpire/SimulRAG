"""Runtime configuration for SimulRAG."""

from __future__ import annotations

from dataclasses import dataclass

CENTRALITY_METRICS = {
    "closeness": "closeness_centrality",
    "degree": "degree_centrality",
    "pagerank": "pagerank",
    "eigenvalue": "eigenvalue_centrality",
    "betweenness": "betweenness_centrality",
}


@dataclass(slots=True)
class PipelineConfig:
    """Paper defaults, exposed by the command-line interface."""

    provider: str = "openai"
    model: str = "gpt-5-nano"
    num_answers: int = 5
    centrality: str = "closeness"
    selection_threshold: float = 0.4
    filtering_threshold: float = 0.4
    answer_temperature: float = 1.0
    decomposition_temperature: float = 0.1
    boundary_temperature: float = 0.1
    tool_temperature: float = 0.1
    update_temperature: float = 0.1
    final_temperature: float = 0.1
    max_retries: int = 3
    concurrency: int = 8

    @property
    def centrality_key(self) -> str:
        return CENTRALITY_METRICS[self.centrality]

    def validate(self) -> None:
        if self.num_answers < 1:
            raise ValueError("num_answers must be at least 1")
        if self.centrality not in CENTRALITY_METRICS:
            choices = ", ".join(CENTRALITY_METRICS)
            raise ValueError(f"centrality must be one of: {choices}")
        if not 0.0 <= self.selection_threshold <= 1.0:
            raise ValueError("selection_threshold must be in [0, 1]")
        if not 0.0 <= self.filtering_threshold <= 1.0:
            raise ValueError("filtering_threshold must be in [0, 1]")
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if self.concurrency < 1:
            raise ValueError("concurrency must be at least 1")
