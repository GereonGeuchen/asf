from dataclasses import dataclass


@dataclass
class ScenarioMetadata:
    algorithms: list[str]
    features: list[str]
    performance_metric: str | list[str]
    maximize: bool
    budget: int | None
