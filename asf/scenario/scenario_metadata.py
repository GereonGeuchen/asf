from dataclasses import dataclass


@dataclass
class ScenarioMetadata:
    algorithms: list[str]
    features: list[str]
    algorith_features: list[str] | None
    performance_metric: str | list[str]
    feature_groups: dict[str, dict[str, list[str]]]
    maximize: bool
    budget: int | None

    def to_dict(self):
        """Converts the metadata into a dictionary format."""
        return {
            "algorithms": self.algorithms,
            "features": self.features,
            "performance_metric": self.performance_metric,
            "feature_groups": self.feature_groups,
            "maximize": self.maximize,
            "budget": self.budget,
        }
