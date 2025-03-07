from dataclasses import dataclass


@dataclass
class SelectionScenarioMetadata:
    algorithms: list[str]
    features: list[str]
    performance_metric: str | list[str]
    feature_groups: dict[str, dict[str, list[str]]]
    maximize: bool
    budget: int | None
    algorithm_features: list[str] | None = None

    def to_dict(self):
        """Converts the metadata into a dictionary format."""
        ret_dict = {
            "algorithms": self.algorithms,
            "features": self.features,
            "performance_metric": self.performance_metric,
            "feature_groups": self.feature_groups,
            "maximize": self.maximize,
            "budget": self.budget,
        }

        if self.algorithm_features is not None:
            ret_dict["algorithm_features"] = self.algorithm_features

        return ret_dict


@dataclass
class PerformancePredictionScenarioMetadata:
    targets: list[str]
    features: list[str]

    def to_dict(self):
        """Converts the metadata into a dictionary format."""
        return {
            "targets": self.targets,
            "features": self.features,
        }
