import pandas as pd
from asf.scenario.scenario_metadata import SelectionScenarioMetadata
from asf.selectors.feature_generator import (
    AbstractFeatureGenerator,
)


class AbstractSelector:
    def __init__(
        self,
        metadata: SelectionScenarioMetadata,
        hierarchical_generator: AbstractFeatureGenerator = None,
    ):
        self.metadata = metadata
        self.hierarchical_generator = hierarchical_generator
        self.algorithm_features = None

    def fit(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        algorithm_features: pd.DataFrame = None,
    ):
        if self.hierarchical_generator is not None:
            features = pd.concat(
                [features, self.hierarchical_generator.generate_features(features)],
                axis=1,
            )
        self.algorithm_features = algorithm_features
        self._fit(features, performance)

    def predict(self, features: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
        if self.hierarchical_generator is not None:
            features = pd.concat(
                [features, self.hierarchical_generator.generate_features(features)],
                axis=1,
            )
        return self._predict(features)

    def save(self, path):
        pass

    def load(self, path):
        pass
