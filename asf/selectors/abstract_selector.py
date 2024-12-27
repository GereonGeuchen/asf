import pandas as pd
from asf.scenario.scenario_metadata import ScenarioMetadata
from asf.selectors.feature_generator import (
    AbstractFeatureGenerator,
    DummyFeatureGenerator,
)


class AbstractSelector:
    def __init__(
        self,
        metadata: ScenarioMetadata,
        hierarchical_generator: AbstractFeatureGenerator = DummyFeatureGenerator(),
    ):
        self.metadata = metadata
        self.hierarchical_generator = hierarchical_generator

    def fit(self, features: pd.DataFrame, performance: pd.DataFrame):
        features = pd.concat(
            [features, self.hierarchical_generator.generate_features(features)], axis=1
        )
        self._fit(features, performance)

    def predict(self, features: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
        features = pd.concat(
            [features, self.hierarchical_generator.generate_features(features)], axis=1
        )
        return self._predict(features)
