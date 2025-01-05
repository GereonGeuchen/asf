import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector


class SimpleRanking(AbstractModelBasedSelector):
    """
    Algorithm Selection via Ranking (Oentaryo et al.) + algo features (optional).
    Attributes:
        model_class: The class of the classification model to be used.
        metadata: Metadata containing information about the algorithms.
        classifier: The trained classification model.
    """

    def __init__(self, model_class, metadata, hierarchical_generator=None):
        """
        Initializes the MultiClassClassifier with the given parameters.

        Args:
            model_class: The class of the classification model to be used. Assumes XGBoost API.
            metadata: Metadata containing information about the algorithms.
            hierarchical_generator: Feature generator to be used.
        """
        AbstractModelBasedSelector.__init__(
            self, model_class, metadata, hierarchical_generator
        )
        self.classifier = None

    def _fit(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        algorithm_features: pd.DataFrame = None,
    ):
        """
        Fits the classification model to the given feature and performance data.

        Args:
            features: DataFrame containing the feature data.
            performance: DataFrame containing the performance data.
        """
        if algorithm_features is None:
            encoder = OneHotEncoder()
            self.algorithm_features = pd.DataFrame(
                encoder.fit_transform(self.metadata.algorithms),
                index=self.metadata.algorithms,
            )

        total_features = pd.merge(features, self.algorithm_features, how="cross")
        self.classifier = self.model_class()
        self.classifier.fit(total_features, np.argmin(performance.values, axis=1))

    def _predict(self, features: pd.DataFrame):
        """
        Predicts the best algorithm for each instance in the given feature data.

        Args:
            features: DataFrame containing the feature data.

        Returns:
            A dictionary mapping instance names to the predicted best algorithm.
        """
        predictions = self.classifier.predict(features)

        return {
            instance_name: self.metadata.algorithms[predictions[i]]
            for i, instance_name in enumerate(features.index)
        }
