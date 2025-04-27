import pandas as pd
import numpy as np
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector


class MultiClassClassifier(AbstractModelBasedSelector):
    def __init__(self, model_class, **kwargs):
        AbstractModelBasedSelector.__init__(self, model_class, **kwargs)
        self.classifier = None

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame):
        """
        Fits the classification model to the given feature and performance data.

        Args:
            features: DataFrame containing the feature data.
            performance: DataFrame containing the performance data.
        """
        assert self.algorithm_features is None, (
            "MultiClassClassifier does not use algorithm features."
        )
        self.classifier = self.model_class()
        self.classifier.fit(features, np.argmin(performance.values, axis=1))

    def _predict(self, features: pd.DataFrame):
        """
        Predicts the best algorithm for each instance in the given feature data using simple multi class classification.

        Args:
            features: DataFrame containing the feature data.

        Returns:
            A dictionary mapping instance names to the predicted best algorithm.
        """
        predictions = self.classifier.predict(features)

        return {
            instance_name: [(self.algorithms[predictions[i]], self.budget)]
            for i, instance_name in enumerate(features.index)
        }
