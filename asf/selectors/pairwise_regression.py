import numpy as np
import pandas as pd
from asf.selectors.abstract_selector import AbstractSelector
from asf.selectors.feature_generator import (
    AbstractFeatureGenerator,
    DummyFeatureGenerator,
)


class PairwiseRegressor(AbstractSelector, AbstractFeatureGenerator):
    """
    PairwiseRegressor is a selector that uses pairwise regression of algorithms
    to predict the best algorithm for a given instance.

    Attributes:
        model_class: The regression model to be used for pairwise comparisons.
        regressors: List of trained regressors for pairwise comparisons.
    """

    def __init__(self, model_class, hierarchical_generator=DummyFeatureGenerator()):
        """
        Initializes the PairwiseRegressor with a given model class and hierarchical feature generator.

        Args:
            model_class: The regression model to be used for pairwise comparisons.
            hierarchical_generator (AbstractFeatureGenerator, optional): The feature generator to be used. Defaults to DummyFeatureGenerator.
        """
        super().__init__(hierarchical_generator)
        self.model_class = model_class
        self.regressors = []

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame):
        """
        Fits the pairwise regressors using the provided features and performance data.

        Args:
            features (pd.DataFrame): The feature data for the instances.
            performance (pd.DataFrame): The performance data for the algorithms.
        """
        for i, algorithm in enumerate(self.metadata.algorithms):
            for other_algorithm in self.metadata.algorithms[i + 1 :]:
                algo1_times = performance[algorithm]
                algo2_times = performance[other_algorithm]

                diffs = algo1_times - algo2_times
                cur_model = self.model_class()
                cur_model.fit(features, diffs)
                self.regressors.append(cur_model)

    def _predict(self, features: pd.DataFrame):
        """
        Predicts the best algorithm for each instance using the trained pairwise regressors.

        Args:
            features (pd.DataFrame): The feature data for the instances.

        Returns:
            dict: A dictionary mapping instance names to the predicted best algorithm.
        """
        predictions_sum = self.generate_features(features)
        return {
            instance_name: self.metadata.algorithms[np.argmin(predictions_sum[i])]
            for i, instance_name in enumerate(features)
        }

    def generate_features(self, features: pd.DataFrame):
        """
        Generates features for the pairwise regressors.

        Args:
            features (pd.DataFrame): The feature data for the instances.

        Returns:
            np.ndarray: An array of predictions for each instance and algorithm pair.
        """
        predictions_sum = np.zeros((features.shape[0], len(self.metadata.algorithms)))
        for i, algorithm in enumerate(self.metadata.algorithms):
            for other_algorithm in self.metadata.algorithms[i + 1 :]:
                prediction = self.regressors[i].predict(features)

                predictions_sum[:, algorithm] += prediction
                predictions_sum[:, other_algorithm] -= prediction

        return predictions_sum
