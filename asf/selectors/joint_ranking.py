import numpy as np
import pandas as pd
from asf.predictors.ranking_mlp import RankingMLP
from sklearn.preprocessing import OneHotEncoder

from asf.selectors.abstract_model_based_selector import AbstractSelector
from asf.selectors.feature_generator import (
    AbstractFeatureGenerator,
)


class JointRanking(AbstractSelector, AbstractFeatureGenerator):
    """
    Joint ranking (Ortuzk et al. 2022)

    Attributes:
        metadata: Metadata containing information about the algorithms.
        use_multi_target: Boolean indicating whether to use multi-target regression.
        normalize: Method to normalize the performance data.
        regressors: List of trained regression models.
    """

    def __init__(
        self,
        metadata,
        model=None,
        hierarchical_generator=None,
    ):
        """
        Initializes the PerformancePredictor with the given parameters.

        Args:
            model_class: The class of the regression model to be used.
            metadata: Metadata containing information about the algorithms.
            use_multi_target: Boolean indicating whether to use multi-target regression.
            normalize: Method to normalize the performance data.
            hierarchical_generator: Feature generator to be used.
        """
        AbstractSelector.__init__(self, metadata, hierarchical_generator)
        AbstractFeatureGenerator.__init__(self)
        self.model = model

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame):
        """
        Fits the regression models to the given features and performance data.

        Args:
            features: DataFrame containing the feature data.
            performance: DataFrame containing the performance data.
        """
        if self.algorithm_features is None:
            encoder = OneHotEncoder(sparse_output=False)
            self.algorithm_features = pd.DataFrame(
                encoder.fit_transform(
                    np.array(self.metadata.algorithms).reshape(-1, 1)
                ),
                index=self.metadata.algorithms,
                columns=[f"algo_{i}" for i in range(len(self.metadata.algorithms))],
            )

        print(features)
        print(performance)
        if self.model is None:
            self.model = RankingMLP(
                input_size=len(self.metadata.features) + len(self.metadata.algorithms)
            )

        self.model.fit(
            features[self.metadata.features], performance, self.algorithm_features
        )

    def _predict(self, features: pd.DataFrame):
        """
        Predicts the performance of algorithms for the given features.

        Args:
            features: DataFrame containing the feature data.

        Returns:
            A dictionary mapping instance names to the predicted best algorithm.
        """
        predictions = self.generate_features(features)

        return {
            instance_name: [
                (
                    self.metadata.algorithms[np.argmin(predictions[i])],
                    self.metadata.budget,
                )
            ]
            for i, instance_name in enumerate(features.index)
        }

    def generate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generates predictions for the given features using the trained models.

        Args:
            features: DataFrame containing the feature data.

        Returns:
            DataFrame containing the predictions for each algorithm.
        """

        predictions = np.zeros((features.shape[0], len(self.metadata.algorithms)))

        features = features[self.metadata.features]
        for i, algorithm in enumerate(self.metadata.algorithms):
            # import pdb; pdb.set_trace()
            data = features.assign(**self.algorithm_features.loc[algorithm])
            data = data[
                self.algorithm_features.columns.to_list() + self.metadata.features
            ]
            prediction = self.model.predict(data)
            predictions[:, i] = prediction.flatten()
            print(predictions)

        return predictions
