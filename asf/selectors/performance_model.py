import numpy as np
import pandas as pd
import inspect

from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from asf.selectors.feature_generator import (
    AbstractFeatureGenerator,
)


class PerformanceModel(AbstractModelBasedSelector, AbstractFeatureGenerator):
    """
    PerformancePredictor is a class that predicts the performance of algorithms
    based on given features. It can handle both single-target and multi-target
    regression models.

    Attributes:
        model_class: The class of the regression model to be used.
        metadata: Metadata containing information about the algorithms.
        use_multi_target: Boolean indicating whether to use multi-target regression.
        normalize: Method to normalize the performance data.
        regressors: List of trained regression models.
    """

    def __init__(
        self,
        model_class,
        metadata,
        use_multi_target=False,
        normalize="log",
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
        AbstractModelBasedSelector.__init__(
            self, model_class, metadata, hierarchical_generator
        )
        AbstractFeatureGenerator.__init__(self)
        self.regressors = []
        self.use_multi_target = use_multi_target
        self.normalize = normalize

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame):
        """
        Fits the regression models to the given features and performance data.

        Args:
            features: DataFrame containing the feature data.
            performance: DataFrame containing the performance data.
        """
        assert self.algorithm_features is None, (
            "PerformanceModel does not use algorithm features."
        )
        if self.normalize == "log":
            performance = np.log10(performance + 1e-6)

        regressor_init_args = {}
        if "input_size" in inspect.signature(self.model_class).parameters.keys():
            regressor_init_args["input_size"] = features.shape[1]

        if self.use_multi_target:
            assert self.algorithm_features is None, (
                "PerformanceModel does not use algorithm features for multi-target regression."
            )
            self.regressors = self.model_class(**regressor_init_args)
            self.regressors.fit(features, performance)
        else:
            if self.algorithm_features is None:
                for i, algorithm in enumerate(self.metadata.algorithms):
                    algo_times = performance.iloc[:, i]

                    cur_model = self.model_class(**regressor_init_args)
                    cur_model.fit(features, algo_times)
                    self.regressors.append(cur_model)
            else:
                train_data = []
                for i, algorithm in enumerate(self.metadata.algorithms):
                    data = pd.merge(
                        features,
                        self.algorithm_features.loc[algorithm],
                        left_index=True,
                        right_index=True,
                    )
                    data = pd.merge(
                        data, performance.iloc[:, i], left_index=True, right_index=True
                    )
                    train_data.append(data)
                train_data = pd.concat(train_data)
                self.regressors = self.model_class(**regressor_init_args)
                self.regressors.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])

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
        if self.use_multi_target:
            predictions = self.regressors.predict(features)
        else:
            if self.algorithm_features is None:
                predictions = np.zeros(
                    (features.shape[0], len(self.metadata.algorithms))
                )
                for i, algorithm in enumerate(self.metadata.algorithms):
                    prediction = self.regressors[i].predict(features)
                    predictions[:, i] = prediction
            else:
                predictions = np.zeros(
                    (features.shape[0], len(self.metadata.algorithms))
                )
                for i, algorithm in enumerate(self.metadata.algorithms):
                    data = pd.merge(
                        features,
                        self.algorithm_features.loc[algorithm],
                        left_index=True,
                        right_index=True,
                    )
                    prediction = self.regressors.predict(data)
                    predictions[:, i] = prediction

        return predictions
