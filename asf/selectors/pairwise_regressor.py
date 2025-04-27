import pandas as pd
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from asf.selectors.feature_generator import (
    AbstractFeatureGenerator,
)
from asf.predictors import (
    AbstractPredictor,
    RandomForestRegressorWrapper,
    XGBoostRegressorWrapper,
)
from ConfigSpace import ConfigurationSpace, Categorical, Configuration
from functools import partial
from typing import Optional


class PairwiseRegressor(AbstractModelBasedSelector, AbstractFeatureGenerator):
    PREFIX = "pairwise_regressor"
    """
    PairwiseRegressor is a selector that uses pairwise regression of algorithms
    to predict the best algorithm for a given instance.

    Attributes:
        model_class: The regression model to be used for pairwise comparisons.
        regressors: List of trained regressors for pairwise comparisons.
    """

    def __init__(self, model_class, **kwargs):
        """
        Initializes the PairwiseRegressor with a given model class and hierarchical feature generator.

        Args:
            model_class: The regression model to be used for pairwise comparisons.
            hierarchical_generator (AbstractFeatureGenerator, optional): The feature generator to be used. Defaults to DummyFeatureGenerator.
        """
        AbstractModelBasedSelector.__init__(self, model_class, **kwargs)
        AbstractFeatureGenerator.__init__(self)
        self.regressors = []

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame):
        """
        Fits the pairwise regressors using the provided features and performance data.

        Args:
            features (pd.DataFrame): The feature data for the instances.
            performance (pd.DataFrame): The performance data for the algorithms.
        """
        assert self.algorithm_features is None, (
            "PairwiseRegressor does not use algorithm features."
        )
        for i, algorithm in enumerate(self.algorithms):
            for other_algorithm in self.algorithms[i + 1 :]:
                algo1_times = performance[algorithm]
                algo2_times = performance[other_algorithm]

                diffs = algo1_times - algo2_times
                cur_model = self.model_class()
                cur_model.fit(
                    features,
                    diffs,
                    sample_weight=None,
                )
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
            instance_name: [
                (
                    predictions_sum.loc[instance_name].idmax()
                    if self.maximize
                    else predictions_sum.loc[instance_name].idxmin(),
                    self.budget,
                )
            ]
            for i, instance_name in enumerate(features.index)
        }

    def generate_features(self, features: pd.DataFrame):
        """
        Generates features for the pairwise regressors.

        Args:
            features (pd.DataFrame): The feature data for the instances.

        Returns:
            np.ndarray: An array of predictions for each instance and algorithm pair.
        """

        cnt = 0
        predictions_sum = pd.DataFrame(0, index=features.index, columns=self.algorithms)
        for i, algorithm in enumerate(self.algorithms):
            for j, other_algorithm in enumerate(self.algorithms[i + 1 :]):
                prediction = self.regressors[cnt].predict(features)
                predictions_sum[algorithm] += prediction
                predictions_sum[other_algorithm] -= prediction
                cnt += 1

        return predictions_sum

    @staticmethod
    def get_configuration_space(
        cs: Optional[ConfigurationSpace] = None,
        cs_transform: Optional[dict] = None,
        model_class: list[AbstractPredictor] = [
            RandomForestRegressorWrapper,
            XGBoostRegressorWrapper,
        ],
        hierarchical_generator: list[AbstractFeatureGenerator] | None = None,
        **kwargs,
    ):
        """
        Get the configuration space for the predictor.
        Parameters
        ----------
        cs : ConfigurationSpace, optional
            The configuration space to use. If None, a new one will be created.
        model_class : list, optional
            The list of model classes to use. Defaults to [RandomForestClassifierWrapper, XGBoostClassifierWrapper].
        **kwargs : dict, optional
            Additional keyword arguments to pass to the model class.
        Returns
        -------
        ConfigurationSpace
            The configuration space for the predictor.
        """
        if cs is None:
            cs = ConfigurationSpace()

        cs.add(
            Categorical(
                name=f"{PairwiseRegressor.PREFIX}:model_class",
                items=[str(c.__name__) for c in model_class],
            )
        )
        cs_transform[f"{PairwiseRegressor.PREFIX}:model_class"] = {
            str(c.__name__): c for c in model_class
        }

        PairwiseRegressor._add_hierarchical_generator_space(
            cs=cs,
            hierarchical_generator=hierarchical_generator,
        )

        for model in model_class:
            model.get_configuration_space(cs=cs, **kwargs)

        return cs, cs_transform

    @staticmethod
    def get_from_configuration(configuration: Configuration, cs_transform):
        """
        Get the configuration space for the predictor.

        Returns
        -------
        AbstractPredictor
            The predictor.
        """
        model_class = cs_transform[f"{PairwiseRegressor.PREFIX}:model_class"][
            configuration[f"{PairwiseRegressor.PREFIX}:model_class"]
        ]

        model = model_class.get_from_configuration(configuration, cs_transform)

        return partial(
            PairwiseRegressor,
            model_class=model,
            hierarchical_generator=None,
        )
