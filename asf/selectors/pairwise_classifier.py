import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace, Categorical, Configuration
from asf.predictors import (
    AbstractPredictor,
    RandomForestClassifierWrapper,
    XGBoostClassifierWrapper,
)
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from asf.selectors.feature_generator import (
    AbstractFeatureGenerator,
)
from functools import partial
from typing import Optional, List, Dict, Tuple, Union


class PairwiseClassifier(AbstractModelBasedSelector, AbstractFeatureGenerator):
    PREFIX = "pairwise_classifier"
    """
    PairwiseClassifier is a selector that uses pairwise comparison of algorithms
    to predict the best algorithm for a given instance.

    Attributes:
        PREFIX (str): Prefix used for configuration space parameters.
        classifiers (List[AbstractPredictor]): List of trained classifiers for pairwise comparisons.
        use_weights (bool): Whether to use weights based on performance differences.
    """

    def __init__(
        self, model_class: type[AbstractPredictor], use_weights: bool = True, **kwargs
    ):
        """
        Initializes the PairwiseClassifier with a given model class and hierarchical feature generator.

        Args:
            model_class (type[AbstractPredictor]): The classifier model to be used for pairwise comparisons.
            use_weights (bool): Whether to use weights based on performance differences. Defaults to True.
            **kwargs: Additional keyword arguments for the parent class.
        """
        AbstractModelBasedSelector.__init__(self, model_class, **kwargs)
        AbstractFeatureGenerator.__init__(self)
        self.classifiers: List[AbstractPredictor] = []
        self.use_weights: bool = use_weights

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """
        Fits the pairwise classifiers using the provided features and performance data.

        Args:
            features (pd.DataFrame): The feature data for the instances.
            performance (pd.DataFrame): The performance data for the algorithms.
        """
        assert self.algorithm_features is None, (
            "PairwiseClassifier does not use algorithm features."
        )
        for i, algorithm in enumerate(self.algorithms):
            for other_algorithm in self.algorithms[i + 1 :]:
                algo1_times = performance[algorithm]
                algo2_times = performance[other_algorithm]

                if self.maximize:
                    diffs = algo1_times > algo2_times
                else:
                    diffs = algo1_times < algo2_times

                cur_model = self.model_class()
                cur_model.fit(
                    features,
                    diffs,
                    sample_weight=None
                    if not self.use_weights
                    else np.abs(algo1_times - algo2_times),
                )
                self.classifiers.append(cur_model)

    def _predict(
        self, features: pd.DataFrame
    ) -> Dict[str, List[Tuple[str, Union[int, float]]]]:
        """
        Predicts the best algorithm for each instance using the trained pairwise classifiers.

        Args:
            features (pd.DataFrame): The feature data for the instances.

        Returns:
            Dict[str, List[Tuple[str, Union[int, float]]]]: A dictionary mapping instance names to the predicted best algorithm and budget.
        """
        predictions_sum = self.generate_features(features)

        return {
            instance_name: [
                (
                    predictions_sum.loc[instance_name].idxmax(),
                    self.budget,
                )
            ]
            for i, instance_name in enumerate(features.index)
        }

    def generate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generates features for the pairwise classifiers.

        Args:
            features (pd.DataFrame): The feature data for the instances.

        Returns:
            pd.DataFrame: A DataFrame of predictions for each instance and algorithm pair.
        """
        cnt = 0
        predictions_sum = pd.DataFrame(0, index=features.index, columns=self.algorithms)
        for i, algorithm in enumerate(self.algorithms):
            for j, other_algorithm in enumerate(self.algorithms[i + 1 :]):
                prediction = self.classifiers[cnt].predict(features)
                predictions_sum.loc[prediction, algorithm] += 1
                predictions_sum.loc[~prediction, other_algorithm] += 1
                cnt += 1

        return predictions_sum

    @staticmethod
    def get_configuration_space(
        cs: Optional[ConfigurationSpace] = None,
        cs_transform: Optional[Dict[str, dict]] = None,
        model_class: List[type[AbstractPredictor]] = [
            RandomForestClassifierWrapper,
            XGBoostClassifierWrapper,
        ],
        hierarchical_generator: Optional[List[AbstractFeatureGenerator]] = None,
        **kwargs,
    ) -> Tuple[ConfigurationSpace, Dict[str, dict]]:
        """
        Get the configuration space for the predictor.

        Args:
            cs (Optional[ConfigurationSpace]): The configuration space to use. If None, a new one will be created.
            cs_transform (Optional[Dict[str, dict]]): A dictionary for transforming configuration space parameters.
            model_class (List[type[AbstractPredictor]]): The list of model classes to use. Defaults to [RandomForestClassifierWrapper, XGBoostClassifierWrapper].
            hierarchical_generator (Optional[List[AbstractFeatureGenerator]]): List of hierarchical feature generators.
            **kwargs: Additional keyword arguments to pass to the model class.

        Returns:
            Tuple[ConfigurationSpace, Dict[str, dict]]: The configuration space and its transformation dictionary.
        """
        if cs is None:
            cs = ConfigurationSpace()

        if cs_transform is None:
            cs_transform = dict()

        if f"{PairwiseClassifier.PREFIX}:model_class" not in cs:
            cs.add(
                Categorical(
                    name=f"{PairwiseClassifier.PREFIX}:model_class",
                    items=[str(c.__name__) for c in model_class],
                )
            )
            cs_transform[f"{PairwiseClassifier.PREFIX}:model_class"] = {
                str(c.__name__): c for c in model_class
            }

        if f"{PairwiseClassifier.PREFIX}:use_weights" not in cs:
            cs.add(
                Categorical(
                    name=f"{PairwiseClassifier.PREFIX}:use_weights",
                    items=[True, False],
                )
            )

        PairwiseClassifier._add_hierarchical_generator_space(
            cs=cs,
            hierarchical_generator=hierarchical_generator,
        )

        for model in model_class:
            model.get_configuration_space(cs=cs, **kwargs)

        return cs, cs_transform

    @staticmethod
    def get_from_configuration(
        configuration: Configuration, cs_transform: Dict[str, dict]
    ) -> partial:
        """
        Get the predictor from a given configuration.

        Args:
            configuration (Configuration): The configuration object.
            cs_transform (Dict[str, dict]): The transformation dictionary for the configuration space.

        Returns:
            partial: A partial function to initialize the PairwiseClassifier with the given configuration.
        """
        model_class = cs_transform[f"{PairwiseClassifier.PREFIX}:model_class"][
            configuration[f"{PairwiseClassifier.PREFIX}:model_class"]
        ]
        use_weights = configuration[f"{PairwiseClassifier.PREFIX}:use_weights"]

        model = model_class.get_from_configuration(configuration, cs_transform)

        return partial(
            PairwiseClassifier,
            model_class=model,
            use_weights=use_weights,
            hierarchical_generator=None,
        )
