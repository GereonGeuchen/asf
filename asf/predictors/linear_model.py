from ConfigSpace import ConfigurationSpace, Float
from sklearn.linear_model import SGDClassifier, SGDRegressor

from asf.predictors.sklearn_wrapper import SklearnWrapper

from functools import partial
from typing import Optional, Dict, Any


class LinearClassifierWrapper(SklearnWrapper):
    """
    A wrapper for the SGDClassifier from scikit-learn, providing additional functionality
    for configuration space generation and parameter extraction.
    """

    PREFIX = "linear_classifier"

    def __init__(self, init_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the LinearClassifierWrapper.

        Parameters
        ----------
        init_params : dict, optional
            A dictionary of initialization parameters for the SGDClassifier.
        """
        super().__init__(SGDClassifier, init_params or {})

    @staticmethod
    def get_configuration_space(
        cs: Optional[ConfigurationSpace] = None,
    ) -> ConfigurationSpace:
        """
        Get the configuration space for the Linear Classifier.

        Parameters
        ----------
        cs : ConfigurationSpace, optional
            The configuration space to add the parameters to. If None, a new ConfigurationSpace will be created.

        Returns
        -------
        ConfigurationSpace
            The configuration space with the Linear Classifier parameters.
        """
        if cs is None:
            cs = ConfigurationSpace(name="Linear Classifier")

        alpha = Float(
            f"{LinearClassifierWrapper.PREFIX}:alpha", (1e-5, 1), log=True, default=1e-3
        )
        eta0 = Float(
            f"{LinearClassifierWrapper.PREFIX}:eta0", (1e-5, 1), log=True, default=1e-2
        )
        cs.add([alpha, eta0])

        return cs

    @staticmethod
    def get_from_configuration(
        configuration: Dict[str, Any],
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> partial:
        """
        Create a partial function to initialize LinearClassifierWrapper with parameters from a configuration.

        Parameters
        ----------
        configuration : dict
            A dictionary containing the configuration parameters.
        additional_params : dict, optional
            Additional parameters to include in the initialization.

        Returns
        -------
        partial
            A partial function to initialize LinearClassifierWrapper.
        """
        additional_params = additional_params or {}
        linear_classifier_params = {
            "alpha": configuration[f"{LinearClassifierWrapper.PREFIX}:alpha"],
            "eta0": configuration[f"{LinearClassifierWrapper.PREFIX}:eta0"],
            **additional_params,
        }

        return partial(LinearClassifierWrapper, init_params=linear_classifier_params)


class LinearRegressorWrapper(SklearnWrapper):
    """
    A wrapper for the SGDRegressor from scikit-learn, providing additional functionality
    for configuration space generation and parameter extraction.
    """

    PREFIX = "linear_regressor"

    def __init__(self, init_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the LinearRegressorWrapper.

        Parameters
        ----------
        init_params : dict, optional
            A dictionary of initialization parameters for the SGDRegressor.
        """
        super().__init__(SGDRegressor, init_params or {})

    @staticmethod
    def get_configuration_space(
        cs: Optional[ConfigurationSpace] = None,
    ) -> ConfigurationSpace:
        """
        Get the configuration space for the Linear Regressor.

        Parameters
        ----------
        cs : ConfigurationSpace, optional
            The configuration space to add the parameters to. If None, a new ConfigurationSpace will be created.

        Returns
        -------
        ConfigurationSpace
            The configuration space with the Linear Regressor parameters.
        """
        if cs is None:
            cs = ConfigurationSpace(name="Linear Regressor")

        alpha = Float(
            f"{LinearRegressorWrapper.PREFIX}:alpha", (1e-5, 1), log=True, default=1e-3
        )
        eta0 = Float(
            f"{LinearRegressorWrapper.PREFIX}:eta0", (1e-5, 1), log=True, default=1e-2
        )
        cs.add([alpha, eta0])

        return cs

    @staticmethod
    def get_from_configuration(
        configuration: Dict[str, Any],
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> partial:
        """
        Create a partial function to initialize LinearRegressorWrapper with parameters from a configuration.

        Parameters
        ----------
        configuration : dict
            A dictionary containing the configuration parameters.
        additional_params : dict, optional
            Additional parameters to include in the initialization.

        Returns
        -------
        partial
            A partial function to initialize LinearRegressorWrapper.
        """
        additional_params = additional_params or {}
        linear_regressor_params = {
            "alpha": configuration[f"{LinearRegressorWrapper.PREFIX}:alpha"],
            "eta0": configuration[f"{LinearRegressorWrapper.PREFIX}:eta0"],
            **additional_params,
        }

        return partial(LinearRegressorWrapper, init_params=linear_regressor_params)
