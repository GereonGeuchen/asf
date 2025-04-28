try:
    from typing import override, Optional, Dict, Any
except ImportError:

    def override(func):
        return func


from ConfigSpace import ConfigurationSpace, Float, Integer
from sklearn.neural_network import MLPClassifier, MLPRegressor

from asf.predictors.sklearn_wrapper import SklearnWrapper

from typing import Optional, Dict, Any

from functools import partial


class MLPClassifierWrapper(SklearnWrapper):
    """
    A wrapper for the MLPClassifier from scikit-learn, providing additional functionality
    for configuration space and parameter handling.
    """

    PREFIX = "mlp_classifier"

    def __init__(self, init_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the MLPClassifierWrapper.

        Parameters
        ----------
        init_params : dict, optional
            Initial parameters for the MLPClassifier.
        """
        super().__init__(MLPClassifier, init_params or {})

    @override
    def fit(
        self, X: Any, Y: Any, sample_weight: Optional[Any] = None, **kwargs: Any
    ) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like
            Training data.
        Y : array-like
            Target values.
        sample_weight : array-like, optional
            Sample weights. Not supported for MLPClassifier.
        kwargs : dict
            Additional arguments for the fit method.
        """
        assert sample_weight is None, (
            "Sample weights are not supported for MLPClassifier"
        )
        self.model_class.fit(X, Y, **kwargs)

    @staticmethod
    def get_configuration_space(
        cs: Optional[ConfigurationSpace] = None,
    ) -> ConfigurationSpace:
        """
        Get the configuration space for the MLP Classifier.

        Parameters
        ----------
        cs : ConfigurationSpace, optional
            The configuration space to add the parameters to. If None, a new ConfigurationSpace will be created.

        Returns
        -------
        ConfigurationSpace
            The configuration space with the MLP Classifier parameters.
        """
        if cs is None:
            cs = ConfigurationSpace(name="MLP Classifier")

        depth = Integer(
            f"{MLPClassifierWrapper.PREFIX}:depth", (1, 3), default=3, log=False
        )

        width = Integer(
            f"{MLPClassifierWrapper.PREFIX}:width", (16, 1024), default=64, log=True
        )

        batch_size = Integer(
            f"{MLPClassifierWrapper.PREFIX}:batch_size",
            (256, 1024),
            default=32,
            log=True,
        )  # MODIFIED from HPOBENCH

        alpha = Float(
            f"{MLPClassifierWrapper.PREFIX}:alpha",
            (10**-8, 1),
            default=10**-3,
            log=True,
        )

        learning_rate_init = Float(
            f"{MLPClassifierWrapper.PREFIX}:learning_rate_init",
            (10**-5, 1),
            default=10**-3,
            log=True,
        )

        cs.add([depth, width, batch_size, alpha, learning_rate_init])

        return cs

    @staticmethod
    def get_from_configuration(
        configuration: ConfigurationSpace,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> partial:
        """
        Create an MLPClassifierWrapper instance from a configuration.

        Parameters
        ----------
        configuration : ConfigurationSpace
            The configuration containing the parameters.
        additional_params : dict, optional
            Additional parameters to override the default configuration.

        Returns
        -------
        partial
            A partial function to create an MLPClassifierWrapper instance.
        """
        additional_params = additional_params or {}
        hidden_layers = [
            configuration[f"{MLPClassifierWrapper.PREFIX}:width"]
        ] * configuration[f"{MLPClassifierWrapper.PREFIX}:depth"]

        if "activation" not in additional_params:
            additional_params["activation"] = "relu"
        if "solver" not in additional_params:
            additional_params["solver"] = "adam"

        mlp_params = {
            "hidden_layer_sizes": hidden_layers,
            "batch_size": configuration[f"{MLPClassifierWrapper.PREFIX}:batch_size"],
            "alpha": configuration[f"{MLPClassifierWrapper.PREFIX}:alpha"],
            "learning_rate_init": configuration[
                f"{MLPClassifierWrapper.PREFIX}:learning_rate_init"
            ],
            **additional_params,
        }

        return partial(MLPClassifierWrapper, init_params=mlp_params)


class MLPRegressorWrapper(SklearnWrapper):
    """
    A wrapper for the MLPRegressor from scikit-learn, providing additional functionality
    for configuration space and parameter handling.
    """

    PREFIX = "mlp_regressor"

    def __init__(self, init_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the MLPRegressorWrapper.

        Parameters
        ----------
        init_params : dict, optional
            Initial parameters for the MLPRegressor.
        """
        super().__init__(MLPRegressor, init_params or {})

    @override
    def fit(
        self, X: Any, Y: Any, sample_weight: Optional[Any] = None, **kwargs: Any
    ) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like
            Training data.
        Y : array-like
            Target values.
        sample_weight : array-like, optional
            Sample weights. Not supported for MLPRegressor.
        kwargs : dict
            Additional arguments for the fit method.
        """
        assert sample_weight is None, (
            "Sample weights are not supported for MLPRegressor"
        )
        self.model_class.fit(X, Y, **kwargs)

    @staticmethod
    def get_configuration_space(
        cs: Optional[ConfigurationSpace] = None,
    ) -> ConfigurationSpace:
        """
        Get the configuration space for the MLP Regressor.

        Parameters
        ----------
        cs : ConfigurationSpace, optional
            The configuration space to add the parameters to. If None, a new ConfigurationSpace will be created.

        Returns
        -------
        ConfigurationSpace
            The configuration space with the MLP Regressor parameters.
        """
        if cs is None:
            cs = ConfigurationSpace(name="MLP Regressor")

        depth = Integer(
            f"{MLPRegressorWrapper.PREFIX}:depth", (1, 3), default=3, log=False
        )

        width = Integer(
            f"{MLPRegressorWrapper.PREFIX}:width", (16, 1024), default=64, log=True
        )

        batch_size = Integer(
            f"{MLPRegressorWrapper.PREFIX}:batch_size",
            (256, 1024),
            default=256,
            log=True,
        )

        alpha = Float(
            f"{MLPRegressorWrapper.PREFIX}:alpha",
            (10**-8, 1),
            default=10**-3,
            log=True,
        )

        learning_rate_init = Float(
            f"{MLPRegressorWrapper.PREFIX}:learning_rate_init",
            (10**-5, 1),
            default=10**-3,
            log=True,
        )

        cs.add([depth, width, batch_size, alpha, learning_rate_init])

        return cs

    @staticmethod
    def get_from_configuration(
        configuration: ConfigurationSpace,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> partial:
        """
        Create an MLPRegressorWrapper instance from a configuration.

        Parameters
        ----------
        configuration : ConfigurationSpace
            The configuration containing the parameters.
        additional_params : dict, optional
            Additional parameters to override the default configuration.

        Returns
        -------
        partial
            A partial function to create an MLPRegressorWrapper instance.
        """
        additional_params = additional_params or {}
        hidden_layers = [
            configuration[f"{MLPRegressorWrapper.PREFIX}:width"]
        ] * configuration[f"{MLPRegressorWrapper.PREFIX}:depth"]

        if "activation" not in additional_params:
            additional_params["activation"] = "relu"
        if "solver" not in additional_params:
            additional_params["solver"] = "adam"

        mlp_params = {
            "hidden_layer_sizes": hidden_layers,
            "batch_size": configuration[f"{MLPRegressorWrapper.PREFIX}:batch_size"],
            "alpha": configuration[f"{MLPRegressorWrapper.PREFIX}:alpha"],
            "learning_rate_init": configuration[
                f"{MLPRegressorWrapper.PREFIX}:learning_rate_init"
            ],
            **additional_params,
        }

        return partial(MLPRegressorWrapper, init_params=mlp_params)
