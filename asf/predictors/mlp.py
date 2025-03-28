try:
    from typing import override
except ImportError:

    def override(func):
        return func


from ConfigSpace import ConfigurationSpace, Float, Integer
from sklearn.neural_network import MLPClassifier, MLPRegressor

from asf.predictors.sklearn_wrapper import SklearnWrapper

from functools import partial


class MLPClassifierWrapper(SklearnWrapper):
    PREFIX = "mlp_classifier"

    def __init__(self, init_params: dict = {}):
        super().__init__(MLPClassifier, init_params)

    @override
    def fit(self, X, Y, sample_weight=None, **kwargs):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like
            Training data.
        Y : array-like
            Target values.
        """
        assert sample_weight is None, (
            "Sample weights are not supported for MLPClassifier"
        )
        self.model_class.fit(X, Y, **kwargs)

    def get_configuration_space():
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
    def get_from_configuration(configuration, additional_params={}):
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

        return partial(MLPClassifierWrapper, init_params=mlp_params)


class MLPRegressorWrapper(SklearnWrapper):
    PREFIX = "mlp_regressor"

    def __init__(self, init_params: dict = {}):
        super().__init__(MLPRegressor, init_params)

    @override
    def fit(self, X, Y, sample_weight=None, **kwargs):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like
            Training data.
        Y : array-like
            Target values.
        """
        assert sample_weight is None, (
            "Sample weights are not supported for MLPRegressor"
        )
        self.model_class.fit(X, Y, **kwargs)

    @staticmethod
    def get_configuration_space():
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
    def get_from_configuration(configuration, additional_params={}):
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
