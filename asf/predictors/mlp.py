from ConfigSpace import ConfigurationSpace, Float, Integer
from sklearn.neural_network import MLPClassifier, MLPRegressor

from asf.predictors.sklearn_wrapper import SklearnWrapper


class MLPClassifierWrapper(SklearnWrapper):
    PREFIX = "mlp_classifier"

    def __init__(self, init_params: dict = {}):
        super().__init__(MLPClassifier, init_params)

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

        mlp_params = {
            "hidden_layers": hidden_layers,
            "depth": configuration[f"{MLPRegressorWrapper.PREFIX}:depth"],
            "batch_size": configuration[f"{MLPRegressorWrapper.PREFIX}:batch_size"],
            "alpha": configuration[f"{MLPRegressorWrapper.PREFIX}:alpha"],
            "learning_rate_init": configuration[
                f"{MLPRegressorWrapper.PREFIX}:learning_rate_init"
            ],
            **additional_params,
        }

        return MLPRegressorWrapper(init_params=mlp_params)


class MLPRegressorWrapper(SklearnWrapper):
    PREFIX = "mlp_regressor"

    def __init__(self, init_params: dict = {}):
        super().__init__(MLPRegressor, init_params)

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
            f"{MLPRegressorWrapper.PREFIX}:batch_size", (4, 256), default=32, log=True
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

        mlp_params = {
            "hidden_layers": hidden_layers,
            "depth": configuration[f"{MLPRegressorWrapper.PREFIX}:depth"],
            "batch_size": configuration[f"{MLPRegressorWrapper.PREFIX}:batch_size"],
            "alpha": configuration[f"{MLPRegressorWrapper.PREFIX}:alpha"],
            "learning_rate_init": configuration[
                f"{MLPRegressorWrapper.PREFIX}:learning_rate_init"
            ],
            **additional_params,
        }

        return MLPRegressorWrapper(init_params=mlp_params)
