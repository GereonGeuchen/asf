from ConfigSpace import ConfigurationSpace, Float
from sklearn.linear_model import SGDClassifier, SGDRegressor

from asf.predictors.sklearn_wrapper import SklearnWrapper

from functools import partial


class LinearClassifierWrapper(SklearnWrapper):
    PREFIX = "linear_classifier"

    def __init__(self, init_params: dict = {}):
        super().__init__(SGDClassifier, init_params)

    def get_configuration_space():
        cs = ConfigurationSpace(name="Linear Classifier")
        # HPOBENCH
        alpha = Float(
            f"{LinearClassifierWrapper.PREFIX}:alpha", (1e-5, 1), log=True, default=1e-3
        )

        eta0 = Float(
            f"{LinearClassifierWrapper.PREFIX}:eta0", (1e-5, 1), log=True, default=1e-2
        )
        cs.add([alpha, eta0])

        return cs

    @staticmethod
    def get_from_configuration(configuration, additional_params={}):
        linear_classifier_params = {
            "alpha": configuration[f"{LinearClassifierWrapper.PREFIX}:alpha"],
            "eta0": configuration[f"{LinearClassifierWrapper.PREFIX}:eta0"],
            **additional_params,
        }

        return partial(LinearClassifierWrapper, init_params=linear_classifier_params)


class LinearRegressorWrapper(SklearnWrapper):
    PREFIX = "linear_regressor"

    def __init__(self, init_params: dict = {}):
        super().__init__(SGDRegressor, init_params)

    @staticmethod
    def get_configuration_space():
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
    def get_from_configuration(configuration, additional_params={}):
        linear_regressor_params = {
            "alpha": configuration[f"{LinearRegressorWrapper.PREFIX}:alpha"],
            "eta0": configuration[f"{LinearRegressorWrapper.PREFIX}:eta0"],
            **additional_params,
        }

        return partial(LinearRegressorWrapper, init_params=linear_regressor_params)
