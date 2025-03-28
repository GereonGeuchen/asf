from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from sklearn.svm import SVC, SVR
from functools import partial

from asf.predictors.sklearn_wrapper import SklearnWrapper


class SVMClassifierWrapper(SklearnWrapper):
    PREFIX = "svm_classifier"

    def __init__(self, init_params: dict = {}):
        super().__init__(SVC, init_params)

    def get_configuration_space():
        # NB301
        cs = ConfigurationSpace(name="SVM")

        kernel = Categorical(
            f"{SVMClassifierWrapper.PREFIX}:kernel",
            items=["linear", "rbf", "poly", "sigmoid"],
            default="rbf",
        )
        degree = Integer(
            f"{SVMClassifierWrapper.PREFIX}:degree", (1, 128), log=True, default=1
        )
        coef0 = Float(
            f"{SVMClassifierWrapper.PREFIX}:coef0",
            (-0.5, 0.5),
            log=False,
            default=0.49070634552851977,
        )
        tol = Float(
            f"{SVMClassifierWrapper.PREFIX}:tol",
            (1e-4, 1e-2),
            log=True,
            default=0.0002154969698207585,
        )
        gamma = Categorical(
            f"{SVMClassifierWrapper.PREFIX}:gamma",
            items=["scale", "auto"],
            default="scale",
        )
        C = Float(
            f"{SVMClassifierWrapper.PREFIX}:C",
            (1.0, 20),
            log=True,
            default=3.2333262862494365,
        )
        epsilon = Float(
            f"{SVMClassifierWrapper.PREFIX}:epsilon",
            (0.01, 0.99),
            log=True,
            default=0.14834562300010581,
        )
        shrinking = Categorical(
            f"{SVMClassifierWrapper.PREFIX}:shrinking",
            items=[True, False],
            default=True,
        )

        cs.add([kernel, degree, coef0, tol, gamma, C, epsilon, shrinking])

        return cs

    @staticmethod
    def get_from_configuration(configuration, additional_params={}):
        svm_params = {
            "kernel": configuration[f"{SVMClassifierWrapper.PREFIX}:kernel"],
            "degree": configuration[f"{SVMClassifierWrapper.PREFIX}:degree"],
            "coef0": configuration[f"{SVMClassifierWrapper.PREFIX}:coef0"],
            "tol": configuration[f"{SVMClassifierWrapper.PREFIX}:tol"],
            "gamma": configuration[f"{SVMClassifierWrapper.PREFIX}:gamma"],
            "C": configuration[f"{SVMClassifierWrapper.PREFIX}:C"],
            "epsilon": configuration[f"{SVMClassifierWrapper.PREFIX}:epsilon"],
            "shrinking": configuration[f"{SVMClassifierWrapper.PREFIX}:shrinking"],
            **additional_params,
        }

        return partial(SVMClassifierWrapper, init_params=svm_params)


class SVMRegressorWrapper(SklearnWrapper):
    PREFIX = "svm_regressor"

    def __init__(self, init_params: dict = {}):
        super().__init__(SVR, init_params)

    @staticmethod
    def get_configuration_space():
        cs = ConfigurationSpace(name="SVM Regressor")

        kernel = Categorical(
            f"{SVMRegressorWrapper.PREFIX}:kernel",
            items=["linear", "rbf", "poly", "sigmoid"],
            default="rbf",
        )
        degree = Integer(
            f"{SVMRegressorWrapper.PREFIX}:degree", (1, 128), log=True, default=1
        )
        coef0 = Float(
            f"{SVMRegressorWrapper.PREFIX}:coef0",
            (-0.5, 0.5),
            log=False,
            default=0.0,
        )
        tol = Float(
            f"{SVMRegressorWrapper.PREFIX}:tol",
            (1e-4, 1e-2),
            log=True,
            default=0.001,
        )
        gamma = Categorical(
            f"{SVMRegressorWrapper.PREFIX}:gamma",
            items=["scale", "auto"],
            default="scale",
        )
        C = Float(f"{SVMRegressorWrapper.PREFIX}:C", (1.0, 20), log=True, default=1.0)
        epsilon = Float(
            f"{SVMRegressorWrapper.PREFIX}:epsilon",
            (0.01, 0.99),
            log=True,
            default=0.1,
        )
        shrinking = Categorical(
            f"{SVMRegressorWrapper.PREFIX}:shrinking",
            items=[True, False],
            default=True,
        )

        cs.add([kernel, degree, coef0, tol, gamma, C, epsilon, shrinking])

        return cs

    @staticmethod
    def get_from_configuration(configuration, additional_params={}):
        svr_params = {
            "kernel": configuration[f"{SVMRegressorWrapper.PREFIX}:kernel"],
            "degree": configuration[f"{SVMRegressorWrapper.PREFIX}:degree"],
            "coef0": configuration[f"{SVMRegressorWrapper.PREFIX}:coef0"],
            "tol": configuration[f"{SVMRegressorWrapper.PREFIX}:tol"],
            "gamma": configuration[f"{SVMRegressorWrapper.PREFIX}:gamma"],
            "C": configuration[f"{SVMRegressorWrapper.PREFIX}:C"],
            "epsilon": configuration[f"{SVMRegressorWrapper.PREFIX}:epsilon"],
            "shrinking": configuration[f"{SVMRegressorWrapper.PREFIX}:shrinking"],
            **additional_params,
        }

        return partial(SVMRegressorWrapper, init_params=svr_params)
