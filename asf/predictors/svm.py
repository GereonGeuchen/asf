from asf.predictors.sklearn_wrapper import SklearnWrapper
from sklearn.svm import SVR, SVC
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical


class SVMClassifierWrapper(SklearnWrapper):
    PREFIX = "svm_classifier"

    def __init__(self, init_params: dict = {}):
        super().__init__(SVC, init_params)

    def get_configuration_space(self):
        cs = ConfigurationSpace(name="SVM")

        kernel = Categorical(
            f"{self.PREFIX}:kernel",
            choices=["linear", "rbf", "poly", "sigmoid"],
            default_value="rbf",
        )
        degree = Integer(f"{self.PREFIX}:degree", (1, 128), log=True, default_value=1)
        coef0 = Float(
            f"{self.PREFIX}:coef0",
            (-0.5, 0.5),
            log=False,
            default_value=0.49070634552851977,
        )
        tol = Float(
            f"{self.PREFIX}:tol",
            (1e-4, 1e-2),
            log=True,
            default_value=0.0002154969698207585,
        )
        gamma = Categorical(
            f"{self.PREFIX}:gamma", choices=["scale", "auto"], default_value="scale"
        )
        C = Float(
            f"{self.PREFIX}:C", (1.0, 20), log=True, default_value=3.2333262862494365
        )
        epsilon = Float(
            f"{self.PREFIX}:epsilon",
            (0.01, 0.99),
            log=True,
            default_value=0.14834562300010581,
        )
        shrinking = Categorical(
            f"{self.PREFIX}:shrinking", choices=[True, False], default_value=True
        )

        cs.add([kernel, degree, coef0, tol, gamma, C, epsilon, shrinking])

        return cs

    @staticmethod
    def get_from_configuration(configuration, additional_params={}):
        svm_params = {
            "kernel": configuration["svm:kernel"],
            "degree": configuration["svm:degree"],
            "coef0": configuration["svm:coef0"],
            "tol": configuration["svm:tol"],
            "gamma": configuration["svm:gamma"],
            "C": configuration["svm:C"],
            "epsilon": configuration["svm:epsilon"],
            "shrinking": configuration["svm:shrinking"],
            **additional_params,
        }

        return SVMClassifierWrapper(init_params=svm_params)


class SVMRegressorWrapper(SklearnWrapper):
    PREFIX = "svm_regressor"

    def __init__(self, init_params: dict = {}):
        super().__init__(SVR, init_params)

    def get_configuration_space(self):
        cs = ConfigurationSpace(name="SVM Regressor")

        kernel = Categorical(
            f"{self.PREFIX}:kernel",
            choices=["linear", "rbf", "poly", "sigmoid"],
            default_value="rbf",
        )
        degree = Integer(f"{self.PREFIX}:degree", (1, 128), log=True, default_value=1)
        coef0 = Float(f"{self.PREFIX}:coef0", (-0.5, 0.5), log=False, default_value=0.0)
        tol = Float(f"{self.PREFIX}:tol", (1e-4, 1e-2), log=True, default_value=0.001)
        gamma = Categorical(
            f"{self.PREFIX}:gamma", choices=["scale", "auto"], default_value="scale"
        )
        C = Float(f"{self.PREFIX}:C", (1.0, 20), log=True, default_value=1.0)
        epsilon = Float(
            f"{self.PREFIX}:epsilon", (0.01, 0.99), log=True, default_value=0.1
        )
        shrinking = Categorical(
            f"{self.PREFIX}:shrinking", choices=[True, False], default_value=True
        )

        cs.add([kernel, degree, coef0, tol, gamma, C, epsilon, shrinking])

        return cs

    @staticmethod
    def get_from_configuration(configuration, additional_params={}):
        svr_params = {
            "kernel": configuration["svm_regressor:kernel"],
            "degree": configuration["svm_regressor:degree"],
            "coef0": configuration["svm_regressor:coef0"],
            "tol": configuration["svm_regressor:tol"],
            "gamma": configuration["svm_regressor:gamma"],
            "C": configuration["svm_regressor:C"],
            "epsilon": configuration["svm_regressor:epsilon"],
            "shrinking": configuration["svm_regressor:shrinking"],
            **additional_params,
        }

        return SVMRegressorWrapper(init_params=svr_params)
