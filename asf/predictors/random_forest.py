from asf.predictors.sklearn_wrapper import SklearnWrapper
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical


class RandomForestClassifierWrapper(SklearnWrapper):
    PREFIX = "rf_classifier"

    def __init__(self, init_params: dict = {}):
        super().__init__(RandomForestClassifier, init_params)

    def get_configuration_space(self):
        cs = ConfigurationSpace(name="RandomForest")

        n_estimators = Integer(
            f"{self.PREFIX}:n_estimators", (16, 128), log=True, default_value=116
        )
        min_samples_split = Integer(
            f"{self.PREFIX}:min_samples_split", (2, 20), log=False, default_value=2
        )
        min_samples_leaf = Integer(
            f"{self.PREFIX}:min_samples_leaf", (1, 20), log=False, default_value=2
        )
        max_features = Float(
            f"{self.PREFIX}:max_features",
            (0.1, 1.0),
            log=False,
            default_value=0.17055852159745608,
        )
        bootstrap = Categorical(
            f"{self.PREFIX}:bootstrap", choices=[True, False], default_value=False
        )

        cs.add(
            [n_estimators, min_samples_split, min_samples_leaf, max_features, bootstrap]
        )

        return cs

    @staticmethod
    def get_from_configuration(configuration, additional_params={}):
        rf_params = {
            "n_estimators": configuration["rf:n_estimators"],
            "min_samples_split": configuration["rf:min_samples_split"],
            "min_samples_leaf": configuration["rf:min_samples_leaf"],
            "max_features": configuration["rf:max_features"],
            "bootstrap": configuration["rf:bootstrap"],
            **additional_params,
        }

        return RandomForestClassifierWrapper(init_params=rf_params)


class RandomForestRegressorWrapper(SklearnWrapper):
    PREFIX = "rf_regressor"

    def __init__(self, init_params: dict = {}):
        super().__init__(RandomForestRegressor, init_params)

    def get_configuration_space(self):
        cs = ConfigurationSpace(name="RandomForestRegressor")

        n_estimators = Integer(
            f"{self.PREFIX}:n_estimators", (16, 128), log=True, default_value=116
        )
        min_samples_split = Integer(
            f"{self.PREFIX}:min_samples_split", (2, 20), log=False, default_value=2
        )
        min_samples_leaf = Integer(
            f"{self.PREFIX}:min_samples_leaf", (1, 20), log=False, default_value=2
        )
        max_features = Float(
            f"{self.PREFIX}:max_features",
            (0.1, 1.0),
            log=False,
            default_value=0.17055852159745608,
        )
        bootstrap = Categorical(
            f"{self.PREFIX}:bootstrap", choices=[True, False], default_value=False
        )

        cs.add(
            [n_estimators, min_samples_split, min_samples_leaf, max_features, bootstrap]
        )

        return cs

    @staticmethod
    def get_from_configuration(configuration, additional_params={}):
        rf_params = {
            "n_estimators": configuration["rf:n_estimators"],
            "min_samples_split": configuration["rf:min_samples_split"],
            "min_samples_leaf": configuration["rf:min_samples_leaf"],
            "max_features": configuration["rf:max_features"],
            "bootstrap": configuration["rf:bootstrap"],
            **additional_params,
        }

        return RandomForestRegressorWrapper(init_params=rf_params)
