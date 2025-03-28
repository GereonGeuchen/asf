from asf.predictors.sklearn_wrapper import SklearnWrapper
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical
from functools import partial


class RandomForestClassifierWrapper(SklearnWrapper):
    PREFIX = "rf_classifier"

    def __init__(self, init_params: dict = {}):
        super().__init__(RandomForestClassifier, init_params)

    @staticmethod
    def get_configuration_space():
        cs = ConfigurationSpace(name="RandomForest")
        # NB 301
        n_estimators = Integer(
            f"{RandomForestClassifierWrapper.PREFIX}:n_estimators",
            (16, 128),
            log=True,
            default=116,
        )
        min_samples_split = Integer(
            f"{RandomForestClassifierWrapper.PREFIX}:min_samples_split",
            (2, 20),
            log=False,
            default=2,
        )
        min_samples_leaf = Integer(
            f"{RandomForestClassifierWrapper.PREFIX}:min_samples_leaf",
            (1, 20),
            log=False,
            default=2,
        )
        max_features = Float(
            f"{RandomForestClassifierWrapper.PREFIX}:max_features",
            (0.1, 1.0),
            log=False,
            default=0.17055852159745608,
        )
        bootstrap = Categorical(
            f"{RandomForestClassifierWrapper.PREFIX}:bootstrap",
            items=[True, False],
            default=False,
        )

        cs.add(
            [n_estimators, min_samples_split, min_samples_leaf, max_features, bootstrap]
        )

        return cs

    @staticmethod
    def get_from_configuration(configuration, additional_params={}):
        rf_params = {
            "n_estimators": configuration[
                f"{RandomForestClassifierWrapper.PREFIX}:n_estimators"
            ],
            "min_samples_split": configuration[
                f"{RandomForestClassifierWrapper.PREFIX}:min_samples_split"
            ],
            "min_samples_leaf": configuration[
                f"{RandomForestClassifierWrapper.PREFIX}:min_samples_leaf"
            ],
            "max_features": configuration[
                f"{RandomForestClassifierWrapper.PREFIX}:max_features"
            ],
            "bootstrap": configuration[
                f"{RandomForestClassifierWrapper.PREFIX}:bootstrap"
            ],
            **additional_params,
        }

        return partial(RandomForestClassifierWrapper, init_params=rf_params)


class RandomForestRegressorWrapper(SklearnWrapper):
    PREFIX = "rf_regressor"

    def __init__(self, init_params: dict = {}):
        super().__init__(RandomForestRegressor, init_params)

    def get_configuration_space():
        cs = ConfigurationSpace(name="RandomForestRegressor")

        n_estimators = Integer(
            f"{RandomForestRegressorWrapper.PREFIX}:n_estimators",
            (16, 128),
            log=True,
            default=116,
        )
        min_samples_split = Integer(
            f"{RandomForestRegressorWrapper.PREFIX}:min_samples_split",
            (2, 20),
            log=False,
            default=2,
        )
        min_samples_leaf = Integer(
            f"{RandomForestRegressorWrapper.PREFIX}:min_samples_leaf",
            (1, 20),
            log=False,
            default=2,
        )
        max_features = Float(
            f"{RandomForestRegressorWrapper.PREFIX}:max_features",
            (0.1, 1.0),
            log=False,
            default=0.17055852159745608,
        )
        bootstrap = Categorical(
            f"{RandomForestRegressorWrapper.PREFIX}:bootstrap",
            items=[True, False],
            default=False,
        )

        cs.add(
            [n_estimators, min_samples_split, min_samples_leaf, max_features, bootstrap]
        )

        return cs

    @staticmethod
    def get_from_configuration(configuration, additional_params={}):
        rf_params = {
            "n_estimators": configuration[
                f"{RandomForestRegressorWrapper.PREFIX}:n_estimators"
            ],
            "min_samples_split": configuration[
                f"{RandomForestRegressorWrapper.PREFIX}:min_samples_split"
            ],
            "min_samples_leaf": configuration[
                f"{RandomForestRegressorWrapper.PREFIX}:min_samples_leaf"
            ],
            "max_features": configuration[
                f"{RandomForestRegressorWrapper.PREFIX}:max_features"
            ],
            "bootstrap": configuration[
                f"{RandomForestRegressorWrapper.PREFIX}:bootstrap"
            ],
            **additional_params,
        }

        return partial(RandomForestRegressorWrapper, init_params=rf_params)
