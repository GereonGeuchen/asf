from ConfigSpace import ConfigurationSpace, Constant, Float, Integer
from xgboost import XGBRegressor, XGBClassifier

from asf.predictors.sklearn_wrapper import SklearnWrapper
from functools import partial


class XGBoostClassifierWrapper(SklearnWrapper):
    PREFIX = "xgb_classifier"

    def __init__(self, init_params: dict = {}):
        super().__init__(XGBClassifier, init_params)

    @staticmethod
    def get_configuration_space():
        cs = ConfigurationSpace(name="XGBoost")
        # NB301
        booster = Constant(f"{XGBoostClassifierWrapper.PREFIX}:booster", "gbtree")
        max_depth = Integer(
            f"{XGBoostClassifierWrapper.PREFIX}:max_depth",
            (1, 20),
            log=False,
            default=13,
        )
        min_child_weight = Integer(
            f"{XGBoostClassifierWrapper.PREFIX}:min_child_weight",
            (1, 100),
            log=True,
            default=39,
        )
        colsample_bytree = Float(
            f"{XGBoostClassifierWrapper.PREFIX}:colsample_bytree",
            (0.0, 1.0),
            log=False,
            default=0.2545374925231651,
        )
        colsample_bylevel = Float(
            f"{XGBoostClassifierWrapper.PREFIX}:colsample_bylevel",
            (0.0, 1.0),
            log=False,
            default=0.6909224923784677,
        )
        lambda_param = Float(
            f"{XGBoostClassifierWrapper.PREFIX}:lambda",
            (0.001, 1000),
            log=True,
            default=31.393252465064943,
        )
        alpha = Float(
            f"{XGBoostClassifierWrapper.PREFIX}:alpha",
            (0.001, 1000),
            log=True,
            default=0.24167936088332426,
        )
        learning_rate = Float(
            f"{XGBoostClassifierWrapper.PREFIX}:learning_rate",
            (0.001, 0.1),
            log=True,
            default=0.008237525103357958,
        )

        cs.add(
            [
                booster,
                max_depth,
                min_child_weight,
                colsample_bytree,
                colsample_bylevel,
                lambda_param,
                alpha,
                learning_rate,
            ]
        )

        return cs

    @staticmethod
    def get_from_configuration(configuration, additional_params={}):
        xgb_params = {
            "booster": configuration[f"{XGBoostClassifierWrapper.PREFIX}:booster"],
            "max_depth": configuration[f"{XGBoostClassifierWrapper.PREFIX}:max_depth"],
            "min_child_weight": configuration[
                f"{XGBoostClassifierWrapper.PREFIX}:min_child_weight"
            ],
            "colsample_bytree": configuration[
                f"{XGBoostClassifierWrapper.PREFIX}:colsample_bytree"
            ],
            "colsample_bylevel": configuration[
                f"{XGBoostClassifierWrapper.PREFIX}:colsample_bylevel"
            ],
            "lambda": configuration[f"{XGBoostClassifierWrapper.PREFIX}:lambda"],
            "alpha": configuration[f"{XGBoostClassifierWrapper.PREFIX}:alpha"],
            "learning_rate": configuration[
                f"{XGBoostClassifierWrapper.PREFIX}:learning_rate"
            ],
            **additional_params,
        }

        return partial(XGBoostClassifierWrapper, init_params=xgb_params)


class XGBoostRegressorWrapper(SklearnWrapper):
    PREFIX = "xgb_regressor"

    def __init__(self, init_params: dict = {}):
        super().__init__(XGBRegressor, init_params)

    @staticmethod
    def get_configuration_space():
        cs = ConfigurationSpace(name="XGBoostRegressor")

        booster = Constant(f"{XGBoostRegressorWrapper.PREFIX}:booster", "gbtree")
        max_depth = Integer(
            f"{XGBoostRegressorWrapper.PREFIX}:max_depth",
            (1, 20),
            log=False,
            default=13,
        )
        min_child_weight = Integer(
            f"{XGBoostRegressorWrapper.PREFIX}:min_child_weight",
            (1, 100),
            log=True,
            default=39,
        )
        colsample_bytree = Float(
            f"{XGBoostRegressorWrapper.PREFIX}:colsample_bytree",
            (0.0, 1.0),
            log=False,
            default=0.2545374925231651,
        )
        colsample_bylevel = Float(
            f"{XGBoostRegressorWrapper.PREFIX}:colsample_bylevel",
            (0.0, 1.0),
            log=False,
            default=0.6909224923784677,
        )
        lambda_param = Float(
            f"{XGBoostRegressorWrapper.PREFIX}:lambda",
            (0.001, 1000),
            log=True,
            default=31.393252465064943,
        )
        alpha = Float(
            f"{XGBoostRegressorWrapper.PREFIX}:alpha",
            (0.001, 1000),
            log=True,
            default=0.24167936088332426,
        )
        learning_rate = Float(
            f"{XGBoostRegressorWrapper.PREFIX}:learning_rate",
            (0.001, 0.1),
            log=True,
            default=0.008237525103357958,
        )

        cs.add(
            [
                booster,
                max_depth,
                min_child_weight,
                colsample_bytree,
                colsample_bylevel,
                lambda_param,
                alpha,
                learning_rate,
            ]
        )

        return cs

    @staticmethod
    def get_from_configuration(configuration, additional_params={}):
        xgb_params = {
            "booster": configuration[f"{XGBoostRegressorWrapper.PREFIX}:booster"],
            "max_depth": configuration[f"{XGBoostRegressorWrapper.PREFIX}:max_depth"],
            "min_child_weight": configuration[
                f"{XGBoostRegressorWrapper.PREFIX}:min_child_weight"
            ],
            "colsample_bytree": configuration[
                f"{XGBoostRegressorWrapper.PREFIX}:colsample_bytree"
            ],
            "colsample_bylevel": configuration[
                f"{XGBoostRegressorWrapper.PREFIX}:colsample_bylevel"
            ],
            "lambda": configuration[f"{XGBoostRegressorWrapper.PREFIX}:lambda"],
            "alpha": configuration[f"{XGBoostRegressorWrapper.PREFIX}:alpha"],
            "learning_rate": configuration[
                f"{XGBoostRegressorWrapper.PREFIX}:learning_rate"
            ],
            **additional_params,
        }

        return partial(XGBoostRegressorWrapper, init_params=xgb_params)
