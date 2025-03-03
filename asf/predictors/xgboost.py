from ConfigSpace import ConfigurationSpace, Constant, Float, Integer
from xgboost import XGBRegressor, XGBClassifier

from asf.predictors.sklearn_wrapper import SklearnWrapper


class XGBoostClassifierWrapper(SklearnWrapper):
    PREFIX = "xgb_classifier"

    def __init__(self, init_params: dict = {}):
        super().__init__(XGBClassifier, init_params)

    def get_configuration_space(self):
        cs = ConfigurationSpace(name="XGBoost")

        booster = Constant(f"{self.PREFIX}:booster", "gbtree")
        max_depth = Integer(
            f"{self.PREFIX}:max_depth", (1, 20), log=False, default_value=13
        )
        min_child_weight = Integer(
            f"{self.PREFIX}:min_child_weight", (1, 100), log=True, default_value=39
        )
        colsample_bytree = Float(
            f"{self.PREFIX}:colsample_bytree",
            (0.0, 1.0),
            log=False,
            default_value=0.2545374925231651,
        )
        colsample_bylevel = Float(
            f"{self.PREFIX}:colsample_bylevel",
            (0.0, 1.0),
            log=False,
            default_value=0.6909224923784677,
        )
        lambda_param = Float(
            f"{self.PREFIX}:lambda",
            (0.001, 1000),
            log=True,
            default_value=31.393252465064943,
        )
        alpha = Float(
            f"{self.PREFIX}:alpha",
            (0.001, 1000),
            log=True,
            default_value=0.24167936088332426,
        )
        learning_rate = Float(
            f"{self.PREFIX}:learning_rate",
            (0.001, 0.1),
            log=True,
            default_value=0.008237525103357958,
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
            "booster": configuration["xgb:booster"],
            "max_depth": configuration["xgb:max_depth"],
            "min_child_weight": configuration["xgb:min_child_weight"],
            "colsample_bytree": configuration["xgb:colsample_bytree"],
            "colsample_bylevel": configuration["xgb:colsample_bylevel"],
            "lambda": configuration["xgb:lambda"],
            "alpha": configuration["xgb:alpha"],
            "learning_rate": configuration["xgb:learning_rate"],
            **additional_params,
        }

        return XGBoostClassifierWrapper(init_params=xgb_params)


class XGBoostRegressorWrapper(SklearnWrapper):
    PREFIX = "xgb_regressor"

    def __init__(self, init_params: dict = {}):
        super().__init__(XGBRegressor, init_params)

    def get_configuration_space(self):
        cs = ConfigurationSpace(name="XGBoostRegressor")

        booster = Constant(f"{self.PREFIX}:booster", "gbtree")
        max_depth = Integer(
            f"{self.PREFIX}:max_depth", (1, 20), log=False, default_value=13
        )
        min_child_weight = Integer(
            f"{self.PREFIX}:min_child_weight", (1, 100), log=True, default_value=39
        )
        colsample_bytree = Float(
            f"{self.PREFIX}:colsample_bytree",
            (0.0, 1.0),
            log=False,
            default_value=0.2545374925231651,
        )
        colsample_bylevel = Float(
            f"{self.PREFIX}:colsample_bylevel",
            (0.0, 1.0),
            log=False,
            default_value=0.6909224923784677,
        )
        lambda_param = Float(
            f"{self.PREFIX}:lambda",
            (0.001, 1000),
            log=True,
            default_value=31.393252465064943,
        )
        alpha = Float(
            f"{self.PREFIX}:alpha",
            (0.001, 1000),
            log=True,
            default_value=0.24167936088332426,
        )
        learning_rate = Float(
            f"{self.PREFIX}:learning_rate",
            (0.001, 0.1),
            log=True,
            default_value=0.008237525103357958,
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
            "booster": configuration["xgb:booster"],
            "max_depth": configuration["xgb:max_depth"],
            "min_child_weight": configuration["xgb:min_child_weight"],
            "colsample_bytree": configuration["xgb:colsample_bytree"],
            "colsample_bylevel": configuration["xgb:colsample_bylevel"],
            "lambda": configuration["xgb:lambda"],
            "alpha": configuration["xgb:alpha"],
            "learning_rate": configuration["xgb:learning_rate"],
            **additional_params,
        }

        return XGBoostRegressorWrapper(init_params=xgb_params)
