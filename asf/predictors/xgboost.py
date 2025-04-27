from ConfigSpace import ConfigurationSpace, Constant, Float, Integer
from typing import Optional, Dict, Any, Callable
from functools import partial

try:
    from xgboost import XGBRegressor, XGBClassifier, XGBRanker

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from asf.predictors.sklearn_wrapper import SklearnWrapper


class XGBoostClassifierWrapper(SklearnWrapper):
    """
    Wrapper for the XGBoost classifier to integrate with the ASF framework.
    """

    PREFIX: str = "xgb_classifier"

    def __init__(self, init_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the XGBoostClassifierWrapper.

        Parameters
        ----------
        init_params : dict, optional
            Initialization parameters for the XGBoost classifier.
        """
        if not XGB_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Please install it using pip install asf-lib[xgb]."
            )
        super().__init__(XGBClassifier, init_params or {})

    @staticmethod
    def get_configuration_space(
        cs: Optional[ConfigurationSpace] = None,
    ) -> ConfigurationSpace:
        """
        Get the configuration space for the XGBoost classifier.

        Parameters
        ----------
        cs : ConfigurationSpace, optional
            The configuration space to add the parameters to. If None, a new ConfigurationSpace will be created.

        Returns
        -------
        ConfigurationSpace
            The configuration space with the XGBoost parameters.
        """
        if cs is None:
            cs = ConfigurationSpace(name="XGBoost")

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
    def get_from_configuration(
        configuration: Dict[str, Any],
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Callable[..., "XGBoostClassifierWrapper"]:
        """
        Create an XGBoostClassifierWrapper from a configuration.

        Parameters
        ----------
        configuration : dict
            The configuration dictionary.
        additional_params : dict, optional
            Additional parameters to include in the configuration.

        Returns
        -------
        Callable[..., XGBoostClassifierWrapper]
            A callable that initializes the wrapper with the given configuration.
        """
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
            **(additional_params or {}),
        }

        return partial(XGBoostClassifierWrapper, init_params=xgb_params)


class XGBoostRegressorWrapper(SklearnWrapper):
    """
    Wrapper for the XGBoost regressor to integrate with the ASF framework.
    """

    PREFIX: str = "xgb_regressor"

    def __init__(self, init_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the XGBoostRegressorWrapper.

        Parameters
        ----------
        init_params : dict, optional
            Initialization parameters for the XGBoost regressor.
        """
        super().__init__(XGBRegressor, init_params or {})

    @staticmethod
    def get_configuration_space(
        cs: Optional[ConfigurationSpace] = None,
    ) -> ConfigurationSpace:
        """
        Get the configuration space for the XGBoost regressor.

        Parameters
        ----------
        cs : ConfigurationSpace, optional
            The configuration space to add the parameters to. If None, a new ConfigurationSpace will be created.

        Returns
        -------
        ConfigurationSpace
            The configuration space with the XGBoost parameters.
        """
        if cs is None:
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
    def get_from_configuration(
        configuration: Dict[str, Any],
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Callable[..., "XGBoostRegressorWrapper"]:
        """
        Create an XGBoostRegressorWrapper from a configuration.

        Parameters
        ----------
        configuration : dict
            The configuration dictionary.
        additional_params : dict, optional
            Additional parameters to include in the configuration.

        Returns
        -------
        Callable[..., XGBoostRegressorWrapper]
            A callable that initializes the wrapper with the given configuration.
        """
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
            **(additional_params or {}),
        }

        return partial(XGBoostRegressorWrapper, init_params=xgb_params)


class XGBoostRankerWrapper(SklearnWrapper):
    """
    Wrapper for the XGBoost ranker to integrate with the ASF framework.
    """

    PREFIX: str = "xgb_ranker"

    def __init__(self, init_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the XGBoostRankerWrapper.

        Parameters
        ----------
        init_params : dict, optional
            Initialization parameters for the XGBoost ranker.
        """
        super().__init__(XGBRanker, init_params or {})

    @staticmethod
    def get_configuration_space(
        cs: Optional[ConfigurationSpace] = None,
    ) -> ConfigurationSpace:
        """
        Get the configuration space for the XGBoost ranker.

        Parameters
        ----------
        cs : ConfigurationSpace, optional
            The configuration space to add the parameters to. If None, a new ConfigurationSpace will be created.

        Returns
        -------
        ConfigurationSpace
            The configuration space with the XGBoost parameters.
        """
        if cs is None:
            cs = ConfigurationSpace(name="XGBoostRanker")

        booster = Constant(f"{XGBoostRankerWrapper.PREFIX}:booster", "gbtree")
        max_depth = Integer(
            f"{XGBoostRankerWrapper.PREFIX}:max_depth",
            (1, 20),
            log=False,
            default=13,
        )
        min_child_weight = Integer(
            f"{XGBoostRankerWrapper.PREFIX}:min_child_weight",
            (1, 100),
            log=True,
            default=39,
        )
        colsample_bytree = Float(
            f"{XGBoostRankerWrapper.PREFIX}:colsample_bytree",
            (0.0, 1.0),
            log=False,
            default=0.2545374925231651,
        )
        colsample_bylevel = Float(
            f"{XGBoostRankerWrapper.PREFIX}:colsample_bylevel",
            (0.0, 1.0),
            log=False,
            default=0.6909224923784677,
        )
        lambda_param = Float(
            f"{XGBoostRankerWrapper.PREFIX}:lambda",
            (0.001, 1000),
            log=True,
            default=31.393252465064943,
        )
        alpha = Float(
            f"{XGBoostRankerWrapper.PREFIX}:alpha",
            (0.001, 1000),
            log=True,
            default=0.24167936088332426,
        )
        learning_rate = Float(
            f"{XGBoostRankerWrapper.PREFIX}:learning_rate",
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
    def get_from_configuration(
        configuration: Dict[str, Any],
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Callable[..., "XGBoostRankerWrapper"]:
        """
        Create an XGBoostRankerWrapper from a configuration.

        Parameters
        ----------
        configuration : dict
            The configuration dictionary.
        additional_params : dict, optional
            Additional parameters to include in the configuration.

        Returns
        -------
        Callable[..., XGBoostRankerWrapper]
            A callable that initializes the wrapper with the given configuration.
        """
        xgb_params = {
            "booster": configuration[f"{XGBoostRankerWrapper.PREFIX}:booster"],
            "max_depth": configuration[f"{XGBoostRankerWrapper.PREFIX}:max_depth"],
            "min_child_weight": configuration[
                f"{XGBoostRankerWrapper.PREFIX}:min_child_weight"
            ],
            "colsample_bytree": configuration[
                f"{XGBoostRankerWrapper.PREFIX}:colsample_bytree"
            ],
            "colsample_bylevel": configuration[
                f"{XGBoostRankerWrapper.PREFIX}:colsample_bylevel"
            ],
            "lambda": configuration[f"{XGBoostRankerWrapper.PREFIX}:lambda"],
            "alpha": configuration[f"{XGBoostRankerWrapper.PREFIX}:alpha"],
            "learning_rate": configuration[
                f"{XGBoostRankerWrapper.PREFIX}:learning_rate"
            ],
            **(additional_params or {}),
        }

        return partial(XGBoostRankerWrapper, init_params=xgb_params)
