from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from sklearn.svm import SVC, SVR
from functools import partial
from typing import Dict, Any, Optional

from asf.predictors.sklearn_wrapper import SklearnWrapper


class SVMClassifierWrapper(SklearnWrapper):
    """
    A wrapper for the Scikit-learn SVC (Support Vector Classifier) model.
    Provides methods to define a configuration space and create an instance
    of the classifier from a configuration.

    Attributes
    ----------
    PREFIX : str
        Prefix used for parameter names in the configuration space.
    """

    PREFIX = "svm_classifier"

    def __init__(self, init_params: Dict[str, Any] = {}):
        """
        Initialize the SVMClassifierWrapper.

        Parameters
        ----------
        init_params : dict, optional
            Dictionary of parameters to initialize the SVC model.
        """
        super().__init__(SVC, init_params)

    def get_configuration_space() -> ConfigurationSpace:
        """
        Define the configuration space for the SVM classifier.

        Returns
        -------
        ConfigurationSpace
            The configuration space containing hyperparameters for the SVM classifier.
        """
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
    def get_from_configuration(
        configuration: Dict[str, Any], additional_params: Optional[Dict[str, Any]] = {}
    ) -> partial:
        """
        Create an SVMClassifierWrapper instance from a configuration.

        Parameters
        ----------
        configuration : dict
            Dictionary containing the configuration parameters.
        additional_params : dict, optional
            Additional parameters to include in the model initialization.

        Returns
        -------
        partial
            A partial function to create an SVMClassifierWrapper instance.
        """
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
    """
    A wrapper for the Scikit-learn SVR (Support Vector Regressor) model.
    Provides methods to define a configuration space and create an instance
    of the regressor from a configuration.

    Attributes
    ----------
    PREFIX : str
        Prefix used for parameter names in the configuration space.
    """

    PREFIX = "svm_regressor"

    def __init__(self, init_params: Dict[str, Any] = {}):
        """
        Initialize the SVMRegressorWrapper.

        Parameters
        ----------
        init_params : dict, optional
            Dictionary of parameters to initialize the SVR model.
        """
        super().__init__(SVR, init_params)

    @staticmethod
    def get_configuration_space(
        cs: Optional[ConfigurationSpace] = None,
    ) -> ConfigurationSpace:
        """
        Define the configuration space for the SVM regressor.

        Parameters
        ----------
        cs : ConfigurationSpace, optional
            The configuration space to add the parameters to. If None, a new
            ConfigurationSpace will be created.

        Returns
        -------
        ConfigurationSpace
            The configuration space containing hyperparameters for the SVM regressor.
        """
        if cs is None:
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
    def get_from_configuration(
        configuration: Dict[str, Any], additional_params: Optional[Dict[str, Any]] = {}
    ) -> partial:
        """
        Create an SVMRegressorWrapper instance from a configuration.

        Parameters
        ----------
        configuration : dict
            Dictionary containing the configuration parameters.
        additional_params : dict, optional
            Additional parameters to include in the model initialization.

        Returns
        -------
        partial
            A partial function to create an SVMRegressorWrapper instance.
        """
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
