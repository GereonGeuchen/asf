from typing import Any
from abc import ABC, abstractmethod


class AbstractPredictor(ABC):
    """
    Abstract base class for all predictors.

    Methods
    -------
    fit(X, Y)
        Fit the model to the data.
    predict(X)
        Predict using the model.
    save(file_path)
        Save the model to a file.
    load(file_path)
        Load the model from a file.
    """

    def __init__(self):
        """
        Initialize the predictor.
        """
        pass

    @abstractmethod
    def fit(self, X: Any, Y: Any, **kwargs):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like
            Training data.
        Y : array-like
            Target values.
        """
        pass

    @abstractmethod
    def predict(self, X: Any, **kwargs) -> Any:
        """
        Predict using the model.

        Parameters
        ----------
        X : array-like
            Data to predict on.

        Returns
        -------
        array-like
            Predicted values.
        """
        pass

    @abstractmethod
    def save(self, file_path: str):
        """
        Save the model to a file.

        Parameters
        ----------
        file_path : str
            Path to the file where the model will be saved.
        """
        pass

    @abstractmethod
    def load(self, file_path: str):
        """
        Load the model from a file.

        Parameters
        ----------
        file_path : str
            Path to the file from which the model will be loaded.
        """
        pass

    def get_configuration_space(self):
        """
        Get the configuration space for the predictor.

        Returns
        -------
        ConfigurationSpace
            The configuration space for the predictor.
        """
        raise NotImplementedError(
            "get_configuration_space() is not implemented for this predictor"
        )

    @staticmethod
    def get_from_configuration(configuration):
        """
        Get the configuration space for the predictor.

        Returns
        -------
        AbstractPredictor
            The predictor.
        """
        raise NotImplementedError(
            "get_from_configuration() is not implemented for this predictor"
        )
