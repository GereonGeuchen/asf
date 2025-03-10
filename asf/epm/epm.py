from typing import Type

from sklearn.model_selection import cross_val_score
from smac import HyperparameterOptimizationFacade, Scenario

from asf.normalization.normalizations import AbstractNormalization, LogNormalization
from asf.predictors.abstract_predictor import AbstractPredictor


class EPM:
    def __init__(
        self,
        predictor_class: Type[AbstractPredictor],
        normalization_class: Type[AbstractNormalization] = LogNormalization,
        transform_back: bool = True,
        predictor_config=None,
        predictor_kwargs=None,
    ):
        self.predictor_class = predictor_class
        self.normalization_class = normalization_class
        self.transform_back = transform_back
        self.predictor_config = predictor_config
        self.predictor_kwargs = predictor_kwargs or {}

    def fit(self, X, y, sample_weight=None):
        """
        Fit the EPM model to the data.

        Parameters:
        X: Features
        y: Target variable
        sample_weight: Sample weights (optional)
        """
        self.normalization = self.normalization_class()
        y = self.normalization.fit(y)

        if self.predictor_config is None:
            self.configuration_space = self.predictor_class()
        else:
            self.predictor = self.predictor_class.get_from_configuration(
                self.predictor_config, self.predictor_kwargs
            )

        self.predictor.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        """
        Predict using the fitted EPM model.

        Parameters:
        X: Features"
        "
        """
        y_pred = self.predictor.predict(X)

        if self.transform_back:
            y_pred = self.normalization.inverse_transform(y_pred)

        return y_pred

    
