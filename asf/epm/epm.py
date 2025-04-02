from functools import partial
from typing import Type

import pandas as pd
from sklearn.base import RegressorMixin

from asf.normalization.normalizations import AbstractNormalization, LogNormalization
from asf.predictors import SklearnWrapper
from asf.preprocessing.sklearn_preprocessor import get_default_preprocessor
from sklearn.base import TransformerMixin
from asf.predictors.abstract_predictor import AbstractPredictor


class EPM:
    def __init__(
        self,
        predictor_class: Type[AbstractPredictor] | Type[RegressorMixin],
        normalization_class: Type[AbstractNormalization] = LogNormalization,
        transform_back: bool = True,
        features_preprocessing: str | TransformerMixin = "default",
        categorical_features: list = None,
        numerical_features: list = None,
        predictor_config=None,
        predictor_kwargs=None,
    ):
        if isinstance(predictor_class, type) and issubclass(
            predictor_class, (RegressorMixin)
        ):
            self.model_class = partial(SklearnWrapper, predictor_class)
        else:
            self.model_class = predictor_class

        self.predictor_class = predictor_class
        self.normalization_class = normalization_class
        self.transform_back = transform_back
        self.predictor_config = predictor_config
        self.predictor_kwargs = predictor_kwargs or {}

        if features_preprocessing == "default":
            self.features_preprocessing = get_default_preprocessor(
                categorical_features=categorical_features,
                numerical_features=numerical_features,
            )
        else:
            self.features_preprocessing = features_preprocessing

    def fit(self, X, y, sample_weight=None):
        """
        Fit the EPM model to the data.

        Parameters:
        X: Features
        y: Target variable
        sample_weight: Sample weights (optional)
        """
        if self.features_preprocessing is not None:
            X = self.features_preprocessing.fit_transform(X)

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.normalization = self.normalization_class()
        self.normalization.fit(y)
        y = self.normalization.transform(y)

        if self.predictor_config is None:
            self.predictor = self.predictor_class()
        else:
            self.predictor = self.predictor_class.get_from_configuration(
                self.predictor_config, self.predictor_kwargs
            )()

        self.predictor.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        """
        Predict using the fitted EPM model.

        Parameters:
        X: Features"
        "
        """
        if self.features_preprocessing is not None:
            X = self.features_preprocessing.transform(X)

        y_pred = self.predictor.predict(X)

        if self.transform_back:
            y_pred = self.normalization.inverse_transform(y_pred)

        return y_pred
