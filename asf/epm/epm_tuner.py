from typing import Type

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold
from smac import HyperparameterOptimizationFacade, Scenario

from asf.epm.epm import EPM
from asf.normalization.normalizations import AbstractNormalization, LogNormalization
from asf.predictors.abstract_predictor import AbstractPredictor


class GroupKFoldShuffle(_BaseKFold):
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y=None, groups=None):
        # Find the unique groups in the dataset.
        unique_groups = np.unique(groups)

        # Shuffle the unique groups if shuffle is true.
        if self.shuffle:
            np.random.seed(self.random_state)
            unique_groups = np.random.permutation(unique_groups)

        # Split the shuffled groups into n_splits.
        split_groups = np.array_split(unique_groups, self.n_splits)

        # For each split, determine the train and test indices.
        for test_group_ids in split_groups:
            test_mask = np.isin(groups, test_group_ids)
            train_mask = ~test_mask

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            yield train_idx, test_idx


def tune_epm(
    X,
    y,
    model_class: Type[AbstractPredictor],
    normalization_class: Type[AbstractNormalization] = LogNormalization,
    features_preprocessing: str | TransformerMixin = "default",
    categorical_features: list = None,
    numerical_features: list = None,
    groups=None,
    cv: int = 5,
    timeout: int = 3600,
    runcount_limit: int = 100,
    output_dir: str = "./smac_output",
    seed=0,
    smac_metric=root_mean_squared_error,
    smac_scenario_kwargs: dict = {},
    smac_kwargs: dict = {},
    predictor_kwargs: dict = {},
):
    """
    Tune the EPM model using SMAC.

    Parameters:
    timeout: Time limit for tuning
    """
    scenario = Scenario(
        configspace=model_class.get_configuration_space(),
        n_trials=runcount_limit,
        walltime_limit=timeout,
        deterministic=True,
        output_directory=output_dir,
        seed=seed,
        **smac_scenario_kwargs,
    )

    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    def target_function(config, seed):
        if groups is not None:
            kfold = GroupKFoldShuffle(n_splits=cv, shuffle=True, random_state=seed)
        else:
            kfold = KFold(n_splits=cv, shuffle=True, random_state=seed)

        scores = []
        for train_idx, test_idx in kfold.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            epm = EPM(
                predictor_class=model_class,
                normalization_class=normalization_class,
                transform_back=True,
                predictor_config=config,
                predictor_kwargs=predictor_kwargs,
                features_preprocessing=features_preprocessing,
                categorical_features=categorical_features,
                numerical_features=numerical_features,
            )
            epm.fit(X_train, y_train)

            y_pred = epm.predict(X_test)
            score = smac_metric(y_test, y_pred)
            scores.append(score)

        return np.mean(scores)

    smac = HyperparameterOptimizationFacade(scenario, target_function, **smac_kwargs)
    best_config = smac.optimize()

    return EPM(
        predictor_class=model_class,
        normalization_class=normalization_class,
        transform_back=True,
        predictor_config=best_config,
        features_preprocessing=features_preprocessing,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
    )
