from typing import Type

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold
from sklearn.metrics import root_mean_squared_error
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


def tune(
    X,
    y,
    model_class: Type[AbstractPredictor],
    normalization: Type[AbstractNormalization] = LogNormalization(),
    groups=None,
    cv: int = 5,
    timeout: int = 3600,
    runcount_limit: int = 100,
    output_dir: str = "./smac_output",
    seed=0,
    smac_metric=root_mean_squared_error,
    smac_scenario_kwargs: dict = {},
    smac_kwargs: dict = {},
):
    """
    Tune the EPM model using SMAC.

    Parameters:
    timeout: Time limit for tuning
    """
    scenario = Scenario(
        {
            "run_obj": "quality",
            "n_trials": runcount_limit,
            "wallclock_limit": timeout,
            "cs": model_class.get_configuration_space(),
            "deterministic": True,
            "output_dir": output_dir,
            "seed": seed,
            **smac_scenario_kwargs,
        }
    )

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
                normalization=normalization,
                transform_back=True,
                predictor_config=config,
            )
            epm.fit(X_train, y_train)

            y_pred = epm.predict(X_test)
            score = smac_metric(y_test, y_pred)
            scores.append(score)

        return -np.mean(scores) # sklearn metrics are maximising but smac is minimising

    smac = HyperparameterOptimizationFacade(scenario, target_function, **smac_kwargs)
    best_config = smac.optimize()

    return best_config
