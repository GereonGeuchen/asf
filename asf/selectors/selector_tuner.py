import numpy as np
import pandas as pd
from ConfigSpace import Categorical, ConfigurationSpace
from sklearn.model_selection import KFold
from smac import HyperparameterOptimizationFacade, Scenario

from asf.metrics.baselines import running_time_selector_performance
from asf.preprocessing.abstrtract_preprocessor import AbstractPreprocessor
from asf.scenario.scenario_metadata import SelectionScenarioMetadata
from asf.selectors.abstract_selector import AbstractSelector
from asf.selectors.pairwise_classifier import PairwiseClassifier
from asf.selectors.pairwise_regressor import PairwiseRegressor
from asf.selectors.selector_pipeline import SelectorPipeline
from asf.utils.groupkfoldshuffle import GroupKFoldShuffle


def selector_tuner(
    X: pd.DataFrame,
    y: pd.DataFrame,
    metadata: SelectionScenarioMetadata,
    selector_class: AbstractSelector = [PairwiseClassifier, PairwiseRegressor],
    selector_space_kwargs: dict = {},
    selector_kwargs: dict = {},
    preprocessing_class: AbstractPreprocessor = None,
    pre_solving=None,
    feature_selector=None,
    algorithm_pre_selector=None,
    output_dir: str = "./smac_output",
    smac_metric=running_time_selector_performance,
    smac_kwargs: dict = {},
    smac_scenario_kwargs: dict = {},
    runcount_limit=100,
    timeout=None,
    seed=None,
    cv=10,
    groups=None,
):
    if type(selector_class) is not list:
        selector_class = [selector_class]

    cs = ConfigurationSpace()
    cs.add(
        Categorical(
            name="selector",
            choices=selector_class,
        )
    )
    for selector in selector_class:
        selector.get_configuration_space(cs=cs, **selector_space_kwargs)

    scenario = Scenario(
        configspace=cs,
        n_trials=runcount_limit,
        walltime_limit=timeout,
        deterministic=True,
        output_directory=output_dir,
        seed=seed,
        **smac_scenario_kwargs,
    )

    def target_function(config, seed):
        if groups is not None:
            kfold = GroupKFoldShuffle(n_splits=cv, shuffle=True, random_state=seed)
        else:
            kfold = KFold(n_splits=cv, shuffle=True, random_state=seed)

        scores = []
        for train_idx, test_idx in kfold.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            selector = SelectorPipeline(
                metadata=metadata,
                selector=config["selector"].get_from_configuration(
                    config, **selector_kwargs
                ),
                preprocessor=preprocessing_class,
                pre_solving=pre_solving,
                feature_selector=feature_selector,
                algorithm_pre_selector=algorithm_pre_selector,
            )
            selector.fit(X_train, y_train)

            y_pred = selector.predict(X_test)
            score = smac_metric(y_test, y_pred)
            scores.append(score)

        return np.mean(scores)

    smac = HyperparameterOptimizationFacade(scenario, target_function, **smac_kwargs)
    best_config = smac.optimize()

    del smac  # clean up SMAC to free memory and delete dask client
    return SelectorPipeline(
        metadata=metadata,
        selector=best_config["selector"].get_from_configuration(
            best_config, **selector_kwargs
        ),
        preprocessor=preprocessing_class,
        pre_solving=pre_solving,
        feature_selector=feature_selector,
        alggorithm_pre_selector=algorithm_pre_selector,
    )
