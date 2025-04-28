from asf.pre_selector.abstract_pre_selector import AbstractPreSelector
import pandas as pd
import numpy as np
from typing import Union, Callable


class MarginalContributionBasedPreSelector(AbstractPreSelector):
    """
    MarginalContributionBasedPreSelector is a pre-selector that selects algorithms based on their marginal contribution
    to the performance of the selected algorithms.
    """

    def __init__(self, metric: Callable, n_algorithms: int, maximize=False, **kwargs):
        """
        Initializes the MarginalContributionBasedPreSelector with the given configuration.

        Args:
            config (dict): Configuration for the pre-selector.
        """
        super().__init__(**kwargs)
        self.metric = metric
        self.n_algorithms = n_algorithms
        self.maximize = maximize

    def fit_transform(
        self, performance: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        if isinstance(performance, np.ndarray):
            performance_frame = pd.DataFrame(
                performance,
                columns=[f"Algorithm_{i}" for i in range(performance.shape[1])],
            )
            numpy = True
        else:
            performance_frame = performance
            numpy = False

        mcs = []
        total_performance = self.metric(performance_frame)
        for algorithm in performance_frame.columns:
            performance_without_algorithm = performance_frame.drop(columns=[algorithm])
            total_performance_without_algorithm = self.metric(
                performance_without_algorithm
            )
            marginal_contribution = (
                total_performance - total_performance_without_algorithm
                if self.maximize
                else total_performance_without_algorithm - total_performance
            )

            mcs.append((algorithm, marginal_contribution))
        mcs.sort(key=lambda x: x[1], reverse=True)
        selected_algorithms = [x[0] for x in mcs[: self.n_algorithms]]
        selected_performance = performance_frame[selected_algorithms]

        if numpy:
            selected_performance = selected_performance.values

        return selected_performance
