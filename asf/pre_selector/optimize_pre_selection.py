from asf.pre_selector.abstract_pre_selector import AbstractPreSelector
import pandas as pd
import numpy as np
from typing import Union, Callable
from functools import partial

try:
    import scipy.optimize

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class OptimizePreSelection(AbstractPreSelector):
    """
    MarginalContributionBasedPreSelector is a pre-selector that selects algorithms based on their marginal contribution
    to the performance of the selected algorithms.
    """

    def __init__(
        self,
        metric: Callable,
        n_algorithms: int,
        maximize=False,
        fmin_function: Union[str, Callable] = "SLSQP",
        **kwargs,
    ):
        """
        Initializes the MarginalContributionBasedPreSelector with the given configuration.

        Args:
            config (dict): Configuration for the pre-selector.
        """
        super().__init__(**kwargs)
        self.metric = metric
        self.n_algorithms = n_algorithms
        self.maximize = maximize

        if isinstance(fmin_function, str):
            if not SCIPY_AVAILABLE:
                raise ImportError(
                    "Scipy is not available. Please install scipy to use this feature."
                )

            else:
                self.fmin_function = partial(
                    scipy.optimize.minimize, method=fmin_function
                )

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

        def objective_function(x):
            selected_algorithms = performance_frame.columns[x.astype(bool)]
            performance_without_algorithm = performance_frame.drop(
                columns=selected_algorithms
            )
            total_performance_without_algorithm = self.metric(
                performance_without_algorithm
            )

            return (
                total_performance_without_algorithm
                if self.maximize
                else -total_performance_without_algorithm
            )

        initial_guess = np.zeros(performance_frame.shape[1])
        initial_guess[: self.n_algorithms] = 1
        bounds = [(0, 1) for _ in range(performance_frame.shape[1])]
        constraints = {
            "type": "eq",
            "fun": lambda x: np.sum(x) - self.n_algorithms,
        }
        result = self.fmin_function(
            objective_function,
            initial_guess,
            bounds=bounds,
            constraints=constraints,
        )
        selected_algorithms = performance_frame.columns[result.x.astype(bool)]
        selected_performance = performance_frame[selected_algorithms]
        if numpy:
            selected_performance = selected_performance.values
        if selected_performance.shape[1] < self.n_algorithms:
            raise ValueError(
                f"Selected performance has {selected_performance.shape[1]} algorithms, "
                f"but expected {self.n_algorithms}."
            )
        if selected_performance.shape[1] == 0:
            raise ValueError(
                f"Selected performance has 0 algorithms, "
                f"but expected {self.n_algorithms}."
            )
        if selected_performance.shape[1] > self.n_algorithms:
            raise ValueError(
                f"Selected performance has {selected_performance.shape[1]} algorithms, "
                f"but expected {self.n_algorithms}."
            )

        return selected_performance
