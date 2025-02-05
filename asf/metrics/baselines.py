from asf.metrics.abstract_metric import AbstractMetric


class SingleBestSolver(AbstractMetric):
    """
    SingleBestSolver is a metric that selects the best solver for each instance.
    """

    def __call__(self, schedules, performance, metadata):
        """
        Selects the best solver for each instance.

        Args:
            schedules (pd.DataFrame): The schedules to evaluate.
            performance (pd.DataFrame): The performance data for the algorithms.
            metadata (ScenarioMetadata): The metadata for the scenario.

        Returns:
            pd.Series: The selected solvers.
        """
        perf_sum = performance.sum(axis=0)
        if metadata.maximize:
            return perf_sum.min() / len(schedules)
        else:
            return perf_sum.min() / len(schedules)


class VirtualBestSolver(AbstractMetric):
    """
    VirtualBestSolver is a metric that selects the best solver for each instance.
    """

    def __call__(self, schedules, performance, metadata):
        """
        Selects the best solver for each instance.

        Args:
            schedules (pd.DataFrame): The schedules to evaluate.
            performance (pd.DataFrame): The performance data for the algorithms.
            metadata (ScenarioMetadata): The metadata for the scenario.

        Returns:
            pd.Series: The selected solvers.
        """
        if metadata.maximize:
            return performance.max(axis=1).sum() / len(schedules)
        else:
            return performance.min(axis=1).sum() / len(schedules)


class RunningTimeSelectorPerformance(AbstractMetric):
    """
    Calculates the performance of a selector.
    """

    def __init__(self, par):
        super().__init__()
        self.par = par

    def __call__(self, schedules, performance, metadata):
        total_time = 0
        for instance, schedule in schedules.items():
            allocated_times = {algorithm: 0 for algorithm in metadata.algorithms}
            solved = False
            for algorithm, algo_budget in schedule:
                remaining_budget = metadata.budget - sum(allocated_times.values())
                remaining_time_to_solve = performance.loc[instance, algorithm] - (
                    algo_budget + allocated_times[algorithm]
                )
                if remaining_time_to_solve < 0:
                    allocated_times[algorithm] = performance.loc[instance, algorithm]
                    solved = True
                    break
                elif remaining_time_to_solve <= remaining_budget:
                    allocated_times[algorithm] += remaining_time_to_solve
                else:
                    allocated_times[algorithm] += remaining_budget
                    break
            if solved:
                total_time += sum(allocated_times.values())
            else:
                total_time += metadata.budget * self.par
        return total_time / len(schedules)


class RunningTimeClosedGap(AbstractMetric):
    """
    ClosedGap is a metric that selects the best solver for each instance.
    """

    def __init__(self, par):
        self.sbs = SingleBestSolver()
        self.vbs = VirtualBestSolver()
        self.selector_perf = RunningTimeSelectorPerformance(par)

    def __call__(self, schedules, performance, metadata):
        """
        Selects the best solver for each instance.

        Args:
            schedules (pd.DataFrame): The schedules to evaluate.
            performance (pd.DataFrame): The performance data for the algorithms.
            metadata (ScenarioMetadata): The metadata for the scenario.

        Returns:
            pd.Series: The selected solvers.
        """
        sbs_val = self.sbs(schedules, performance, metadata)
        vbs_val = self.vbs(schedules, performance, metadata)
        s_val = self.selector_perf(schedules, performance, metadata)

        return (sbs_val - s_val) / (sbs_val - vbs_val)
