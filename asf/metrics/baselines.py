def single_best_solver(schedules, performance, metadata):
    """
    Selects the best solver for each instance.

    Args:
        schedules (pd.DataFrame): The schedules to evaluate.
        performance (pd.DataFrame): The performance data for the algorithms.
        metadata (SelectionScenarioMetadata): The metadata for the scenario.

    Returns:
        pd.Series: The selected solvers.
    """
    perf_sum = performance.sum(axis=0)
    if metadata.maximize:
        return perf_sum.min()
    else:
        return perf_sum.min()


def virtual_best_solver(schedules, performance, metadata):
    """
    Selects the best solver for each instance.

    Args:
        schedules (pd.DataFrame): The schedules to evaluate.
        performance (pd.DataFrame): The performance data for the algorithms.
        metadata (SelectionScenarioMetadata): The metadata for the scenario.

    Returns:
        pd.Series: The selected solvers.
    """
    if metadata.maximize:
        return performance.max(axis=1).sum()
    else:
        return performance.min(axis=1).sum()


def running_time_selector_performance(schedules, performance, metadata, par):
    """
    Calculates the performance of a selector.

    Args:
        schedules (dict): The schedules to evaluate.
        performance (pd.DataFrame): The performance data for the algorithms.
        metadata (SelectionScenarioMetadata): The metadata for the scenario.
        par (float): The penalization factor.

    Returns:
        float: The total running time.
    """
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
            total_time += metadata.budget * par
    return total_time


def running_time_closed_gap(schedules, performance, metadata, par):
    """
    Selects the best solver for each instance.

    Args:
        schedules (pd.DataFrame): The schedules to evaluate.
        performance (pd.DataFrame): The performance data for the algorithms.
        metadata (SelectionScenarioMetadata): The metadata for the scenario.
        par (float): The penalization factor.

    Returns:
        float: The closed gap value.
    """
    sbs_val = single_best_solver(schedules, performance, metadata)
    vbs_val = virtual_best_solver(schedules, performance, metadata)
    s_val = running_time_selector_performance(schedules, performance, metadata, par)

    return (sbs_val - s_val) / (sbs_val - vbs_val)
