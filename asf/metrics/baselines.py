import pandas as pd


def single_best_solver(schedules, performance, maximize):
    """
    Selects the best solver for each instance.

    Args:
        schedules (pd.DataFrame): The schedules to evaluate.
        performance (pd.DataFrame): The performance data for the algorithms.
        maximize (bool): Whether to maximize or minimize the performance.

    Returns:
        pd.Series: The selected solvers.
    """
    perf_sum = performance.sum(axis=0)
    if maximize:
        return perf_sum.max()
    else:
        return perf_sum.min()


def virtual_best_solver(schedules, performance, maximize):
    """
    Selects the best solver for each instance.

    Args:
        schedules (pd.DataFrame): The schedules to evaluate.
        performance (pd.DataFrame): The performance data for the algorithms.
        maximize (bool): Whether to maximize or minimize the performance.

    Returns:
        pd.Series: The selected solvers.
    """
    if maximize:
        return performance.max(axis=1).sum()
    else:
        return performance.min(axis=1).sum()


def running_time_selector_performance(
    schedules, performance, budget, par=10, feature_time=None
):
    """
    Calculates the performance of a selector.

    Args:
        schedules (dict): The schedules to evaluate.
        performance (pd.DataFrame): The performance data for the algorithms.
        maximize (bool): Whether to maximize or minimize the performance.
        budget (float): The budget for the scenario.
        par (float): The penalization factor.
        feature_time (pd.DataFrame): The feature time data.

    Returns:
        float: The total running time.
    """
    if feature_time is None:
        feature_time = pd.DataFrame(
            0, index=performance.index, columns=["feature_time"]
        )
    total_time = {}
    for instance, schedule in schedules.items():
        allocated_times = {algorithm: 0 for algorithm in performance.columns}
        solved = False
        for algorithm, algo_budget in schedule:
            remaining_budget = (
                budget
                - sum(allocated_times.values())
                - feature_time.loc[instance].item()
            )
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
            total_time[instance] = (
                sum(allocated_times.values()) + feature_time.loc[instance].item()
            )
        else:
            total_time[instance] = budget * par
    return total_time


def running_time_closed_gap(schedules, performance, budget, par, feature_time):
    """
    Selects the best solver for each instance.

    Args:
        schedules (pd.DataFrame): The schedules to evaluate.
        performance (pd.DataFrame): The performance data for the algorithms.
        budget (float): The budget for the scenario.
        par (float): The penalization factor.

    Returns:
        float: The closed gap value.
    """
    sbs_val = single_best_solver(schedules, performance, False)
    vbs_val = virtual_best_solver(schedules, performance, False)
    s_val = sum(
        list(
            running_time_selector_performance(
                schedules, performance, budget, par, feature_time
            ).values()
        )
    )

    return (sbs_val - s_val) / (sbs_val - vbs_val)
