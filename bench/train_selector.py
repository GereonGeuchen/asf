import asf.scenario.aslib_reader as aslib_reader
from asf.selectors.abstract_selector import AbstractSelector
from asf.selectors import PairwiseClassifier, PairwiseRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from asf.metrics import RunningTimeClosedGap


def train_selector(scenario_name: str, selector_class: AbstractSelector, model) -> None:
    """
    Train a selector using 10-fold cross-validation on the given scenario.

    Parameters:
    scenario_name (str): The name of the scenario to load.
    selector_class (Type[AbstractSelector]): The class of the selector to be trained.

    Returns:
    None
    """
    # Load the scenario
    metadata, features, performance, cv = aslib_reader.read_scenario(scenario_name)

    performance[performance >= metadata.budget] = metadata.budget * 10
    solvers_performance = performance.sum(axis=0).sort_values(ascending=True)
    best_solvers = solvers_performance.index[:5]
    metadata.algorithms = best_solvers
    # Initialize the selector
    selector = selector_class(model, metadata)

    total_score = 0
    # Perform 10-fold cross-validation
    for i in range(10):
        X_train, X_test = features[cv["fold"] != i + 1], features[cv["fold"] == i + 1]
        y_train, y_test = (
            performance[cv["fold"] != i + 1],
            performance[cv["fold"] == i + 1],
        )

        # Train the selector
        selector.fit(X_train, y_train)
        schedules = selector.predict(X_test)
        # Evaluate the selector
        cg = RunningTimeClosedGap(10)
        score = cg(schedules, y_test, metadata)
        total_score += score
        print(f"Fold score: {score}")

    return total_score / 10


if __name__ == "__main__":
    print(
        train_selector(
            "bench/aslib_data/SAT12-ALL", PairwiseClassifier, RandomForestClassifier
        )
    )
    print(
        train_selector(
            "bench/aslib_data/SAT12-ALL", PairwiseRegressor, RandomForestRegressor
        )
    )
