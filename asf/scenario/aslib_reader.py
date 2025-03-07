import os

import pandas as pd

from asf import SelectionScenarioMetadata

try:
    import yaml
    from yaml import SafeLoader as Loader

    from arff import load

    ASLIB_AVAILABLE = True
except ImportError:
    ASLIB_AVAILABLE = False


def read_scenario(
    path: str, add_running_time_features: bool = True
) -> tuple[SelectionScenarioMetadata, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read an ASlib scenario from a file.

    Args:
        path (str): The path to the ASlib scenario.
        add_running_time_features (bool, optional): Whether to add running time features. Defaults to True.

    Returns:
        tuple[SelectionScenarioMetadata, pd.DataFrame, pd.DataFrame, pd.DataFrame]: The metadata, features, performance, and cross-validation data.
    """
    if not ASLIB_AVAILABLE:
        raise ImportError(
            "The aslib library is not available. Install it via 'pip install asf-lib[aslib]'."
        )

    description_path = os.path.join(path, "description.txt")
    performance_path = os.path.join(path, "algorithm_runs.arff")
    features_path = os.path.join(path, "feature_values.arff")
    features_running_time = os.path.join(path, "feature_costs.arff")
    cv_path = os.path.join(path, "cv.arff")

    with open(description_path, "r") as f:
        description = yaml.load(f, Loader=Loader)

    algorithms = list(description["metainfo_algorithms"].keys())
    features = description["features_deterministic"]
    performance_metric = description["performance_measures"][0]
    feature_groups = description["feature_steps"]
    maximize = description["maximize"][0]
    budget = description["algorithm_cutoff_time"]

    metadata = SelectionScenarioMetadata(
        algorithms=algorithms,
        algorith_features=None,
        features=features,
        performance_metric=performance_metric,
        feature_groups=feature_groups,
        maximize=maximize,
        budget=budget,
    )

    with open(performance_path, "r") as f:
        performance = load(f)
    performance = pd.DataFrame(
        performance["data"], columns=[a[0] for a in performance["attributes"]]
    )
    performance = performance.set_index("instance_id")
    performance = performance.pivot(columns="algorithm", values="runtime")

    with open(features_path, "r") as f:
        features = load(f)
    features = pd.DataFrame(
        features["data"], columns=[a[0] for a in features["attributes"]]
    )
    features = features.set_index("instance_id")

    if add_running_time_features:
        with open(features_running_time, "r") as f:
            features_running_time = load(f)
        features_running_time = pd.DataFrame(
            features_running_time["data"],
            columns=[a[0] for a in features_running_time["attributes"]],
        )
        features_running_time = features_running_time.set_index("instance_id")

        features = pd.concat([features, features_running_time], axis=1)

    with open(cv_path, "r") as f:
        cv = load(f)
    cv = pd.DataFrame(cv["data"], columns=[a[0] for a in cv["attributes"]])

    cv = cv.set_index("instance_id")

    features = features.sort_index()
    performance = performance.sort_index()
    cv = cv.sort_index()
    return metadata, features, performance, cv
