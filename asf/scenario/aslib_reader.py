import os

import pandas as pd
from scipy.io.arff import loadarff

from asf import ScenarioMetadata

try:
    import yaml
    from yaml import SafeLoader as Loader

    ASLIB_AVAILABLE = True
except ImportError:
    ASLIB_AVAILABLE = False


def read_scenario(
    path: str, add_running_time_features: bool = True
) -> tuple[ScenarioMetadata, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read an ASlib scenario from a file.

    Args:
        path (str): The path to the ASlib scenario.
        add_running_time_features (bool, optional): Whether to add running time features. Defaults to True.

    Returns:
        tuple[ScenarioMetadata, pd.DataFrame, pd.DataFrame, pd.DataFrame]: The metadata, features, performance, and cross-validation data.
    """
    if not ASLIB_AVAILABLE:
        raise ImportError(
            "The aslib library is not available. Install it via 'pip install asf-lib[aslib]'."
        )

    description_path = os.path.join(path, "description.yaml")
    performance_path = os.path.join(path, "algorithm_runs.arff")
    features_path = os.path.join(path, "feature_values.arff")
    features_running_time = os.path.join(path, "feature_costs.arff")
    cv_path = os.path.join(path, "cv.arff")

    with open(description_path, "r") as f:
        description = yaml.load(f, Loader=Loader)

    algorithms = description["algorithms"]
    features = description["features"]
    performance_metric = description["performance_metric"]
    feature_groups = description["feature_groups"]
    maximize = description["maximize"]
    budget = description["budget"]

    metadata = ScenarioMetadata(
        algorithms=algorithms,
        features=features,
        performance_metric=performance_metric,
        feature_groups=feature_groups,
        maximize=maximize,
        budget=budget,
    )

    performance = loadarff(performance_path)
    performance = pd.DataFrame(performance[0])

    features = loadarff(features_path)
    features = pd.DataFrame(features[0])

    if add_running_time_features:
        features_running_time = loadarff(features_running_time)
        features_running_time = pd.DataFrame(features_running_time[0])

        features = pd.concat([features, features_running_time], axis=1)

    cv = loadarff(cv_path)
    cv = pd.DataFrame(cv[0])

    return metadata, features, performance, cv
