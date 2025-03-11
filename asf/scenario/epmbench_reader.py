import json
import os

import pandas as pd


def read_epmbench_scenario(path):
    """
    Reads the EPMBench scenario from the given path.

    Args:

        path (str): Path to the EPMBench scenario file.

        Returns:
        dict: A dictionary containing the scenario metadata.

    """
    with open(os.path.join(path, "metadata.json"), "r") as f:
        metadata = json.load(f)

    data = pd.read_parquet(os.path.join(path, "data.parquet"))
    if "instance_column" in metadata:
        instances = data[metadata["instance_column"]]
        data.drop(columns=[metadata["instance_column"]], inplace=True)
    else:
        instances = None

    return data, metadata["features"], metadata["targets"], instances


def get_cv_fold(data, fold, features, targets, instances=None):
    """
    Splits the data into training and testing sets based on the specified fold.

    Args:
        data (pd.DataFrame): The dataset.
        fold (int): The fold number.
        features (list): List of feature names.
        targets (list): List of target names.

    Returns:
        tuple: A tuple containing the training and testing sets.
    """
    train_idx = data["fold"] != fold
    test_idx = data["fold"] == fold

    train_data = data[train_idx]
    test_data = data[test_idx]

    X_train = train_data[features]
    y_train = train_data[targets]
    X_test = test_data[features]
    y_test = test_data[targets]

    if instances is not None:
        instances_train = instances[train_idx]
        instances_test = instances[test_idx]
    else:
        instances_train = None
        instances_test = None

    return X_train, y_train, X_test, y_test, instances_train, instances_test
