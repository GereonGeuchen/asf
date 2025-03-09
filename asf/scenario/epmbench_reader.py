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

    return data, metadata["features"], metadata["targets"]


def get_cv_fold(data, fold, features, targets):
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
    train_data = data[data["fold"] != fold]
    test_data = data[data["fold"] == fold]

    X_train = train_data[features]
    y_train = train_data[targets]
    X_test = test_data[features]
    y_test = test_data[targets]

    return X_train, y_train, X_test, y_test
