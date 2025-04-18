import json
import os
import pickle

import pandas as pd


def read_epmbench_scenario(path, load_subsample=False):
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
    if "groups" in metadata:
        groups = data[metadata["groups"]]
        data.drop(columns=[metadata["groups"]], inplace=True)
    else:
        groups = None

    if load_subsample:
        with open(os.path.join(path, "subsamples.pkl"), "rb") as f:
            subsample_dict = pickle.load(f)

    if not load_subsample:
        return data, metadata["features"], metadata["targets"], groups, metadata
    else:
        return (
            data,
            metadata["features"],
            metadata["targets"],
            groups,
            metadata,
            subsample_dict,
        )


def get_cv_fold(data, fold, features, target, groups=None):
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
    train_idx = data["cv"] != fold
    test_idx = data["cv"] == fold

    train_data = data[train_idx]
    test_data = data[test_idx]

    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]

    if groups is not None:
        groups_train = groups[train_idx]
        groups_test = groups[test_idx]
    else:
        groups_train = None
        groups_test = None

    return X_train, y_train, X_test, y_test, groups_train, groups_test


def get_subsample(
    data, iter, subsample_size, features, target, subsample_dict, groups=None
):
    """
    Splits the data into training and testing sets based on the specified fold.

    Args:
        data (pd.DataFrame): The dataset.
        iter (int): The iteration number.
        features (list): List of feature names.
        targets (list): List of target names.

    Returns:
        tuple: A tuple containing the training and testing sets.
    """
    train_idx = subsample_dict["subsamples"][subsample_size][iter]
    test_idx = subsample_dict["test"]

    train_data = data.loc[train_idx]
    test_data = data.loc[test_idx]

    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]

    if groups is not None:
        groups_train = groups[train_idx]
        groups_test = groups[test_idx]
    else:
        groups_train = None
        groups_test = None

    return X_train, y_train, X_test, y_test, groups_train, groups_test
