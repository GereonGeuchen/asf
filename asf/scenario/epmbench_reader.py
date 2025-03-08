import json
import os

import pandas as pd

from asf.scenario.scenario_metadata import PerformancePredictionScenarioMetadata


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
    metadata = PerformancePredictionScenarioMetadata(
        targets=metadata["targets"], features=metadata["features"]
    )
    data = pd.read_parquet(os.path.join(path, "data.parquet"))

    return metadata, data
