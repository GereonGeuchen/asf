#!/usr/bin/env python3
"""CLI entry point for training selectors."""
import ast
import argparse
from pathlib import Path
from functools import partial

import pandas as pd

from asf import selectors
from asf.scenario.scenario_metadata import ScenarioMetadata

import sklearn

pandas_read_map = {
    ".csv": pd.read_csv,
    ".parquet": pd.read_parquet,
    ".json": pd.read_json,
    ".feather": pd.read_feather,
    ".hdf": pd.read_hdf,
    ".html": pd.read_html,
    ".xml": pd.read_xml,
}


def parser_function() -> argparse.ArgumentParser:
    """Define command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selector",
        choices=selectors.__implemented__,
        required=True,
        help="Selector to train",
    )
    parser.add_argument(
        "--model",
        default="RandomForestClassifier",
        help="Model to use for the selector. "
        "Make sure to specify as a an attribute of sklearn.ensemble.",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        type=str,
        default=None,
        help="Metadata for the selector, represented as dictionary.",
    )
    parser.add_argument(
        "--feature-data", type=Path, required=True, help="Path to feature data"
    )
    parser.add_argument(
        "--performance-data", type=Path, required=True, help="Path to performance data"
    )
    parser.add_argument(
        "--model-path", type=Path, required=True, help="Path to save model"
    )
    return parser


def build_cli_command(
    selector: selectors.AbstractModelBasedSelector,
    feature_data: Path,
    performance_data: Path,
    destination: Path,
) -> list[str]:
    """Build CLI command from variables for async jobs.

    Args:
        selector: Selector to train
        feature_data: Path to feature data DataFrame
        performance_data: Path to performance data DataFrame
        destination: Path to save model
    """
    model_class = (
        selector.model_class.args[0]
        if isinstance(selector.model_class, partial)
        else selector.model_class
    )
    return [
        "python",
        Path(__file__).absolute(),
        "--selector",
        type(selector).__name__,
        "--model",
        f"{model_class.__name__}",
        "--metadata",
        f'"{selector.metadata.to_dict()}"',
        "--feature-data",
        str(feature_data),
        "--performance-data",
        str(performance_data),
        "--model-path",
        str(destination),
    ]


if __name__ == "__main__":
    parser = parser_function()
    args = parser.parse_args()
    metadata = args.metadata

    if metadata:
        metadata = ScenarioMetadata(**ast.literal_eval(metadata))
    # Parse selector in to variable
    selector_class = getattr(selectors, args.selector)
    model_class = getattr(sklearn.ensemble, args.model)
    selector = selector_class(model_class, metadata)

    # Parse training data into variables
    features = pandas_read_map[args.feature_data.suffix](args.feature_data, index_col=0)
    performance_data = pandas_read_map[args.performance_data.suffix](
        args.performance_data, index_col=0
    )
    selector.fit(features, performance_data)

    # Save the model to the specified path
    selector.save(args.model_path)
