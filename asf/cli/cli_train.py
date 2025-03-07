#!/usr/bin/env python3
"""CLI entry point for training selectors."""

import argparse
from pathlib import Path
from functools import partial

import pandas as pd

from asf import selectors
from asf.scenario.scenario_metadata import SelectionScenarioMetadata

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
        "--budget",
        type=int,
        default=None,
        required=False,
        help="Budget for the solvers",
    )
    parser.add_argument(
        "--maximize",
        type=bool,
        default=False,
        required=False,
        help="Maximize the objective",
    )
    parser.add_argument("--performance-metric", type=str, default="", required=False)
    parser.add_argument(
        "--feature-data",
        type=Path,
        required=True,
        help="Path to feature data",
    )
    parser.add_argument(
        "--performance-data",
        type=Path,
        required=True,
        help="Path to performance data",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to save model",
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
        "--budget",
        str(selector.metadata.budget),
        "--maximize",
        str(selector.metadata.maximize),
        "--performance-metric",
        str(selector.metadata.performance_metric),
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

    # Parse selector in to variable
    selector_class = getattr(selectors, args.selector)
    model_class = getattr(sklearn.ensemble, args.model)

    # Parse training data into variables
    features: pd.DataFrame = pandas_read_map[args.feature_data.suffix](
        args.feature_data, index_col=0
    )
    performance_data: pd.DataFrame = pandas_read_map[args.performance_data.suffix](
        args.performance_data, index_col=0
    )
    # Parse metadata
    metadata = SelectionScenarioMetadata(
        performance_data.columns.to_list(),
        features.columns.to_list(),
        performance_metric=args.performance_metric,
        maximize=args.maximize,
        budget=args.budget,
    )
    selector = selector_class(model_class, metadata)
    selector.fit(features, performance_data)

    # Save the model to the specified path
    selector.save(args.model_path)
