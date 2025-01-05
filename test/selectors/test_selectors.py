import pytest
import pandas as pd
from asf.selectors import (
    PairwiseClassifier,
    MultiClassClassifier,
    PairwiseRegressor,
    PerformanceModel,
)
from asf.scenario.scenario_metadata import ScenarioMetadata
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


@pytest.fixture
def dummy_performance():
    return pd.DataFrame(
        {
            "algo1": [1, 2, 3],
            "algo2": [2, 3, 4],
            "algo3": [3, 4, 5],
        }
    )


@pytest.fixture
def dummy_features():
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [2, 3, 4],
            "feature3": [3, 4, 5],
        }
    )


@pytest.fixture
def dummy_metadata():
    return ScenarioMetadata(
        algorithms=["algo1", "algo2", "algo3"],
        features=["feature1", "feature2", "feature3"],
        performance_metric="time",
        maximize=False,
        budget=None,
    )


def test_pairwise_classifier(dummy_performance, dummy_features, dummy_metadata):
    classifier = PairwiseClassifier(
        model_class=RandomForestClassifier, metadata=dummy_metadata
    )
    classifier.fit(dummy_features, dummy_performance)
    predictions = classifier.predict(dummy_features)
    assert len(predictions) == 3
    assert all([isinstance(v, list) for v in predictions.values()])
    assert all([len(v) == 1 for v in predictions.values()])


def test_multi_class_classifier(dummy_performance, dummy_features, dummy_metadata):
    classifier = MultiClassClassifier(
        model_class=RandomForestClassifier, metadata=dummy_metadata
    )
    classifier.fit(dummy_features, dummy_performance)
    predictions = classifier.predict(dummy_features)
    assert len(predictions) == 3
    assert all([isinstance(v, list) for v in predictions.values()])
    assert all([len(v) == 1 for v in predictions.values()])


def test_pairwise_regressor(dummy_performance, dummy_features, dummy_metadata):
    regressor = PairwiseRegressor(
        model_class=RandomForestRegressor, metadata=dummy_metadata
    )
    regressor.fit(dummy_features, dummy_performance)
    predictions = regressor.predict(dummy_features)
    assert len(predictions) == 3
    assert all([isinstance(v, list) for v in predictions.values()])
    assert all([len(v) == 1 for v in predictions.values()])


def test_performance_model(dummy_performance, dummy_features, dummy_metadata):
    model = PerformanceModel(model_class=RandomForestRegressor, metadata=dummy_metadata)
    model.fit(dummy_features, dummy_performance)
    predictions = model.predict(dummy_features)
    assert len(predictions) == 3
    assert all([isinstance(v, list) for v in predictions.values()]), predictions
    assert all([len(v) == 1 for v in predictions.values()])
