import pandas as pd


class AbstractPresolver:
    def __init__(self, metadata, presolver_cutoff: int):
        self.metadata = metadata
        self.presolver_cutoff = presolver_cutoff

    def fit(self, features: pd.DataFrame, performance: pd.DataFrame):
        return self._fit(features, performance)

    def predict(self) -> dict[str, list[tuple[str, float]]]:
        return self._predict()
