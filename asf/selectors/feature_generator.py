import pandas as pd


class AbstractFeatureGenerator:
    def __init__(self, metadata: pd.DataFrame):
        self.metadata = metadata

    def generate_features(self, base_features) -> pd.DataFrame:
        pass


class DummyFeatureGenerator(AbstractFeatureGenerator):
    def generate_features(self, base_features) -> pd.DataFrame:
        return None
