from asf.preprocessing.abstrtract_preprocessor import AbstractPreprocessor
import sklearn.impute
import sklearn.preprocessing
import sklearn.decomposition
import pandas as pd


class SklearnPreprocessor(AbstractPreprocessor):
    def __init__(self, preprocessor, preprocessor_kwargs=None):
        self.preprocessor_class = preprocessor
        self.preprocessor_kwargs = preprocessor_kwargs

    def fit(self, data):
        self.preprocessor = self.preprocessor_class(**self.preprocessor_kwargs)
        self.preprocessor.fit(data.values)

    def transform(self, data):
        return pd.DataFrame(
            self.preprocessor.transform(data.values),
            columns=data.columns,
            index=data.index,
        )


class Imputer(SklearnPreprocessor):
    def __init__(self):
        super().__init__(preprocessor=sklearn.impute.SimpleImputer)


class PCA(SklearnPreprocessor):
    def __init__(self):
        super().__init__(preprocessor=sklearn.decomposition.PCA)
