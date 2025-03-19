from asf.preprocessing.abstrtract_preprocessor import AbstractPreprocessor
import sklearn.impute
import sklearn.preprocessing
import sklearn.decomposition
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def get_default_preprocessor(categorical_features=None, numerical_features=None):
    if categorical_features is None:
        categorical_features = make_column_selector(dtype_include=object)

    if numerical_features is None:
        numerical_features = make_column_selector(dtype_include="number")

    return ColumnTransformer(
        [
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                ),
                categorical_features,
            ),
            (
                "cont",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numerical_features,
            ),
        ]
    )


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
