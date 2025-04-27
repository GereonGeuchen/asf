from asf.preprocessing.abstrtract_preprocessor import AbstractPreprocessor
import sklearn.impute
import sklearn.preprocessing
import sklearn.decomposition
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from typing import Optional, Union, List, Callable


def get_default_preprocessor(
    categorical_features: Optional[Union[List[str], Callable]] = None,
    numerical_features: Optional[Union[List[str], Callable]] = None,
) -> ColumnTransformer:
    """
    Creates a default preprocessor for handling categorical and numerical features.

    Args:
        categorical_features (Optional[Union[List[str], Callable]]):
            List of categorical feature names or a callable selector. Defaults to selecting object dtype columns.
        numerical_features (Optional[Union[List[str], Callable]]):
            List of numerical feature names or a callable selector. Defaults to selecting numeric dtype columns.

    Returns:
        ColumnTransformer: A transformer that applies preprocessing pipelines to categorical and numerical features.
    """
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
    """
    A wrapper class for scikit-learn preprocessors to integrate with the AbstractPreprocessor interface.

    Attributes:
        preprocessor_class (Callable): The scikit-learn preprocessor class.
        preprocessor_kwargs (Optional[dict]): Keyword arguments to initialize the preprocessor.
    """

    def __init__(
        self, preprocessor: Callable, preprocessor_kwargs: Optional[dict] = None
    ):
        """
        Initializes the SklearnPreprocessor.

        Args:
            preprocessor (Callable): The scikit-learn preprocessor class.
            preprocessor_kwargs (Optional[dict]): Keyword arguments to initialize the preprocessor.
        """
        self.preprocessor_class = preprocessor
        self.preprocessor_kwargs = preprocessor_kwargs

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fits the preprocessor to the data.

        Args:
            data (pd.DataFrame): The input data to fit the preprocessor.
        """
        self.preprocessor = self.preprocessor_class(**self.preprocessor_kwargs)
        self.preprocessor.fit(data.values)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data using the fitted preprocessor.

        Args:
            data (pd.DataFrame): The input data to transform.

        Returns:
            pd.DataFrame: The transformed data as a DataFrame.
        """
        return pd.DataFrame(
            self.preprocessor.transform(data.values),
            columns=data.columns,
            index=data.index,
        )


class Imputer(SklearnPreprocessor):
    """
    A preprocessor class for imputing missing values using scikit-learn's SimpleImputer.
    """

    def __init__(self):
        """
        Initializes the Imputer with SimpleImputer as the preprocessor.
        """
        super().__init__(preprocessor=sklearn.impute.SimpleImputer)


class PCA(SklearnPreprocessor):
    """
    A preprocessor class for applying Principal Component Analysis (PCA) using scikit-learn's PCA.
    """

    def __init__(self):
        """
        Initializes the PCA preprocessor with scikit-learn's PCA.
        """
        super().__init__(preprocessor=sklearn.decomposition.PCA)
