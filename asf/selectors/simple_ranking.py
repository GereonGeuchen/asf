import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector


class SimpleRanking(AbstractModelBasedSelector):
    """
    Algorithm Selection via Ranking (Oentaryo et al.) + algo features (optional).
    Attributes:
        model_class: The class of the classification model to be used.
        metadata: Metadata containing information about the algorithms.
        classifier: The trained classification model.
    """

    def __init__(self, model_class, metadata, hierarchical_generator=None):
        """
        Initializes the MultiClassClassifier with the given parameters.

        Args:
            model_class: The class of the classification model to be used. Assumes XGBoost API.
            metadata: Metadata containing information about the algorithms.
            hierarchical_generator: Feature generator to be used.
        """
        AbstractModelBasedSelector.__init__(
            self, model_class, metadata, hierarchical_generator
        )
        self.classifier = None

    def _fit(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
    ):
        """
        Fits the classification model to the given feature and performance data.

        Args:
            features: DataFrame containing the feature data.
            performance: DataFrame containing the performance data.
        """
        if self.algorithm_features is None:
            encoder = OneHotEncoder()
            self.algorithm_features = pd.DataFrame(
                encoder.fit_transform(self.metadata.algorithms),
                index=self.metadata.algorithms,
            )

        total_features = pd.merge(
            features.reset_index(), self.algorithm_features.reset_index(), how="cross"
        )
        qid = total_features["index_x"]
        qid = pd.get_dummies(qid)
        total_features = pd.merge(
            features.reset_index(), self.algorithm_features.reset_index(), how="cross"
        )
        merged = total_features.merge(
            performance.stack().reset_index(),
            right_on=["level_0", "level_1"],
            left_on=["index_x", "index_y"],
            how="left",
        )
        merged["rank"] = merged.groupby("index_x").rank(ascending=True, method="min")[0]
        total_features = merged.drop(
            columns=["level_0", "level_1", 0, "index_x", "index_y"]
        )
        print(total_features)
        self.classifier = self.model_class()
        self.classifier.fit(
            total_features,
            merged["rank"],
            qid=qid,
        )

    def _predict(self, features: pd.DataFrame):
        """
        Predicts the best algorithm for each instance in the given feature data.

        Args:
            features: DataFrame containing the feature data.

        Returns:
            A dictionary mapping instance names to the predicted best algorithm.
        """
        predictions = self.classifier.predict(features)

        return {
            instance_name: self.metadata.algorithms[predictions[i]]
            for i, instance_name in enumerate(features.index)
        }
