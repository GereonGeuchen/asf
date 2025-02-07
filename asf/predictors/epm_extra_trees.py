from sklearn.ensemble._forest import ExtraTreesRegressor
import numpy as np
from asf.predictors import AbstractPredictor


class EPMRandomForest(ExtraTreesRegressor, AbstractPredictor):
    """
    Implementation of random forest as done in the paper
    "Algorithm runtime prediction: Methods & evaluation" by Hutter, Xu, Hoos, and Leyton-Brown (2014).

    Methods
    -------
    fit(X, Y)
        Fit the model to the data.
    predict(X)
        Predict using the model.
    save(file_path)
        Save the model to a file.
    load(file_path)
        Load the model from a file.
    """

    def __init__(  # TODO check hparams
        self,
        *,
        log=False,
        **kwargs,
    ) -> None:
        super().__init__(
            **kwargs,
        )
        self.log = log

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like
            Target values.
        """
        assert sample_weight is None, "Sample weights are not supported"
        super().fit(X=X, y=y, sample_weight=sample_weight)

        self.trainX = X
        self.trainY = y
        if self.log:
            for tree, samples_idx in zip(self.estimators_, self.estimators_samples_):
                curX = X[samples_idx]
                curY = y[samples_idx]
                preds = tree.apply(curX)
                for k in np.unique(preds):
                    tree.tree_.value[k, 0, 0] = np.log(np.exp(curY[preds == k]).mean())

    def predict(self, X):
        """
        Predict using the model.

        Parameters
        ----------
        X : array-like
            Data to predict on.

        Returns
        -------
        array-like
            Predicted values.
        """
        preds = []
        for tree, samples_idx in zip(self.estimators_, self.estimators_samples_):
            preds.append(tree.predict(X))
        preds = np.array(preds).T

        means = preds.mean(axis=1)
        vars = preds.var(axis=1)

        return means.reshape(-1, 1), vars.reshape(-1, 1)

    def save(self, file_path: str):
        """
        Save the model to a file.

        Parameters
        ----------
        file_path : str
            Path to the file where the model will be saved.
        """
        import joblib

        joblib.dump(self, file_path)

    def load(self, file_path: str):
        """
        Load the model from a file.

        Parameters
        ----------
        file_path : str
            Path to the file from which the model will be loaded.

        Returns
        -------
        EPMRandomForest
            The loaded model.
        """
        import joblib

        return joblib.load(file_path)
