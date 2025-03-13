import numpy as np
from sklearn.base import OneToOneFeatureMixin, TransformerMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
import scipy.stats
import scipy.special


class AbstractNormalization(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None, sample_weight=None):
        return self

    def transform(self, X):
        raise NotImplementedError

    def inverse_transform(self, X):
        raise NotImplementedError


class MinMaxNormalization(AbstractNormalization):
    def __init__(self, feature_range=(0, 1)):
        super().__init__()
        self.feature_range = feature_range

    def fit(self, X, y=None, sample_weight=None):
        self.min_max_scale = MinMaxScaler(feature_range=self.feature_range)
        self.min_max_scale.fit(X.reshape(-1, 1))
        return self

    def transform(self, X):
        return self.min_max_scale.transform(X.reshape(-1, 1)).reshape(-1)

    def inverse_transform(self, X):
        return self.min_max_scale.inverse_transform(X.reshape(-1, 1)).reshape(-1)


class ZScoreNormalization(AbstractNormalization):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None, sample_weight=None):
        self.scaler = StandardScaler()
        self.scaler.fit(X.reshape(-1, 1))
        return self

    def transform(self, X):
        return self.scaler.transform(X.reshape(-1, 1)).reshape(-1)

    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X.reshape(-1, 1)).reshape(-1)


class LogNormalization(AbstractNormalization):
    def __init__(self, base=10, eps=1e-6):
        super().__init__()
        self.base = base
        self.eps = eps

    def fit(self, X, y=None, sample_weight=None):
        if X.min() < 0:
            self.min_val = X.min()
            X = X + self.min_val + self.eps
        else:
            self.min_val = 0

        return self

    def transform(self, X):
        X = X + self.min_val + self.eps

        return np.log(X) / np.log(self.base)

    def inverse_transform(self, X):
        X = np.power(self.base, X)
        if self.min_val != 0:
            X = X - self.min_val - self.eps

        return X


class SqrtNormalization(AbstractNormalization):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def fit(self, X, y=None, sample_weight=None):
        if X.min() < 0:
            self.min_val = X.min()
            X = X + self.min_val + self.eps
        else:
            self.min_val = 0
        return self

    def transform(self, X):
        X = X + self.min_val + self.eps
        return np.sqrt(X)

    def inverse_transform(self, X):
        X = np.power(X, 2)
        if self.min_val != 0:
            X = X - self.min_val - self.eps
        return X


class InvSigmoidNormalization(AbstractNormalization):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None, sample_weight=None):
        self.min_max_scale = MinMaxScaler(feature_range=(1e-6, 1 - 1e-6))
        self.min_max_scale.fit(X)
        return self

    def transform(self, X):
        X = self.min_max_scale.transform(X.reshape(-1, 1)).reshape(-1)
        return np.log(X / (1 - X))

    def inverse_transform(self, X):
        X = scipy.special.expit(X)
        return self.min_max_scale.inverse_transform(X.reshape(-1, 1)).reshape(-1)


class NegExpNormalization(AbstractNormalization):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None, sample_weight=None):
        return self

    def transform(self, X):
        return np.exp(-X)

    def inverse_transform(self, X):
        return -np.log(X)


class DummyNormalization(AbstractNormalization):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None, sample_weight=None):
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


class BoxCoxNormalization(AbstractNormalization):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None, sample_weight=None):
        self.box_cox = PowerTransformer(method="yeo-johnson")
        self.box_cox.fit(X.reshape(-1, 1))
        return self

    def transform(self, X):
        return self.box_cox.transform(X.reshape(-1, 1)).reshape(-1)

    def inverse_transform(self, X):
        X = self.box_cox.inverse_transform(X.reshape(-1, 1)).reshape(-1)
        return X
