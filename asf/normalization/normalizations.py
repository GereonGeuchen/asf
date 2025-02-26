import numpy as np
from sklearn.base import OneToOneFeatureMixin, TransformerMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
import scipy.stats
import scipy.special


class AbstractTransform(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None, sample_weight=None):
        return self

    def transform(self, X):
        raise NotImplementedError

    def inverse_transform(self, X):
        raise NotImplementedError


class MinMaxTransform(AbstractTransform):
    def __init__(self, feature_range=(0, 1)):
        super().__init__()
        self.feature_range = feature_range

    def fit(self, X, y=None, sample_weight=None):
        self.min_max_scale = MinMaxScaler(feature_range=self.feature_range)
        self.min_max_scale.fit(X)
        return self

    def transform(self, X):
        return self.min_max_scale.transform(X)

    def inverse_transform(self, X):
        return self.min_max_scale.inverse_transform(X)


class ZScoreTransform(AbstractTransform):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None, sample_weight=None):
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)

    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)


class LogTransform(AbstractTransform):
    def __init__(self, base=10, normalize=True, eps=1e-6):
        super().__init__()
        self.base = base
        self.normalize = normalize
        self.eps = eps

    def fit(self, X, y=None, sample_weight=None):
        if self.normalize:
            self.min_max_scale = MinMaxScaler(feature_range=(self.eps, 1 - self.eps))
            self.min_max_scale.fit(X)

        return self

    def transform(self, X):
        if self.normalize:
            X = self.min_max_scale.transform(X)

        return np.log(X) / np.log(self.base)

    def inverse_transform(self, X):
        X = np.power(self.base, X)
        if self.normalize:
            X = self.min_max_scale.inverse_transform(X)

        return X


class SqrtTransform(AbstractTransform):
    def __init__(self, normalize=True, eps=1e-6):
        super().__init__()
        self.normalize = normalize
        self.eps = eps

    def fit(self, X, y=None, sample_weight=None):
        if self.normalize:
            self.min_max_scale = MinMaxScaler(feature_range=(self.eps, 1 - self.eps))
            self.min_max_scale.fit(X)
        return self

    def transform(self, X):
        if self.normalize:
            X = self.min_max_scale.transform(X)
        return np.sqrt(X)

    def inverse_transform(self, X):
        X = np.power(X, 2)
        if self.normalize:
            X = self.min_max_scale.inverse_transform(X)
        return X


class InvSigmoidTransform(AbstractTransform):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None, sample_weight=None):
        self.min_max_scale = MinMaxScaler(feature_range=(1e-6, 1 - 1e-6))
        self.min_max_scale.fit(X)
        return self

    def transform(self, X):
        X = self.min_max_scale.transform(X)
        return np.log(X / (1 - X))

    def inverse_transform(self, X):
        X = scipy.special.expit(X)
        return self.min_max_scale.inverse_transform(X)


class NegExpTransform(AbstractTransform):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None, sample_weight=None):
        return self

    def transform(self, X):
        return np.exp(-X)

    def inverse_transform(self, X):
        return -np.log(X)


class DummyTransform(AbstractTransform):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None, sample_weight=None):
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


class BoxCoxTransform(AbstractTransform):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None, sample_weight=None):
        self.box_cox = PowerTransformer(method="yeo-johnson")
        self.box_cox.fit(X)
        return self

    def transform(self, X):
        return self.box_cox.transform(X)

    def inverse_transform(self, X):
        X = self.box_cox.inverse_transform(X)
        return X
