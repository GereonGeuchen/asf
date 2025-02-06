from asf.selectors.abstract_selector import AbstractSelector
from asf.predictors import SklearnWrapper
from xgboost import XGBModel
from sklearn.base import ClassifierMixin, RegressorMixin
from functools import partial


class AbstractModelBasedSelector(AbstractSelector):
    def __init__(self, model_class, metadata, hierarchical_generator=...):
        super().__init__(metadata, hierarchical_generator)

        if isinstance(model_class, type) and issubclass(
            model_class, (ClassifierMixin, RegressorMixin, XGBModel)
        ):
            self.model_class = partial(SklearnWrapper, model_class)
        else:
            self.model_class = model_class

    def save(self, path):
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path):
        import joblib

        return joblib.load(path)
