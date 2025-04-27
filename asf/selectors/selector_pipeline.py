from asf.selectors.abstract_selector import AbstractSelector


class SelectorPipeline:
    def __init__(
        self,
        selector: AbstractSelector,
        preprocessor=None,
        pre_solving=None,
        feature_selector=None,
        algorithm_pre_selector=None,
        budget=None,
        maximize=False,
        feature_groups=None,
    ):
        self.selector = selector
        self.preprocessor = preprocessor
        self.pre_solving = pre_solving
        self.feature_selector = feature_selector
        self.algorithm_pre_selector = algorithm_pre_selector
        self.budget = budget
        self.maximize = maximize
        self.feature_groups = feature_groups

    def fit(self, X, y):
        if self.preprocessor:
            self.preprocessor = self.preprocessor()
            X = self.preprocessor.fit_transform(X)

        if self.algorithm_pre_selector:
            self.algorithm_pre_selector = self.algorithm_pre_selector()
            X, y = self.algorithm_pre_selector.fit_transform(X, y)

        if self.feature_selector:
            self.feature_selector = self.feature_selector()
            X, y = self.feature_selector.fit_transform(X, y)

        if self.pre_solving:
            self.pre_solving = self.pre_solving()
            self.pre_solving.fit(X, y)

        self.selector = self.selector(
            budget=self.budget,
            maximize=self.maximize,
            feature_groups=self.feature_groups,
        )
        self.selector.fit(X, y)

    def predict(self, X):
        if self.preprocessor:
            (X,) = self.preprocessor.transform(X)

        if self.pre_solving:
            X = self.pre_solving.transform(X)

        if self.feature_selector:
            X = self.feature_selector.transform(X)

        return self.selector.predict(X)

    def save(self, path):
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path):
        import joblib

        return joblib.load(path)
