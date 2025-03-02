class AbstractPreprocessor:
    def __init__(self):
        pass

    def fit(self, data):
        raise NotImplementedError

    def transform(self, data):
        raise NotImplementedError
