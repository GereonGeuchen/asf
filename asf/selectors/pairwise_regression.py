from asf.aslib_scenario import ASlibScenario

class PairwiseRegressor:
    def __init__(self, model_class, algorithms):
        self.model = model_class()
        self.regressors = []
        self.algorithms = algorithms

    def fit(self, scenario: ASlibScenario):
        feats = scenario.feature_data.values
        for i, algorithm in enumerate(self.algorithms):
            for other_algorithm in self.algorithms[i + 1:]:
                algo1_times = scenario.performance_data[algorithm]
                algo2_times = scenario.performance_data[other_algorithm]

                diffs = algo1_times - algo2_times
                self.model.fit(feats, diffs)
                self.regressors.append(self.model)

    def predict(self, scenario: ASlibScenario):
        feats = scenario.feature_data.values
        predictions_sum = {a: 0 for a in self.algorithms}
        for i, algorithm in enumerate(self.algorithms):
            for other_algorithm in self.algorithms[i + 1:]:
                prediction = self.model.predict(feats)
                
                predictions_sum[algorithm] += prediction
                predictions_sum[other_algorithm] -= prediction

        return [(max(predictions_sum.items(), key=lambda k: k[1])[0], scenario.algorithm_cutoff_time)]




