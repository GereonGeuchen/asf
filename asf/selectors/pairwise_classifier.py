from asf.aslib_scenario import ASlibScenario
import numpy as np


class PairwiseClassifier:
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

                diffs = algo1_times < algo2_times
                cur_model = self.model()
                cur_model.fit(feats, diffs)
                self.regressors.append(cur_model)

    def predict(self, scenario: ASlibScenario):
        feats = scenario.feature_data.values
        predictions_sum = np.zeros((feats.shape[0], len(self.algorithms)))
        for i, algorithm in enumerate(self.algorithms):
            for other_algorithm in self.algorithms[i + 1:]:
                prediction = self.model.predict(feats)
                
                predictions_sum[prediction == 1, algorithm] += 1
                predictions_sum[prediction == 0, other_algorithm] -= 1

        return {instance_name: self.algorithms[np.argmax(predictions_sum[i])] 
                for i, instance_name in enumerate(scenario.feature_data.index)}




