from asf.aslib_scenario import ASlibScenario
import numpy as np


class PerformancePredictor:
    def __init__(self, model_class, algorithms, use_multi_target=False, normalize="log"):
        self.model = model_class
        self.regressors = []
        self.algorithms = algorithms
        self.use_multi_target = use_multi_target
        self.normalize = normalize

    def fit(self, scenario: ASlibScenario):
        feats = scenario.feature_data.values
        performance_data = scenario.performance_data.values

        if self.normalize == "log":
            performance_data = np.log(performance_data + 1e-8)

        if self.use_multi_target:
            self.regressors = self.model()
            self.regressors.fit(feats, performance_data)
        else:
            for i, algorithm in enumerate(self.algorithms):
                algo_times = performance_data[:, i]

                cur_model = self.model()
                cur_model.fit(feats, algo_times)
                self.regressors.append(cur_model)

    def predict(self, scenario: ASlibScenario):
        feats = scenario.feature_data.values
        
        if self.use_multi_target:
            predictions = self.regressors.predict(feats)
        else:
            predictions_sum = np.zeros((feats.shape[0], len(self.algorithms)))
            for i, algorithm in enumerate(self.algorithms):
                prediction = self.regressors[i].predict(feats)
                
                predictions_sum[:, i] = prediction

            predictions = np.argmin(predictions_sum, axis=1)

        return {instance_name: self.algorithms[np.argmin(predictions[i])] 
                for i, instance_name in enumerate(scenario.feature_data.index)}




