from asf.aslib_scenario import ASlibScenario
import numpy as np


class MultiClassClassifier:
    def __init__(self, model_class, algorithms):
        self.model = model_class
        self.classifier
        self.algorithms = algorithms

    def fit(self, scenario: ASlibScenario):
        feats = scenario.feature_data.values
        self.classifier = self.model()
        self.classifier.fit(feats, np.argmin(scenario.performance_data.values, axis=1))

    def predict(self, scenario: ASlibScenario):
        feats = scenario.feature_data.values
        
        predictions = self.classifier.predict(feats)

        return {instance_name: self.algorithms[predictions[i]] 
                for i, instance_name in enumerate(scenario.feature_data.index)}




