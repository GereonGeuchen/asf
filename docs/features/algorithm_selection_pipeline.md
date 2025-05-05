# Algorithm selection pipelines

ASF supports pipelines of selection, including algorithm pre-selection and preprocessing of features.

```python
features, performance = get_data()

selector = SelectorPipeline(
    selector=PairwiseClassifier(model_class=RandomForestClassifier),
    preprocessor=get_default_preprocessor(),
    algorithm_pre_selector=MarginalContributionBasedPreSelector(
        metric=virtual_best_solver, n_algorithms=2
    ),
)

# Fit the selector to the data
selector.fit(features, performance)

predictions = selector.predict(features)

# Print the predictions
print(predictions)
```