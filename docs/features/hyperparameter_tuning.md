# Tuning selectors

ASF supports easy tuning of selectors:
```python
features, performance = get_data()

# Setting configuration space manually
selector = tune_selector(
    features,
    performance,
    selector_class=[
        (PairwiseClassifier, {"model_class": [SVMClassifierWrapper]}),
        (PairwiseRegressor, {"model_class": [SVMRegressorWrapper]}),
    ],
    selector_kwargs={"budget": 5000},
    runcount_limit=10,
)

# Fit the selector to the data
selector.fit(features, performance)

predictions = selector.predict(features)

# Print the predictions
print(predictions)
```