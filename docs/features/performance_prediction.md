# Empirical performance prediction

ASF allows to easily tune and train EPMs. For example, to tune an EPM:

```python
features, performance = get_data()

# Initialize the selector
epm = tune_epm(
    features,
    performance,
    model_class=RandomForestRegressorWrapper,
    features_preprocessing=None,
)

# Fit the selector to the data
epm.fit(features, performance)

predictions = epm.predict(features)

# Print the predictions
print(predictions)
```

By default, ASF uses log scaling of the performance. Other performance scaling methods include standarization, and inverse sigmoid. For more details check the [API](../api/preprocessing.md)